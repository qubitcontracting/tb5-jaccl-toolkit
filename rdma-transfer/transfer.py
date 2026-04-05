#!/usr/bin/env python3
"""
RDMA File Transfer over JACCL (Thunderbolt 5)

Transfers files between nodes using MLX's JACCL RDMA backend.
Expected throughput: ~3.5 GB/s (vs rsync's ~252 MB/s over SSH).

Usage:
  Sender:   python3 transfer.py send <file_or_dir> --rank 0 --peer-rank 1
  Receiver: python3 transfer.py recv <output_dir>  --rank 1 --peer-rank 0

Both sides must run simultaneously. Uses the same JACCL device matrix
and coordinator as Exo.
"""

import argparse
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np


# JACCL config
COORDINATOR_PORT = 5556  # Different from Exo's 5555
DEVICE_MATRIX_PATH = "/tmp/jaccl_3node.json"
CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB chunks for RDMA transfer


def setup_jaccl(rank: int, world_size: int, coordinator_ip: str, device_matrix: str = None):
    """Initialize JACCL distributed group."""
    os.environ["MLX_RANK"] = str(rank)
    os.environ["MLX_JACCL_COORDINATOR"] = f"{coordinator_ip}:{COORDINATOR_PORT}"

    dm = device_matrix or DEVICE_MATRIX_PATH
    if os.path.exists(dm):
        os.environ["MLX_IBV_DEVICES"] = dm

    group = mx.distributed.init(backend="jaccl", strict=True)
    print(f"JACCL initialized: rank={group.rank()}, size={group.size()}")
    return group


# Use CPU stream to avoid Metal GPU timeout on large transfers
CPU_STREAM = mx.cpu


def file_to_chunks(filepath: str, chunk_size: int):
    """Read file and yield chunks as numpy arrays."""
    file_size = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        offset = 0
        while offset < file_size:
            data = f.read(chunk_size)
            # Pad to 4-byte alignment for MLX
            pad_len = (4 - len(data) % 4) % 4
            if pad_len:
                data += b"\x00" * pad_len
            arr = np.frombuffer(data, dtype=np.uint8)
            yield mx.array(arr), len(data) - pad_len  # actual_size excludes padding
            offset += chunk_size


def send_metadata(group, files_meta: list, dst_rank: int):
    """Send file manifest to receiver via RDMA."""
    meta_json = json.dumps(files_meta).encode("utf-8")
    # Send length first
    length_arr = mx.array(np.array([len(meta_json)], dtype=np.int64))
    result = mx.distributed.send(length_arr, dst=dst_rank, group=group, stream=CPU_STREAM)
    mx.eval(result)
    # Send metadata as uint8 array
    pad_len = (4 - len(meta_json) % 4) % 4
    meta_bytes = meta_json + b"\x00" * pad_len
    meta_arr = mx.array(np.frombuffer(meta_bytes, dtype=np.uint8))
    result = mx.distributed.send(meta_arr, dst=dst_rank, group=group, stream=CPU_STREAM)
    mx.eval(result)


def recv_metadata(group, peer_rank: int) -> list:
    """Receive file manifest from sender via RDMA."""
    length_arr = mx.distributed.recv((1,), mx.int64, src=peer_rank, group=group, stream=CPU_STREAM)
    mx.eval(length_arr)
    meta_len = int(length_arr[0].item())
    # Receive metadata
    padded_len = meta_len + (4 - meta_len % 4) % 4
    meta_arr = mx.distributed.recv((padded_len,), mx.uint8, src=peer_rank, group=group, stream=CPU_STREAM)
    mx.eval(meta_arr)
    meta_bytes = bytes(np.array(meta_arr[:meta_len]))
    return json.loads(meta_bytes.decode("utf-8"))


def send_file(group, filepath: str, dst_rank: int):
    """Send a single file over RDMA."""
    file_size = os.path.getsize(filepath)
    num_chunks = (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    sent = 0
    t0 = time.time()

    for i, (chunk, actual_size) in enumerate(file_to_chunks(filepath, CHUNK_SIZE)):
        # Send chunk size
        size_arr = mx.array(np.array([actual_size, len(chunk)], dtype=np.int64))
        result = mx.distributed.send(size_arr, dst=dst_rank, group=group, stream=CPU_STREAM)
        mx.eval(result)
        # Send chunk data
        result = mx.distributed.send(chunk, dst=dst_rank, group=group, stream=CPU_STREAM)
        mx.eval(result)
        sent += actual_size
        elapsed = time.time() - t0
        speed = sent / elapsed / 1e9 if elapsed > 0 else 0
        print(f"\r  [{i+1}/{num_chunks}] {sent/1e9:.1f}/{file_size/1e9:.1f} GB  "
              f"{speed:.2f} GB/s", end="", flush=True)

    # Send zero-length to signal end of file
    end_arr = mx.array(np.array([0, 0], dtype=np.int64))
    result = mx.distributed.send(end_arr, dst=dst_rank, group=group, stream=CPU_STREAM)
    mx.eval(result)
    print()
    return sent, time.time() - t0


def recv_file(group, filepath: str, expected_size: int, src_rank: int):
    """Receive a single file over RDMA."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    received = 0
    t0 = time.time()
    num_chunks = (expected_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    chunk_i = 0

    with open(filepath, "wb") as f:
        while True:
            # Receive chunk size
            size_arr = mx.distributed.recv((2,), mx.int64, src=src_rank, group=group, stream=CPU_STREAM)
            mx.eval(size_arr)
            actual_size = int(size_arr[0].item())
            arr_size = int(size_arr[1].item())

            if actual_size == 0:
                break

            # Receive chunk data
            chunk = mx.distributed.recv((arr_size,), mx.uint8, src=src_rank, group=group, stream=CPU_STREAM)
            mx.eval(chunk)
            data = bytes(np.array(chunk[:actual_size]))
            f.write(data)
            received += actual_size
            chunk_i += 1
            elapsed = time.time() - t0
            speed = received / elapsed / 1e9 if elapsed > 0 else 0
            print(f"\r  [{chunk_i}/{num_chunks}] {received/1e9:.1f}/{expected_size/1e9:.1f} GB  "
                  f"{speed:.2f} GB/s", end="", flush=True)

    print()
    return received, time.time() - t0


def scan_directory(path: str, base: str = None) -> list:
    """Scan directory and return file manifest."""
    if base is None:
        base = path
    files = []
    for entry in sorted(os.scandir(path), key=lambda e: e.name):
        if entry.is_file(follow_symlinks=False):
            rel = os.path.relpath(entry.path, base)
            files.append({
                "path": rel,
                "size": entry.stat().st_size,
            })
        elif entry.is_dir(follow_symlinks=False):
            files.extend(scan_directory(entry.path, base))
    return files


def cmd_send(args):
    """Send files to a peer."""
    source = os.path.abspath(args.path)
    if not os.path.exists(source):
        print(f"Error: {source} does not exist")
        sys.exit(1)

    # Build file manifest
    if os.path.isfile(source):
        files_meta = [{"path": os.path.basename(source), "size": os.path.getsize(source)}]
        base_dir = os.path.dirname(source)
    else:
        files_meta = scan_directory(source)
        base_dir = source

    total_size = sum(f["size"] for f in files_meta)
    print(f"Sending {len(files_meta)} files ({total_size/1e9:.1f} GB)")

    # Determine coordinator IP
    coord_ip = "0.0.0.0" if args.rank == 0 else args.coordinator
    group = setup_jaccl(args.rank, 2, coord_ip, args.device_matrix)

    # Send manifest
    print("Sending file manifest...")
    send_metadata(group, files_meta, dst_rank=args.peer_rank)

    # Send each file
    total_sent = 0
    t0 = time.time()
    for i, meta in enumerate(files_meta):
        filepath = os.path.join(base_dir, meta["path"])
        print(f"[{i+1}/{len(files_meta)}] {meta['path']} ({meta['size']/1e9:.2f} GB)")
        sent, _ = send_file(group, filepath, dst_rank=args.peer_rank)
        total_sent += sent

    elapsed = time.time() - t0
    print(f"\nDone: {total_sent/1e9:.1f} GB in {elapsed:.1f}s "
          f"({total_sent/elapsed/1e9:.2f} GB/s average)")


def cmd_recv(args):
    """Receive files from a peer."""
    output_dir = os.path.abspath(args.path)
    os.makedirs(output_dir, exist_ok=True)

    # Determine coordinator IP
    coord_ip = "0.0.0.0" if args.rank == 0 else args.coordinator
    group = setup_jaccl(args.rank, 2, coord_ip, args.device_matrix)

    # Receive manifest
    print("Waiting for file manifest...")
    files_meta = recv_metadata(group, peer_rank=args.peer_rank)
    total_size = sum(f["size"] for f in files_meta)
    print(f"Receiving {len(files_meta)} files ({total_size/1e9:.1f} GB)")

    # Receive each file
    total_recv = 0
    t0 = time.time()
    for i, meta in enumerate(files_meta):
        filepath = os.path.join(output_dir, meta["path"])
        print(f"[{i+1}/{len(files_meta)}] {meta['path']} ({meta['size']/1e9:.2f} GB)")
        recvd, _ = recv_file(group, filepath, meta["size"], src_rank=args.peer_rank)
        total_recv += recvd

    elapsed = time.time() - t0
    print(f"\nDone: {total_recv/1e9:.1f} GB in {elapsed:.1f}s "
          f"({total_recv/elapsed/1e9:.2f} GB/s average)")


def main():
    parser = argparse.ArgumentParser(description="RDMA File Transfer over JACCL")
    parser.add_argument("--rank", type=int, required=True, help="This node's rank")
    parser.add_argument("--peer-rank", type=int, required=True, help="Peer node's rank")
    parser.add_argument("--coordinator", default="192.168.0.126",
                        help="Coordinator IP (rank 0's LAN IP, default: vader)")
    parser.add_argument("--device-matrix", default=DEVICE_MATRIX_PATH,
                        help="JACCL device matrix JSON path")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"Transfer chunk size in bytes (default: {CHUNK_SIZE})")

    sub = parser.add_subparsers(dest="command", required=True)

    send_p = sub.add_parser("send", help="Send files to peer")
    send_p.add_argument("path", help="File or directory to send")

    recv_p = sub.add_parser("recv", help="Receive files from peer")
    recv_p.add_argument("path", help="Output directory")

    args = parser.parse_args()


    if args.command == "send":
        cmd_send(args)
    elif args.command == "recv":
        cmd_recv(args)


if __name__ == "__main__":
    main()
