"""Evaluation framework for agent harness benchmark results."""

import json
import os
import subprocess
from datetime import datetime


def evaluate_task(workspace: str, task: dict, agent_result: dict) -> dict:
    """Evaluate a completed task.

    Returns scoring dict with:
        file_score: fraction of expected files created
        check_score: fraction of checks passed
        tests_pass: whether tests pass
        code_runs: whether main code runs without errors
        details: per-check results
    """
    # Collect all files in workspace
    files = {}
    for root, dirs, filenames in os.walk(workspace):
        for fname in filenames:
            if fname.startswith('.') or fname.startswith('_exec_tmp'):
                continue
            fpath = os.path.join(root, fname)
            relpath = os.path.relpath(fpath, workspace)
            try:
                with open(fpath) as f:
                    files[relpath] = f.read()
            except:
                files[relpath] = ""

    # Score: expected files
    expected = task.get("expected_files", [])
    found_expected = sum(1 for ef in expected if any(ef in f for f in files))
    file_score = found_expected / len(expected) if expected else 1.0

    # Score: checks
    checks = task.get("checks", {})
    check_results = {}
    passed = 0
    for check_name, check_fn in checks.items():
        try:
            result = check_fn(files)
            check_results[check_name] = result
            if result:
                passed += 1
        except Exception as e:
            check_results[check_name] = False

    check_score = passed / len(checks) if checks else 1.0

    # Score: do tests pass?
    tests_pass = False
    test_output = ""
    test_files = [f for f in files if "test_" in f and f.endswith(".py")]
    if test_files:
        try:
            # Install any requirements first
            req_file = os.path.join(workspace, "requirements.txt")
            if os.path.exists(req_file):
                subprocess.run(
                    ["pip", "install", "-r", req_file, "-q"],
                    capture_output=True, timeout=30, cwd=workspace,
                )

            # Try pytest first, fall back to unittest
            result = subprocess.run(
                ["python3", "-m", "pytest", "-v", "--tb=short"] + [
                    os.path.join(workspace, tf) for tf in test_files
                ],
                capture_output=True, text=True,
                timeout=60, cwd=workspace,
            )
            if result.returncode != 0 and "No module named pytest" in (result.stdout + result.stderr):
                # Fall back to unittest
                result = subprocess.run(
                    ["python3", "-m", "unittest", "discover", "-s", workspace, "-p", "test_*.py", "-v"],
                    capture_output=True, text=True,
                    timeout=60, cwd=workspace,
                )
            test_output = result.stdout + result.stderr
            tests_pass = result.returncode == 0
        except subprocess.TimeoutExpired:
            test_output = "Tests timed out (60s)"
        except Exception as e:
            test_output = f"Test error: {e}"

    # Score: does main code run?
    code_runs = False
    main_files = [f for f in files if f.endswith(".py") and "test_" not in f]
    if main_files:
        try:
            result = subprocess.run(
                ["python3", "-c", f"import importlib.util; spec = importlib.util.spec_from_file_location('m', '{os.path.join(workspace, main_files[0])}'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"],
                capture_output=True, text=True,
                timeout=10, cwd=workspace,
            )
            code_runs = result.returncode == 0
        except:
            pass

    return {
        "file_score": file_score,
        "files_found": list(files.keys()),
        "files_expected": expected,
        "check_score": check_score,
        "check_results": check_results,
        "tests_pass": tests_pass,
        "test_output": test_output[:1000],
        "code_runs": code_runs,
        "iterations": agent_result.get("iterations", 0),
        "tool_calls": len(agent_result.get("tool_calls", [])),
        "total_tokens": agent_result.get("total_tokens", 0),
        "total_time": agent_result.get("total_time", 0),
        "agent_success": agent_result.get("success", False),
    }


def compute_overall_score(eval_result: dict) -> float:
    """Compute weighted overall score (0-100)."""
    weights = {
        "file_score": 15,      # Did it create the right files?
        "check_score": 35,     # Did the code have the right features?
        "tests_pass": 30,      # Do tests actually pass?
        "code_runs": 10,       # Does the code at least import?
        "agent_success": 10,   # Did it finish within iteration limit?
    }

    score = 0
    for key, weight in weights.items():
        val = eval_result.get(key, 0)
        if isinstance(val, bool):
            val = 1.0 if val else 0.0
        score += val * weight

    return round(score, 1)


def generate_report(results: dict, output_path: str):
    """Generate markdown comparison report across models."""
    lines = [
        f"# Agent Harness Benchmark Report",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
    ]

    # Summary table
    models = list(results.keys())
    tasks = list(next(iter(results.values())).keys()) if results else []

    lines.append("## Summary")
    lines.append("")
    header = "| Task |" + "|".join(f" {m[:20]} " for m in models) + "|"
    sep = "|------|" + "|".join("-----:" for _ in models) + "|"
    lines.append(header)
    lines.append(sep)

    model_totals = {m: [] for m in models}
    for task_name in tasks:
        row = f"| {task_name} |"
        for model in models:
            if task_name in results[model]:
                score = compute_overall_score(results[model][task_name])
                model_totals[model].append(score)
                row += f" {score} |"
            else:
                row += " - |"
        lines.append(row)

    # Average row
    avg_row = "| **Average** |"
    for model in models:
        scores = model_totals[model]
        avg = sum(scores) / len(scores) if scores else 0
        avg_row += f" **{avg:.1f}** |"
    lines.append(avg_row)

    # Detail sections
    lines.append("")
    lines.append("## Detail")
    lines.append("")
    for model in models:
        lines.append(f"### {model}")
        lines.append("")
        for task_name, eval_result in results[model].items():
            score = compute_overall_score(eval_result)
            lines.append(f"**{task_name}** — {score}/100")
            lines.append(f"- Files: {eval_result['check_score']*100:.0f}% checks | Tests: {'PASS' if eval_result['tests_pass'] else 'FAIL'}")
            lines.append(f"- Iterations: {eval_result['iterations']} | Tool calls: {eval_result['tool_calls']} | Tokens: {eval_result['total_tokens']} | Time: {eval_result['total_time']:.0f}s")
            lines.append("")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)

    return report
