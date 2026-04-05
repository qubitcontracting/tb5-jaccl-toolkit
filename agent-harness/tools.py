"""Tool implementations for the agent harness."""

import subprocess
import tempfile
import os
import json
import requests

WORKSPACE = None  # Set per-task


def set_workspace(path: str):
    global WORKSPACE
    WORKSPACE = path
    os.makedirs(path, exist_ok=True)


class MCPClient:
    """Simple MCP client that talks to mcp-proxy HTTP bridges."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session_id = None
        self._initialized = False

    def _headers(self):
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.session_id:
            h["mcp-session-id"] = self.session_id
        return h

    def _ensure_init(self):
        if self._initialized:
            return
        try:
            r = requests.post(f"{self.base_url}/mcp", headers=self._headers(), json={
                "jsonrpc": "2.0", "id": 0, "method": "initialize",
                "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                           "clientInfo": {"name": "agent-harness", "version": "1.0"}}
            }, timeout=15)
            self.session_id = r.headers.get("mcp-session-id", "")
            requests.post(f"{self.base_url}/mcp", headers=self._headers(), json={
                "jsonrpc": "2.0", "method": "notifications/initialized", "params": {}
            }, timeout=5)
            self._initialized = True
        except Exception as e:
            print(f"MCP init failed for {self.base_url}: {e}")

    def call_tool(self, name: str, arguments: dict) -> str:
        self._ensure_init()
        try:
            r = requests.post(f"{self.base_url}/mcp", headers=self._headers(), json={
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": name, "arguments": arguments}
            }, timeout=30)
            result = r.json().get("result", {})
            content = result.get("content", [])
            return "\n".join(c.get("text", "") for c in content) if content else str(result)
        except Exception as e:
            return f"MCP call failed: {e}"


# MCP proxy clients (start with: mcp-proxy --port 3008 -- npx -y docfork)
_docfork = MCPClient("http://localhost:3008")
_context7 = MCPClient("http://localhost:3009")


def search_docs(library: str, query: str) -> str:
    """Search library documentation via Docfork and Context7 MCP proxies."""
    results = []

    # Try Docfork (higher quality)
    try:
        doc = _docfork.call_tool("search_docs", {"library": library, "query": query})
        if doc and "failed" not in doc.lower():
            results.append(doc)
    except Exception:
        pass

    # Try Context7 as fallback
    try:
        lib_result = _context7.call_tool("resolve-library-id", {"libraryName": library})
        if lib_result and "failed" not in lib_result.lower():
            # Extract library ID from result
            import re
            match = re.search(r'(/[^\s"]+)', lib_result)
            if match:
                lib_id = match.group(1)
                doc = _context7.call_tool("query-docs", {
                    "context7CompatibleLibraryID": lib_id,
                    "topic": query,
                })
                if doc and "failed" not in doc.lower():
                    results.append(f"[Context7]\n{doc}")
    except Exception:
        pass

    if results:
        return "\n\n".join(results)
    return f"No documentation found for '{library}' - '{query}'"


def execute_python(code: str) -> str:
    """Execute Python code in a sandboxed temp directory."""
    if WORKSPACE is None:
        return "ERROR: No workspace set"

    # Write code to temp file in workspace
    code_file = os.path.join(WORKSPACE, "_exec_tmp.py")
    with open(code_file, "w") as f:
        f.write(code)

    try:
        result = subprocess.run(
            ["python3", code_file],
            capture_output=True, text=True,
            timeout=30,
            cwd=WORKSPACE,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        output += f"\nExit code: {result.returncode}"
        return output.strip()
    except subprocess.TimeoutExpired:
        return "ERROR: Execution timed out (30s)"
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        if os.path.exists(code_file):
            os.remove(code_file)


def write_file(path: str, content: str) -> str:
    """Write a file to the project workspace."""
    if WORKSPACE is None:
        return "ERROR: No workspace set"

    full_path = os.path.join(WORKSPACE, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, "w") as f:
        f.write(content)
    return f"Written: {path} ({len(content)} chars)"


def read_file(path: str) -> str:
    """Read a file from the project workspace."""
    if WORKSPACE is None:
        return "ERROR: No workspace set"

    full_path = os.path.join(WORKSPACE, path)
    if not os.path.exists(full_path):
        return f"ERROR: File not found: {path}"

    with open(full_path) as f:
        return f.read()


def run_tests(test_path: str = "") -> str:
    """Run tests in the workspace."""
    if WORKSPACE is None:
        return "ERROR: No workspace set"

    cmd = ["python3", "-m", "pytest", "-v", "--tb=short"]
    if test_path:
        cmd.append(os.path.join(WORKSPACE, test_path))
    else:
        cmd.append(WORKSPACE)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=60, cwd=WORKSPACE,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n{result.stderr}"
        return output.strip()
    except subprocess.TimeoutExpired:
        return "ERROR: Tests timed out (60s)"
    except Exception as e:
        return f"ERROR: {e}"


def shell(command: str) -> str:
    """Run a whitelisted shell command."""
    allowed_prefixes = ["pip install", "ls", "tree", "cat", "wc", "find", "head"]
    if not any(command.strip().startswith(p) for p in allowed_prefixes):
        return f"ERROR: Command not allowed. Allowed: {allowed_prefixes}"

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=30, cwd=WORKSPACE,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n{result.stderr}"
        return output.strip()[:2000]  # Truncate long output
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out"
    except Exception as e:
        return f"ERROR: {e}"


# Tool registry
TOOLS = {
    "search_docs": {
        "fn": search_docs,
        "description": "Search library documentation. Args: library (str), query (str)",
        "params": ["library", "query"],
    },
    "execute_python": {
        "fn": execute_python,
        "description": "Execute Python code and return output. Args: code (str)",
        "params": ["code"],
    },
    "write_file": {
        "fn": write_file,
        "description": "Write a file to the project workspace. Args: path (str), content (str)",
        "params": ["path", "content"],
    },
    "read_file": {
        "fn": read_file,
        "description": "Read a file from the project workspace. Args: path (str)",
        "params": ["path"],
    },
    "run_tests": {
        "fn": run_tests,
        "description": "Run pytest on the workspace or a specific test file. Args: test_path (str, optional)",
        "params": ["test_path"],
    },
    "shell": {
        "fn": shell,
        "description": "Run a whitelisted shell command (pip install, ls, tree, cat, find). Args: command (str)",
        "params": ["command"],
    },
}


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with given arguments."""
    if name not in TOOLS:
        return f"ERROR: Unknown tool '{name}'. Available: {list(TOOLS.keys())}"

    tool = TOOLS[name]
    try:
        return tool["fn"](**arguments)
    except Exception as e:
        return f"ERROR executing {name}: {e}"
