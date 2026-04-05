"""Core agent loop for benchmarking LLMs with tool access."""

import json
import re
import time
import requests
from typing import Optional
from tools import TOOLS, execute_tool, set_workspace

import os
EXO_URL = os.environ.get("LLM_API_URL", "http://192.168.0.126:52415")
MAX_ITERATIONS = 10
MAX_TOKENS = 8192

SYSTEM_PROMPT = """You are a skilled software engineer. You have access to the following tools:

{tool_descriptions}

To use a tool, write a tool call in this EXACT format (including the tags):
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>

Example - writing a file:
<tool_call>
{{"name": "write_file", "arguments": {{"path": "hello.py", "content": "print('hello')"}}}}
</tool_call>

You MUST use <tool_call> and </tool_call> tags. Do NOT use any other format.
You can use multiple tool calls in a single response. After each tool call, you will receive the result in a <tool_result> block.

When you are done with the task, write your final summary without any tool calls.

Rules:
- Write ONE file per tool call — do not try to write multiple files in a single call
- Keep file contents under 3000 characters per write_file call. Split large files into parts if needed.
- Before writing code that uses external libraries, use search_docs to check the correct API usage
- After writing code, use execute_python or run_tests to verify it works
- Fix any errors before declaring completion
- Be methodical: search docs first, then implement file by file, then test, then fix
"""


def build_system_prompt() -> str:
    """Build system prompt with tool descriptions."""
    descs = []
    for name, tool in TOOLS.items():
        descs.append(f"- **{name}**: {tool['description']}")
    return SYSTEM_PROMPT.format(tool_descriptions="\n".join(descs))


def parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from model output."""
    calls = []
    # Try multiple tool call formats
    patterns = [
        r'<tool_call>\s*(.*?)\s*</tool_call>',           # Our XML format
        r'<\|im_start\|>\s*(.*?)\s*(?:<\|im_end\|>|$)',  # Qwen format
        r'```tool_call\s*(.*?)\s*```',                    # Markdown format
        r'```json\s*(\{[^`]*"name"[^`]*"arguments"[^`]*\})\s*```',  # JSON in code block
        r'<｜tool▁call▁begin｜>.*?<｜tool▁sep｜>\w+\s*```json\s*(.*?)\s*```',  # DeepSeek format
    ]
    matches = []
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.DOTALL))
        if matches:
            break
    for match in matches:
        try:
            call = json.loads(match.group(1))
            if "name" in call and "arguments" in call:
                calls.append(call)
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            raw = match.group(1)
            # Fix single quotes
            raw = raw.replace("'", '"')
            try:
                call = json.loads(raw)
                if "name" in call and "arguments" in call:
                    calls.append(call)
            except:
                pass
    return calls


def _build_openai_tools():
    """Build OpenAI-compatible tools schema from TOOLS registry."""
    schemas = {
        "search_docs": {"type": "object", "properties": {"library": {"type": "string"}, "query": {"type": "string"}}, "required": ["library", "query"]},
        "execute_python": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
        "write_file": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
        "read_file": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        "run_tests": {"type": "object", "properties": {"test_path": {"type": "string"}}, "required": []},
        "shell": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    }
    return [
        {"type": "function", "function": {"name": name, "description": tool["description"],
         "parameters": schemas.get(name, {"type": "object", "properties": {}})}}
        for name, tool in TOOLS.items()
    ]
OPENAI_TOOLS = _build_openai_tools()

def query_exo(messages: list[dict], model: str, max_tokens: int = MAX_TOKENS) -> dict:
    """Send messages to Exo API."""
    start = time.time()
    try:
        payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
        }
        # Add native tools for Ollama compatibility
        if "11434" in EXO_URL:
            payload["tools"] = OPENAI_TOOLS
        r = requests.post(
            f"{EXO_URL}/v1/chat/completions",
            json=payload,
            timeout=1200,
        )
        elapsed = time.time() - start
        data = r.json()

        if "error" in data:
            return {"content": "", "error": data["error"], "elapsed": elapsed, "tokens": 0}

        choices = data.get("choices", [])
        if not choices:
            return {"content": "", "error": "Empty choices in response", "elapsed": elapsed, "tokens": 0}

        message = choices[0].get("message") or {}
        usage = data.get("usage") or {}
        content = message.get("content", "") or ""
        reasoning = message.get("reasoning_content", "") or ""

        # Check for native OpenAI tool_calls format
        native_tool_calls = message.get("tool_calls", []) or []
        if native_tool_calls:
            # Convert native tool_calls to our <tool_call> format
            for tc in native_tool_calls:
                fn = tc.get("function", {})
                import json as _json
                try:
                    args = _json.loads(fn.get("arguments", "{}"))
                except:
                    args = {}
                content += f'\n<tool_call>\n{_json.dumps({"name": fn.get("name",""), "arguments": args})}\n</tool_call>\n'

        return {
            "content": content,
            "reasoning": reasoning,
            "elapsed": elapsed,
            "tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"content": "", "error": str(e), "elapsed": time.time() - start, "tokens": 0}


def run_agent(
    task: str,
    model: str,
    workspace: str,
    max_iterations: int = MAX_ITERATIONS,
    verbose: bool = True,
) -> dict:
    """Run the agent loop for a task.

    Returns dict with:
        iterations: number of loops
        tool_calls: list of all tool calls made
        total_tokens: total tokens consumed
        total_time: total wall time
        final_output: last model response
        success: whether task completed (no tool calls in final response)
    """
    set_workspace(workspace)

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": task},
    ]

    all_tool_calls = []
    total_tokens = 0
    start_time = time.time()

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---", flush=True)

        try:
            response = query_exo(messages, model)
        except Exception as e:
            if verbose:
                print(f"EXCEPTION: {e}", flush=True)
            continue

        if response is None or "error" in response:
            err = (response or {}).get("error", "Unknown")
            if verbose:
                print(f"ERROR: {err}", flush=True)
            continue

        content = response["content"]
        total_tokens += response["tokens"]

        if verbose:
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"Model ({response['tokens']} tok, {response['elapsed']:.1f}s): {preview}", flush=True)

        # Check for tool calls
        tool_calls = parse_tool_calls(content)

        if not tool_calls:
            if verbose:
                print("No tool calls — task complete.", flush=True)
            return {
                "iterations": iteration + 1,
                "tool_calls": all_tool_calls,
                "total_tokens": total_tokens,
                "total_time": time.time() - start_time,
                "final_output": content,
                "success": True,
            }

        # Execute tool calls and build result message
        messages.append({"role": "assistant", "content": content})

        results_text = ""
        for call in tool_calls:
            name = call["name"]
            args = call["arguments"]
            all_tool_calls.append(call)

            if verbose:
                print(f"  Tool: {name}({json.dumps(args)[:80]})", flush=True)

            result = execute_tool(name, args)

            if verbose:
                preview = result[:150] + "..." if len(result) > 150 else result
                print(f"  Result: {preview}", flush=True)

            results_text += f"\n<tool_result>\n{{\"name\": \"{name}\", \"result\": {json.dumps(result)}}}\n</tool_result>\n"

        messages.append({"role": "user", "content": results_text})

    return {
        "iterations": max_iterations,
        "tool_calls": all_tool_calls,
        "total_tokens": total_tokens,
        "total_time": time.time() - start_time,
        "final_output": content if 'content' in dir() else "",
        "success": False,  # Hit max iterations
    }


if __name__ == "__main__":
    import sys
    import tempfile

    model = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/MiniMax-M2.5-8bit"
    task = "Write a Python function that implements a thread-safe LRU cache with TTL expiration. Write it to lru_cache.py and include tests in test_lru_cache.py. Run the tests to verify."

    with tempfile.TemporaryDirectory() as workspace:
        print(f"Model: {model}")
        print(f"Workspace: {workspace}")
        print(f"Task: {task}")

        result = run_agent(task, model, workspace, verbose=True)

        print(f"\n{'='*60}")
        print(f"Iterations: {result['iterations']}")
        print(f"Tool calls: {len(result['tool_calls'])}")
        print(f"Tokens: {result['total_tokens']}")
        print(f"Time: {result['total_time']:.1f}s")
        print(f"Success: {result['success']}")
