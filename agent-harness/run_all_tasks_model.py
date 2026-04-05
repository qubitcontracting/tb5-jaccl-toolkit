"""Run all 5 benchmark tasks with a given model."""

import os
import json
import sys
import tempfile
import time

from tasks import TASKS
from harness import run_agent
from evaluate import evaluate_task, compute_overall_score

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-Coder-Next-bf16"
LABEL = sys.argv[2] if len(sys.argv) > 2 else MODEL.split("/")[-1]
MAX_ITERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
OUTPUT_FILE = f"/tmp/{LABEL}-all-tasks-results.json"

results = []

for task_name, task in TASKS.items():
    print(f"\n{'='*60}")
    print(f"Task: {task['name']} ({task_name})")
    print(f"Model: {LABEL}")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as workspace:
        result = run_agent(task['prompt'], MODEL, workspace, max_iterations=MAX_ITERATIONS, verbose=True)

        eval_result = evaluate_task(workspace, task, result)
        overall = compute_overall_score(eval_result)
        print(f"\nScore: {overall:.1f}/100")
        print(f"Files: {eval_result.get('files_found', [])}")
        for check, passed in eval_result.get('check_results', {}).items():
            status = '✓' if passed else '✗'
            print(f"  {status} {check}")
        print(f"Tests pass: {eval_result.get('tests_pass', False)}")
        print(f"Code runs: {eval_result.get('code_runs', False)}")

        tc = {}
        for call in result['tool_calls']:
            tc[call['name']] = tc.get(call['name'], 0) + 1
        print(f"Iterations: {result['iterations']} | Tools: {len(result['tool_calls'])} | Time: {result['total_time']:.0f}s")
        for name, count in sorted(tc.items(), key=lambda x: -x[1]):
            print(f"  {name}: {count}x")

        results.append({
            "task": task_name,
            "task_name": task['name'],
            "score": overall,
            "iterations": result['iterations'],
            "tools_used": len(result['tool_calls']),
            "time": round(result['total_time']),
            "tests_pass": eval_result.get('tests_pass', False),
            "code_runs": eval_result.get('code_runs', False),
            "checks": eval_result.get('check_results', {}),
        })

print(f"\n{'='*60}")
print(f"ALL TASKS SUMMARY — {LABEL}")
print(f"{'='*60}")
print(f"{'Task':<35} {'Score':>6} {'Iters':>6} {'Time':>6} {'Tests':>6}")
print("-" * 65)
total_score = 0
for r in results:
    total_score += r['score']
    print(f"{r['task_name']:<35} {r['score']:>5.1f}/100 {r['iterations']:>6} {r['time']:>5}s {'✓' if r['tests_pass'] else '✗':>5}")
print("-" * 65)
print(f"{'AVERAGE':<35} {total_score/len(results):>5.1f}/100")

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUTPUT_FILE}")
