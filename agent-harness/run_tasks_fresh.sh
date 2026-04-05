#!/bin/bash
# Run each benchmark task with a fresh cluster restart between tasks.
# Usage: ./run_tasks_fresh.sh <model_id> <label> [task1 task2 ...]
# If no tasks specified, runs all 5.

MODEL="${1:-mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit}"
LABEL="${2:-A22B-8bit}"
shift 2 2>/dev/null
TASKS=("$@")
if [ ${#TASKS[@]} -eq 0 ]; then
    TASKS=(static_site_generator rest_api data_pipeline cli_tool algorithm_no_tools)
fi

LOG="/tmp/${LABEL}-fresh-benchmark.log"
RESULTS_DIR="/tmp/${LABEL}-fresh-results"
mkdir -p "$RESULTS_DIR"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

restart_cluster() {
    log "Killing Exo on all nodes..."
    ssh vader "pkill -f exo" 2>/dev/null &
    ssh voldemort "pkill -f exo" 2>/dev/null &
    ssh gargamel "pkill -f exo" 2>/dev/null &
    wait
    sleep 5

    # Check if wired memory needs reboot
    V_WIRED=$(ssh vader "vm_stat | grep wired | awk '{print \$4}' | tr -d '.'" 2>/dev/null)
    V_GB=$((V_WIRED * 16 / 1048576))
    if [ "$V_GB" -gt 20 ]; then
        log "Vader has ${V_GB}GB wired — sending SIGTERM to free..."
        ssh vader "pkill -TERM -f exo" 2>/dev/null
        sleep 10
        V_WIRED=$(ssh vader "vm_stat | grep wired | awk '{print \$4}' | tr -d '.'" 2>/dev/null)
        V_GB=$((V_WIRED * 16 / 1048576))
        if [ "$V_GB" -gt 20 ]; then
            log "Still ${V_GB}GB wired — rebooting all nodes..."
            ssh vader "sudo reboot" 2>/dev/null &
            ssh voldemort "sudo reboot" 2>/dev/null &
            ssh gargamel "sudo reboot" 2>/dev/null &
            wait
            sleep 90
            # Wait for nodes
            for i in $(seq 1 20); do
                ssh vader "uptime" > /dev/null 2>&1 && ssh voldemort "uptime" > /dev/null 2>&1 && ssh gargamel "uptime" > /dev/null 2>&1 && break
                sleep 10
            done
            log "All nodes back"
        fi
    fi

    log "Starting cluster..."
    ssh vader "nohup bash ~/exo-src/start-cluster.sh > /tmp/exo-start.log 2>&1 &" &
    ssh voldemort "nohup bash ~/exo-src/start-cluster.sh > /tmp/exo-start.log 2>&1 &" &
    ssh gargamel "nohup bash ~/exo-src/start-cluster.sh > /tmp/exo-start.log 2>&1 &" &
    wait
    sleep 40

    log "Placing model..."
    curl -s -X POST http://100.105.116.62:52415/place_instance \
        -H "Content-Type: application/json" \
        -d "{\"model_id\":\"${MODEL}\",\"min_nodes\":3,\"instance_meta\":\"MlxJaccl\"}" > /dev/null 2>&1

    # Wait for model to load
    log "Waiting for model to load..."
    sleep 120

    # Verify
    RESP=$(curl -s --max-time 60 http://100.105.116.62:52415/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"max_tokens\": 20}" 2>&1)
    if echo "$RESP" | grep -q "content"; then
        log "Model loaded and responding!"
        return 0
    else
        log "WARNING: Model may not be ready: $RESP"
        return 1
    fi
}

cd /home/thomas/claude/random/agent-harness

log "Starting fresh benchmarks: ${TASKS[*]}"
log "Model: $MODEL ($LABEL)"

for TASK in "${TASKS[@]}"; do
    log "=========================================="
    log "TASK: $TASK (fresh restart)"
    log "=========================================="

    restart_cluster

    python3 -c "
import os, json, tempfile
from tasks import TASKS
from harness import run_agent
from evaluate import evaluate_task, compute_overall_score

model = '${MODEL}'
task = TASKS['${TASK}']

print(f'Task: {task[\"name\"]}')
print(f'Model: ${LABEL} (fresh)')

with tempfile.TemporaryDirectory() as workspace:
    result = run_agent(task['prompt'], model, workspace, max_iterations=10, verbose=True)
    eval_result = evaluate_task(workspace, task, result)
    overall = compute_overall_score(eval_result)

    print(f'\nScore: {overall:.1f}/100')
    for check, passed in eval_result.get('check_results', {}).items():
        status = '✓' if passed else '✗'
        print(f'  {status} {check}')
    print(f'Tests pass: {eval_result.get(\"tests_pass\", False)}')
    print(f'Code runs: {eval_result.get(\"code_runs\", False)}')
    tc = {}
    for call in result['tool_calls']:
        tc[call['name']] = tc.get(call['name'], 0) + 1
    print(f'Iterations: {result[\"iterations\"]} | Tools: {len(result[\"tool_calls\"])} | Time: {result[\"total_time\"]:.0f}s')
    for name, count in sorted(tc.items(), key=lambda x:-x[1]):
        print(f'  {name}: {count}x')

    with open('${RESULTS_DIR}/${TASK}.json', 'w') as f:
        json.dump({
            'task': '${TASK}', 'task_name': task['name'], 'score': overall,
            'iterations': result['iterations'], 'tools': len(result['tool_calls']),
            'time': round(result['total_time']), 'tests_pass': eval_result.get('tests_pass', False),
            'code_runs': eval_result.get('code_runs', False), 'checks': eval_result.get('check_results', {}),
        }, f, indent=2)
" 2>&1 | tee -a "$LOG"

    log "Task $TASK complete"
done

log "=========================================="
log "ALL TASKS COMPLETE"
log "=========================================="

# Print summary
python3 -c "
import json, os
results = []
for f in sorted(os.listdir('${RESULTS_DIR}')):
    if f.endswith('.json'):
        results.append(json.load(open(os.path.join('${RESULTS_DIR}', f))))
print(f\"{'Task':<35} {'Score':>6} {'Iters':>6} {'Time':>6} {'Tests':>6}\")
print('-' * 65)
total = 0
for r in results:
    total += r['score']
    print(f\"{r['task_name']:<35} {r['score']:>5.1f}/100 {r['iterations']:>6} {r['time']:>5}s {'✓' if r['tests_pass'] else '✗':>5}\")
print('-' * 65)
print(f\"{'AVERAGE':<35} {total/len(results):>5.1f}/100\")
" 2>&1 | tee -a "$LOG"
