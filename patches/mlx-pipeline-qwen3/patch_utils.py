"""Patch utils.py to add pipeline_warmup() call in sharded_load."""
import sys

target = sys.argv[1]
with open(target) as f:
    content = f.read()

if 'pipeline_warmup' in content:
    print(f"  {target}: already patched")
    sys.exit(0)

marker = "    # Synchronize processes to avoid timeout"
insert = """
    # Warm up Metal shaders for pipeline models
    if pipeline_group is not None and hasattr(model.model, "pipeline_warmup"):
        import os
        if os.environ.get('MLX_SKIP_WARMUP') != '1':
            model.model.pipeline_warmup()
"""

# Find the sync line and the line after it, insert warmup after both
lines = content.split('\n')
new_lines = []
found = False
for i, line in enumerate(lines):
    new_lines.append(line)
    if not found and marker in line:
        # Include the mx.eval line that follows
        pass
    elif not found and i > 0 and marker in lines[i-1]:
        # This is the mx.eval(all_sum...) line — insert warmup after it
        new_lines.append(insert)
        found = True

if found:
    with open(target, 'w') as f:
        f.write('\n'.join(new_lines))
    print(f"  {target}: warmup call added")
else:
    print(f"  WARNING: Could not find sync marker in {target}")
