"""Patch server.py to disable batching for pipeline models."""
import sys

target = sys.argv[1]
with open(target) as f:
    content = f.read()

if 'pipeline_group' in content and 'is_batchable = False' in content and content.count('is_batchable = False') > 1:
    print(f"  {target}: already patched")
    sys.exit(0)

# Find the is_batchable = all(...) block and add pipeline override after it
old = """        if self.draft_model is None:
            self.is_batchable = all(
                hasattr(c, "merge") for c in make_prompt_cache(self.model)
            )"""

new = old + """

            # Disable batching for pipeline models (BatchGenerator has no pipeline sync)
            if self.pipeline_group is not None:
                self.is_batchable = False"""

if old in content:
    content = content.replace(old, new)
    with open(target, 'w') as f:
        f.write(content)
    print(f"  {target}: pipeline batchable fix added")
else:
    print(f"  WARNING: Could not find is_batchable block in {target}")
    sys.exit(1)
