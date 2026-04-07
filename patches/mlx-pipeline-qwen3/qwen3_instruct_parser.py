# Tool parser for Qwen3 Instruct models that output <|python_tag|> format
# Output format: <|python_tag|>{"name": "func", "parameters": {...}}

import json

tool_call_start = '<|python_tag|>'
tool_call_end = ''  # Empty - the tool call ends at end of generation


def parse_tool_call(text, tools=None):
    text = text.strip()
    # Handle both {"name": ..., "parameters": ...} and func_name({...}) formats
    try:
        data = json.loads(text)
        # Normalize: "parameters" -> "arguments"
        if 'parameters' in data and 'arguments' not in data:
            data['arguments'] = data.pop('parameters')
        return data
    except json.JSONDecodeError:
        pass

    # Try function call format: func_name({...})
    import re
    m = re.match(r'(\w+)\s*\((.+)\)\s*$', text, re.DOTALL)
    if m:
        try:
            args = json.loads(m.group(2))
            return {'name': m.group(1), 'arguments': args}
        except:
            pass

    raise ValueError(f'Could not parse tool call: {text[:100]}')
