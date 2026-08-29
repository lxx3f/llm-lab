import json
import re
import sys

with open('README.md', encoding='utf-8') as f:
    text = f.read()

# Find all fenced json blocks: opening "```json", content, closing "```"
pattern = re.compile(r'```json\n(.*?)\n```', re.DOTALL)
json_blocks = pattern.findall(text)
print(f'Found {len(json_blocks)} JSON blocks in README.md')

# Map blocks to their schemas (based on order: model_output, tool_calling, evaluation_result)
schema_map = [
    'schemas/model_output.schema.json',
    'schemas/tool_calling_sample.schema.json',
    'schemas/evaluation_result.schema.json',
]

import jsonschema

all_pass = True
for i, block in enumerate(json_blocks[:3]):
    if i >= len(schema_map):
        break
    schema_file = schema_map[i]
    try:
        data = json.loads(block)
        with open(schema_file, encoding='utf-8') as f:
            schema_data = json.load(f)
        jsonschema.validate(data, schema_data)
        print(f'  Block {i} -> {schema_file}: PASS')
    except json.JSONDecodeError as e:
        print(f'  Block {i} -> {schema_file}: JSON parse error: {e}')
        all_pass = False
    except jsonschema.ValidationError as e:
        print(f'  Block {i} -> {schema_file}: VALIDATION FAIL')
        print(f'    Error path: {list(e.absolute_path)}')
        print(f'    Error message: {e.message}')
        all_pass = False
    except Exception as e:
        print(f'  Block {i} -> {schema_file}: ERROR: {e}')
        all_pass = False

sys.exit(0 if all_pass else 1)