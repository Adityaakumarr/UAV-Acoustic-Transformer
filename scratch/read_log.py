import json
import sys

log_path = 'C:/Users/User/.gemini/antigravity-ide/brain/673b7c94-8e7c-44a1-a0f1-b2b9489d2d81/.system_generated/logs/transcript.jsonl'

with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        idx = obj.get('step_index')
        if idx is not None and 540 <= idx <= 580:
            print(f"=== STEP {idx} ===")
            print(f"Source: {obj.get('source')}")
            print(f"Type: {obj.get('type')}")
            content = obj.get('content')
            if content:
                print(content.encode('ascii', errors='ignore').decode('ascii')[:1000])
            print("-" * 50)
