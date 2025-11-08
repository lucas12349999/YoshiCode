#!/usr/bin/env python3
import json, sys

# Usage: python scripts/so_to_sft.py raw_stackoverflow.jsonl > data/processed/sft_so.jsonl

def to_example(q_title, q_body, a_body):
    instruction = f"{q_title}\n\n{q_body}".strip()
    output = a_body.strip()
    return {"instruction": instruction, "input": "", "output": output}

def main(in_path):
    with open(in_path) as f:
        for line in f:
            ex = json.loads(line)
            q_title = ex.get("question_title","")
            q_body  = ex.get("question_body","")
            a_body  = ex.get("accepted_answer","") or ex.get("top_answer","")
            if q_title and a_body:
                print(json.dumps(to_example(q_title, q_body, a_body), ensure_ascii=False))
if __name__ == "__main__":
    main(sys.argv[1])
