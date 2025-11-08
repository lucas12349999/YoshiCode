from openai import OpenAI
import os, sys

base_url = os.environ.get("YOSHICODE_BASE","http://localhost:8000/v1")
client = OpenAI(base_url=base_url, api_key="dummy")

resp = client.chat.completions.create(
    model="yoshicode-1b",
    messages=[
        {"role":"system","content":"You are a helpful coding assistant."},
        {"role":"user","content":"Write a Python function to merge two sorted lists."}
    ],
    temperature=0.2,
    max_tokens=300
)
print(resp.choices[0].message.content)
