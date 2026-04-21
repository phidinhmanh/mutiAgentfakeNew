"""Quick test for LLM connection with timeout."""
import sys
import time
import json
import re
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

log_file = open('llm_test.log', 'w', encoding='utf-8')

from datasets import load_dataset
from openai import OpenAI
from fake_news_detector.config import settings

log_file.write(f"Test started at {datetime.now()}\n")
log_file.write(f"API Key present: {bool(settings.nvidia_api_key)}\n")
log_file.write(f"LLM Model: {settings.llm_model}\n")

dataset = load_dataset('tranthaihoa/vifactcheck', split='test')
item = dataset[0]
claim = item.get('Statement', '')
evidence_text = item.get('Evidence', '')

log_file.write(f'Claim: {claim[:60]}...\n')
log_file.write(f'Evidence: {evidence_text[:60]}...\n')

client = OpenAI(
    base_url='https://integrate.api.nvidia.com/v1',
    api_key=settings.nvidia_api_key,
    timeout=30.0,  # 30 second timeout
)

evidence_text_formatted = f'[0] {evidence_text}'
prompt = f"""Claim: {claim}

Evidence:
{evidence_text_formatted}

Tra ve JSON voi dinh dang:
{{
  "verdict": "REAL|FAKE|UNVERIFIABLE",
  "confidence": 0.0-1.0,
  "reasoning": "Giai thich dua tren evidence"
}}

Chi su dung thong tin trong Evidence de giai thich."""

log_file.write("Sending request...\n")
log_file.flush()
start = time.time()

try:
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": "Ban la chuyen gia xac thuc tin tuc Viet Nam."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )

    elapsed = time.time() - start

    result_text = response.choices[0].message.content.strip()
    log_file.write(f'LLM response time: {elapsed:.2f}s\n')
    log_file.write(f'Response: {result_text[:200]}...\n')

    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
    if json_match:
        result = json.loads(json_match.group())
        log_file.write(f"Parsed verdict: {result.get('verdict')}\n")

except Exception as e:
    log_file.write(f"Error: {e}\n")
    import traceback
    traceback.print_exc(file=log_file)

log_file.write(f"Test completed at {datetime.now()}\n")
log_file.close()
print("Done, check llm_test.log")