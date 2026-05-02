import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('benchmark_multi_v4.json', encoding='utf-8') as f:
    d = json.load(f)

s8 = d['benchmark_results'][7]
print('=== S08 (ID 101) ===')
print('Pipeline Verdict:', s8['verdict_label'])
print('Expected: FAKE (statement says "1 trận thắng" but evidence shows "hòa 0-0")')
print()
for i, c in enumerate(s8['all_claims_results']):
    claim_short = c['claim'][:100]
    verdict = c['verdict']
    reasoning = c['reasoning']
    print(f"Claim {i+1}: [{verdict}] {claim_short}")
    print(f"  Reasoning: {reasoning[:300]}")
    print()
