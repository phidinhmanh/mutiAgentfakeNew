import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('benchmark_multi_v4.json', encoding='utf-8') as f:
    d = json.load(f)

s2 = d['benchmark_results'][1]
print('=== S02 (ID 44) ===')
print('Pipeline Verdict:', s2['verdict_label'])
print('Expected: FAKE (statement date modified from 25/3 to 25/2)')
print()
print('Statement:', d['samples'][1]['statement'])
print()
print('Evidence:', d['samples'][1]['evidence'])
print()
for i, c in enumerate(s2['all_claims_results']):
    claim_short = c['claim']
    verdict = c['verdict']
    reasoning = c['reasoning']
    print(f"Claim {i+1}: [{verdict}] {claim_short}")
    print(f"  Reasoning: {reasoning[:300]}")
    print()