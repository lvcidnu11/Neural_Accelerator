#!/usr/bin/env python3
"""Report both the requested build-up and the fixed-workload comparison."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
B=ROOT/'build'


def main():
    fc=json.loads((B/'comparison.json').read_text())['197x768']
    graph=json.loads((B/'comparison-gelu.json').read_text())['197x768-gelu']
    fcref=json.loads((B/'197x768/reference.json').read_text())
    graphref=json.loads((B/'197x768-gelu/reference.json').read_text())
    assert fcref['input_sha256']==graphref['input_sha256']
    assert fcref['output_sha256']==graphref['fc_output_sha256']
    stages=[
        dict(configuration='qmac8 + software requantization',scope='Full fc1',
             invoke_retired=fc['qmac_only']['invoke_retired']),
        dict(configuration='qmac8 + requant8',scope='Full fc1',
             invoke_retired=fc['custom']['invoke_retired']),
        dict(configuration='qmac8 + requant8 + GELU LUT',scope='Full fc1 -> GELU',
             invoke_retired=graph['custom']['invoke_retired'])]
    for config in [fc,graph]:
        for key in ['baseline','qmac_only','custom']:
            p=config[key]
            assert p['correctness']['outputs_checked']==151296
            assert p['correctness']['failure_stage']==0
    fixed=[dict(configuration=graph[key]['configuration']+' + GELU LUT',
                invoke_retired=graph[key]['invoke_retired'],fc_retired=graph[key]['operator_retired'],
                gelu_retired=graph[key]['gelu_retired']) for key in ['baseline','qmac_only','custom']]
    report=dict(stages=stages,fixed_workload= fixed,
        fc_requant_incremental_reduction_fraction=1-stages[1]['invoke_retired']/stages[0]['invoke_retired'],
        gelu_added_invoke_instructions=stages[2]['invoke_retired']-stages[1]['invoke_retired'],
        gelu_operator_instructions=graph['custom']['gelu_retired'],
        notes=['Stages 1/2 have the same workload; stage 3 adds GELU computation.',
               'All fixed-workload rows execute identical fc1 -> GELU graph and use the SAME LUT.',
               'No direct-formula target GELU baseline was measured, so no standalone GELU speedup is claimed.',
               'Instruction counts are not hardware cycle speedups.'])
    (B/'stages.json').write_text(json.dumps(report,indent=2)+'\n')
    text=['# Full fc1 and GELU ablation — measured 2026-10-24','',
          'All configurations check every output against the integer/quantized reference.','',
          '| Configuration | Executed scope | Invoke retired instructions |',
          '|---|---|---:|']
    for row in stages:
        text.append(f"| {row['configuration']} | {row['scope']} | {row['invoke_retired']:,} |")
    text+=['',f"Requantization adds a {100*report['fc_requant_incremental_reduction_fraction']:.4f}% reduction on the same full fc1 workload.",
        f"Adding GELU increases the Invoke count by {report['gelu_added_invoke_instructions']:,}; the GELU interval itself costs {report['gelu_operator_instructions']:,} instructions.",
        '', '## Same-workload comparison: every row executes full fc1 -> GELU','',
        '| Configuration | Invoke retired | FC interval | GELU interval |','|---|---:|---:|---:|']
    for row in fixed:
        text.append(f"| {row['configuration']} | {row['invoke_retired']:,} | {row['fc_retired']:,} | {row['gelu_retired']:,} |")
    text+=['']+report['notes']+['',
        'The LUT has 256 INT8 entries. Exact GELU uses erfc offline, verified independently against PyTorch float64 GELU for all INT8 inputs.',
        'The graph has two operators, with a project GELU_LUT registration; it is not a complete MLP, Transformer block or calibrated full-model evaluation.','']
    (B/'stages.md').write_text('\n'.join(text))
    print('\n'.join(text))


if __name__=='__main__': main()
