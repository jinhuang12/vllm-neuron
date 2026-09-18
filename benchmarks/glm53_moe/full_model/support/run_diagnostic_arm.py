"""Collect the fixed workload after the unchanged baseline failed repeatability."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--arm', required=True, choices=['candidate', 'postbaseline'])
p.add_argument('--run', required=True)
a = p.parse_args()
root = Path('/home/ubuntu/glm53-moe-fullmodel-20260917')
out = root / 'artifacts' / a.run
source = root / ('candidate-source' if a.arm == 'candidate' else 'baseline-source')
cache = root / 'cache' / ('candidate' if a.arm == 'candidate' else 'baseline')
python = '/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/python'
model = str(root / 'models/GLM-5.3-Flash-04c4e9e9')
env = dict(os.environ, PYTHONPATH=str(root / 'deps') + ':' + str(source),
           VLLM_NEURON_CPU_MODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')

def phase(name):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), name, flush=True)
    (out / 'measurement-status.json').write_text(json.dumps({'phase': name, 'unix': time.time()}))

def run(mode, extra, check=True):
    return subprocess.run([python, str(root / 'harness/experiment.py'), mode, *extra],
                          env=env, check=check).returncode

def cache_state(name):
    subprocess.run(['python3', str(root / 'scripts/cache_file_state.py'), '--cache', str(cache),
                    '--output', str(out / name)], check=True)

phase('waiting_for_server')
while True:
    if (out / 'exit.json').exists():
        raise RuntimeError('Server exited before readiness')
    try:
        with urllib.request.urlopen('http://127.0.0.1:18004/health', timeout=3) as r:
            if r.status == 200:
                break
    except Exception:
        pass
    time.sleep(5)

baseline_repeatability = root / 'artifacts/baseline-r1/repeatability.json'
scope = {'arm': a.arm, 'baseline_repeatability': json.loads(baseline_repeatability.read_text()),
         'scope': 'Diagnostic timing. Baseline A/A repeatability failed before candidate exposure.',
         'measurement_contract_changed': False,
         'acceptance': 'Performance alone cannot pass correctness or authorize a PR.'}
with (out / 'performance-diagnostic-scope.json').open('x') as stream:
    json.dump(scope, stream, indent=2)

if a.arm == 'candidate':
    for name in ['correctness-b1', 'correctness-b2']:
        phase(name)
        run('capture', ['--model', model, '--output', str(out / name)])
    phase('diagnostic_comparisons')
    comparisons = {}
    for name, left, right in [
        ('repeatability', out / 'correctness-b1/capture.json', out / 'correctness-b2/capture.json'),
        ('baseline-comparison', root / 'artifacts/baseline-r1/correctness-a1/capture.json',
         out / 'correctness-b1/capture.json'),
    ]:
        comparisons[name] = run('compare', ['--left', str(left), '--right', str(right),
                                           '--output', str(out / (name + '.json'))], check=False)
        comparison_path = out / (name + '.json')
        if comparisons[name] not in [0, 1] or not comparison_path.is_file():
            raise RuntimeError(f'Comparison process failed: {name}')
        comparison = json.loads(comparison_path.read_text())
        if comparison.get('pass') is not (comparisons[name] == 0):
            raise RuntimeError(f'Comparison report and process disagree: {name}')
    (out / 'comparison-exit-codes.json').write_text(json.dumps(comparisons, indent=2))

cache_state('cache-ready-file-state.json')
phase('diagnostic_performance')
run('benchmark', ['--model', model, '--output', str(out / 'benchmark')])
cache_state('cache-after-timing-file-state.json')
before = json.loads((out / 'cache-ready-file-state.json').read_text())['files']
after = json.loads((out / 'cache-after-timing-file-state.json').read_text())['files']
(out / 'cache-timing-comparison.json').write_text(json.dumps({'unchanged': before == after,
                                                            'before_count': len(before),
                                                            'after_count': len(after)}, indent=2))
subprocess.run(['python3', str(root / 'harness/collect.py'), '--cache', str(cache),
                '--log', str(out / 'server.log'), '--output', str(out / 'cache-after.json')], check=True)
phase('diagnostic_complete')
