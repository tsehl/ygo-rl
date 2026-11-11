"""Compare deux moteurs à banque identique, trois paires dans chaque ordre.

Exemple : .venv/bin/python bench/compare_branches.py /chemin/main/engine engine \
    --out validation/debit
Lancer lorsque la machine ne fait aucun autre travail de calcul.
"""
import argparse
import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('main', type=Path)
p.add_argument('codex', type=Path)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
engines = {'main': a.main.resolve(), 'codex': a.codex.resolve()}
benchmark = Path(__file__).resolve().with_name('engine_bench.py')
order = [('main', 'codex'), ('codex', 'main')] * 3
runs = []
for pair, labels in enumerate(order, 1):
    for label in labels:
        number = len(runs) + 1
        cmd = [sys.executable, str(benchmark), str(engines[label]), '--b', '64',
               '--calls', '20', '--extra', '15']
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        path = a.out / f'{number:02d}-{label}.log'
        path.write_text(result.stdout)
        matches = re.findall(r'B=64\s*:\s*([\d,]+) décisions/s\s*;\s*([\d.]+) ms/appel\s*;\s*(\d+) décisions/appel', result.stdout)
        if result.returncode or len(matches) != 1:
            raise RuntimeError(f'échec de la mesure {path} (code {result.returncode})')
        rate, ms, decisions = matches[0]
        entry = dict(pair=pair, label=label, decisions_per_second=int(rate.replace(',', '')),
                     milliseconds_per_call=float(ms), decisions_per_call=int(decisions), log=path.name)
        runs.append(entry)
        print(json.dumps(entry), flush=True)

def stats(label):
    values = [r['decisions_per_second'] for r in runs if r['label'] == label]
    median = statistics.median(values)
    return dict(values=values, median=median, minimum=min(values), maximum=max(values),
                median_absolute_deviation=statistics.median(abs(x-median) for x in values))

summary = dict(timestamp=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
               python=platform.python_version(), command=['--b', '64', '--calls', '20', '--extra', '15'],
               code_sha256={name: hashlib.sha256((path / 'core.py').read_bytes()).hexdigest()
                            for name, path in engines.items()}, runs=runs,
               main=stats('main'), codex=stats('codex'))
summary['change_percent'] = 100 * (summary['codex']['median'] / summary['main']['median'] - 1)
(a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: summary[k] for k in ('main', 'codex', 'change_percent')}, indent=2), flush=True)
