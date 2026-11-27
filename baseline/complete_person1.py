"""Complete reports, fresh-kernel execution and verification after baseline inference."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import pandas as pd

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true')
args=parser.parse_args()
while True:
    path=ROOT/'results/baseline_metrics.csv'
    ready=False
    if path.exists():
        try:
            metrics=pd.read_csv(path)
            ready=len(metrics)==3 and 'Complete' in metrics and metrics.Complete.all()
        except (pd.errors.EmptyDataError,pd.errors.ParserError):
            ready=False
    if ready:break
    if not args.wait:raise RuntimeError('Full baseline not ready; run evaluation.py first')
    print('Waiting for full baseline inference...',flush=True)
    time.sleep(30)

steps=['reporting.py','execute_person1_notebook.py','verify_person1.py','verify_handover_extensions.py']
if (ROOT/'results/cleaned_black103/baseline_metrics.csv').exists():
    steps.append('verify_clean_test.py')
steps.append('finalize_person1.py')
for name in steps:
    print('Running',name,flush=True)
    with (ROOT/'results'/('person1_'+Path(name).stem+'_console.log')).open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/name)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        print('FAILED:',name,flush=True)
        sys.exit(result.returncode)
    print('PASS:',name,flush=True)
print('All real evaluation artifacts and checks completed.',flush=True)
