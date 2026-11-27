"""Execute all notebook cells in a fresh workspace-local NPR kernel."""
import json
import os
from pathlib import Path
import sys
import time
import traceback
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
kernel_root=ROOT/'results/person1_jupyter'
kernel_dir=kernel_root/'kernels/npr-person1'
kernel_dir.mkdir(parents=True,exist_ok=True)
(kernel_dir/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-X','utf8','-m','ipykernel_launcher','-f','{connection_file}'],display_name='Python (NPR CPU)',language='python')),encoding='utf-8')
os.environ['JUPYTER_PATH']=str(kernel_root)+os.pathsep+os.environ.get('JUPYTER_PATH','')
os.environ['JUPYTER_RUNTIME_DIR']=str(kernel_root/'runtime')
Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(exist_ok=True)
path=ROOT/'NPR_Reproduction_Complete.ipynb'
nb=nbformat.read(path,as_version=4)
started=time.perf_counter()
status=dict(status='NOT RUN',kernel_executable=sys.executable,kernel_name='npr-person1')
try:
    NotebookClient(nb,timeout=3600,kernel_name='npr-person1',resources={'metadata':{'path':str(ROOT)}},allow_errors=False).execute()
    nbformat.validate(nb)
    nbformat.write(nb,path)
    status.update(status='PASS',code_cells=sum(c.cell_type=='code' for c in nb.cells),
                  executed_cells=sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells))
except Exception:
    status.update(status='FAIL',error=traceback.format_exc())
    nbformat.write(nb,ROOT/'results/person1_notebook_failed.ipynb')
finally:
    status['elapsed_seconds']=time.perf_counter()-started
    (ROOT/'results/person1_notebook_execution.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2))
if status['status']!='PASS':sys.exit(1)
