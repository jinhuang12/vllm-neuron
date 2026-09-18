import argparse,hashlib,json,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--cache',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
root=Path(a.cache)
files={}
for f in sorted((root/'neuron/compile_cache').rglob('*')):
 if not f.is_file() or not (f.suffix=='.neff' or f.name in ('.compilation_complete','log-neuron-cc.txt','fxgraph.txt','graph.hlo')):continue
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for part in iter(lambda:stream.read(4194304),b''):h.update(part)
 files[str(f.relative_to(root))]={'size':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns,'sha256':h.hexdigest()}
with Path(a.output).open('x') as stream:json.dump({'unix':time.time(),'cache':str(root),'files':files},stream,indent=2)
print(len(files),'files')
