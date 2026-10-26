#!/usr/bin/env python3
"""Export small, path-sanitized measured evidence; never include model binaries."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES={
    'isa-tile.json':'qmac8-requant-spike/build/comparison.json',
    'fc1.json':'tflm-linear/build/comparison.json',
    'fc1-gelu.json':'tflm-linear/build/comparison-gelu.json',
    'blocking-fusion.json':'tflm-linear/build/optimizations.json',
    'full-inference.json':'deit-full-int8/build/profile.json',
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT,help='Repository root containing completed runs')
    p.add_argument('--label',default='reproduced',help='One directory name under results/')
    a=p.parse_args();source=a.source.resolve()
    if Path(a.label).name!=a.label or a.label in ['.','..']:
        p.error('--label must be one directory name')
    paths={name:source/'experiments'/rel for name,rel in FILES.items()}
    missing=[str(path) for path in paths.values() if not path.exists()]
    if missing: raise SystemExit('Run test, bench and full first. Missing:\n'+'\n'.join(missing))
    out=ROOT/'results'/a.label;out.mkdir(parents=True,exist_ok=True)
    provenance={}
    for name,path in paths.items():
        raw=path.read_text()
        # Absolute paths are metadata only; the recorded original-file hash remains available.
        data=json.loads(raw.replace(str(source),'<repo>'))
        (out/name).write_text(json.dumps(data,indent=2)+'\n')
        provenance[name]={'original_path':str(path.relative_to(source)),
                          'original_sha256':hashlib.sha256(raw.encode()).hexdigest()}
    hashes={}
    for folder in FILES.values():
        directory=source/'experiments'/folder.split('/')[0]
        for path in directory.iterdir():
            if path.is_file() and path.suffix in ['.py','.c','.cc','.h','.S','.ld','.sh']:
                hashes[str(path.relative_to(source))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'provenance.json').write_text(json.dumps(dict(
        note='Measured functional Spike counts. Absolute source-root paths replaced with <repo>. See EXPERIMENTS.md for scope and limitations.',
        files=provenance,source_sha256=hashes),indent=2)+'\n')
    print('Exported:',out)
if __name__=='__main__': main()
