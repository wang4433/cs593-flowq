"""Inspect a trusted official Q2RL checkpoint, never arbitrary untrusted pickle."""
import argparse
import json
from pathlib import Path
import torch
from flowq.data import sha256

p=argparse.ArgumentParser(); p.add_argument('checkpoint'); p.add_argument('--out',required=True)
args=p.parse_args()
ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
config=ck['config']; config=json.loads(config) if isinstance(config,str) else config
report=dict(path=str(Path(args.checkpoint).resolve()),sha256=sha256(args.checkpoint),
            config=config,env_metadata=ck.get('env_metadata'),shape_metadata=ck.get('shape_metadata'))
Path(args.out).parent.mkdir(parents=True,exist_ok=True)
Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
train=config.get('train',{})
print(json.dumps({k:v for k,v in train.items() if any(x in k for x in ('filter','valid','data','hdf5'))},indent=2))
