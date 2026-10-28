"""Original Robomimic v1.4.1 data only; no implicit masks or version migration."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

TASKS = {'lift':'Lift', 'can':'PickPlaceCan', 'square':'NutAssemblySquare'}
KEYS = ['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','object']


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(2**20), b''): h.update(chunk)
    return h.hexdigest()


def audit(cfg):
    path=Path(cfg['dataset'])
    if not path.is_file():
        raise FileNotFoundError(f'Missing original demonstrations: {path}. Run scripts/prepare_data.py in a network-enabled shell.')
    digest=sha256(path)
    sidecar=path.with_suffix(path.suffix+'.source.json')
    source=json.loads(sidecar.read_text()) if sidecar.exists() else None
    if source is not None:
        expected=f"{cfg['task']}/ph/low_dim_v141.hdf5"
        if source['file']!=expected or source['repo']!='theaiinstitute/q2rl_robomimic_datasets' or source['sha256']!=digest:
            raise ValueError('Dataset provenance or checksum mismatch')
    reference = None
    if cfg.get('reference_bc_masks_verified'):
        report_path = Path(cfg['reference_checkpoint_audit'])
        reference = json.loads(report_path.read_text())
        train_config = reference['config']['train']
        if (train_config['hdf5_filter_key'] != cfg['train_mask'] or
                train_config['hdf5_validation_filter_key'] != cfg['valid_mask'] or
                not train_config['data'].endswith(f"/{cfg['task']}/ph/low_dim_v141.hdf5")):
            raise ValueError('Reference checkpoint BC data or masks do not match')
        if sha256(reference['path']) != reference['sha256']:
            raise ValueError('Reference checkpoint checksum mismatch')
        reference = dict(audit_path=str(report_path), audit_sha256=sha256(report_path),
                         checkpoint_sha256=reference['sha256'])
    with h5py.File(path,'r') as f:
        meta=json.loads(f['data'].attrs['env_args'])
        if meta['env_name']!=TASKS[cfg['task']]:
            raise ValueError('Dataset task mismatch')
        if meta.get('env_version')!='1.4.1':
            raise ValueError(f"Expected dataset env_version=1.4.1, got {meta.get('env_version')!r}. No automatic conversion.")
        masks={k:[x.decode() if isinstance(x,bytes) else str(x) for x in v[:]] for k,v in f['mask'].items()}
        train,valid=cfg['train_mask'],cfg['valid_mask']
        if train not in masks or valid not in masks:
            raise ValueError(f'Required masks {train}, {valid}; available: {list(masks)}')
        if set(masks[train]) & set(masks[valid]): raise ValueError('Overlapping train/validation masks')
        dims={k:list(f['data'][masks[train][0]]['obs'][k].shape[1:]) for k in KEYS}
        action=f['data'][masks[train][0]]['actions']
        if action.shape[-1]!=7: raise ValueError('Only single-step 7D Panda commands supported')
        names=sorted(masks[train],key=lambda x:int(x.split('_')[-1]))
        if cfg.get('debug_demos') is not None:
            names=names[:cfg['debug_demos']]
        return {'path':str(path.resolve()),'sha256':digest, 'env':meta,'source':source,
                'source_verified':source is not None,
                'dataset_family':'robomimic_v0.1','variant':'ph/low_dim_v141',
                'masks':masks,'train_demos':names,'valid_demos':masks[valid],
                'observation_keys':KEYS,'observation_shapes':dims,
                'train_mask':train,'valid_mask':valid,
                'reference_bc_masks_verified':reference is not None,
                'reference_checkpoint':reference,
                'train_transitions':sum(len(f['data'][n]['actions']) for n in names),
                'valid_transitions':sum(len(f['data'][n]['actions']) for n in masks[valid]),
                'debug_subset':cfg.get('debug_demos') is not None}


def load_demos(cfg, manifest):
    def read(names):
        states, actions=[],[]
        with h5py.File(cfg['dataset'],'r') as f:
            for name in names:
                d=f['data'][name]
                states.append(np.concatenate([d['obs'][k][:] for k in KEYS],-1))
                actions.append(d['actions'][:])
        s,a=np.concatenate(states),np.concatenate(actions)
        if not np.isfinite(s).all() or not np.isfinite(a).all() or np.max(np.abs(a))>1:
            raise ValueError('Invalid original observations/actions')
        return s,a
    return read(manifest['train_demos']),read(manifest['valid_demos'])
