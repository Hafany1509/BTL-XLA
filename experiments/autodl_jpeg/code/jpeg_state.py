"""Atomic, complete-epoch resume state for the JPEG experiment."""
import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch


def configuration(opt):
    keys = ('arch', 'classes', 'train_root', 'val_root', 'batch_size', 'num_threads',
            'loadSize', 'cropSize', 'no_flip', 'jpg_prob', 'jpg_qual', 'jpg_method',
            'blur_prob', 'blur_sig', 'test_jpeg_quality', 'optim', 'lr', 'beta1',
            'delr_freq', 'niter', 'serial_batches', 'class_bal')
    values = {key: getattr(opt, key) for key in keys}
    source = Path(__file__).resolve().parent
    values['source_sha256'] = {
        relative: hashlib.sha256((source / relative).read_bytes()).hexdigest()
        for relative in ('data/datasets.py', 'networks/resnet.py', 'networks/trainer.py',
                         'train.py', 'jpeg_state.py')}
    values['seed'] = 100
    return values


def save_training_state(trainer, opt, completed_epochs):
    state = dict(model=trainer.model.state_dict(), optimizer=trainer.optimizer.state_dict(),
                 completed_epochs=completed_epochs, total_steps=trainer.total_steps,
                 lr=trainer.lr, configuration=configuration(opt),
                 python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                 torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    path = Path(trainer.save_dir) / 'training_state.pth'
    temporary = path.with_name(path.name + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def restore_training_state(trainer, opt):
    state = torch.load(opt.resume_state, map_location='cpu', weights_only=False)
    if state['configuration'] != configuration(opt):
        raise ValueError('Resume settings or source code differ from the completed run.')
    trainer.model.load_state_dict(state['model'], strict=True)
    trainer.optimizer.load_state_dict(state['optimizer'])
    for optimizer_state in trainer.optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(trainer.device)
    trainer.total_steps, trainer.lr = state['total_steps'], state['lr']
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    print('Resumed after epoch', state['completed_epochs'], flush=True)
    return state['completed_epochs']
