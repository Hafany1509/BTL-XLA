import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger

import random
from copy import deepcopy
from dataset_layout import resolve_split
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


# test config
vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]


def get_val_opt(opt):
    val_opt = deepcopy(opt)
    val_opt.dataroot = opt.val_root
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True

    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    seed_torch(100)
    base_root = opt.dataroot
    try:
        opt.train_root = resolve_split(base_root, opt.train_split, opt.classes, opt.train_root)
        opt.val_root = resolve_split(base_root, opt.val_split, opt.classes, opt.val_root)
        if os.path.realpath(opt.train_root) == os.path.realpath(opt.val_root):
            raise ValueError('Training and validation must use different directories.')
    except ValueError as exc:
        raise SystemExit(str(exc))
    Testdataroot = opt.test_root or os.path.join(base_root, 'test')
    run_tests = not opt.skip_test and os.path.isdir(Testdataroot)
    if opt.test_root and not opt.skip_test and not os.path.isdir(Testdataroot):
        raise SystemExit('Test directory does not exist: ' + Testdataroot)
    if run_tests:
        missing = [v for v in vals if not os.path.isdir(os.path.join(Testdataroot, v))]
        if missing:
            raise SystemExit('Incomplete test suite at {}: missing {}. Use --skip_test to train/validate only.'.format(
                Testdataroot, ', '.join(missing)))
    opt.dataroot = opt.train_root
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)) )
    print('Train directory:', opt.train_root)
    print('Validation directory:', opt.val_root)
    print('Cross-generator test suite:', Testdataroot if run_tests else 'disabled (validation still runs)')
    val_opt = get_val_opt(opt)
    Testopt = deepcopy(val_opt)
    data_loader = create_dataloader(opt)
    print('Training images:', len(data_loader.dataset))
    if opt.check_data:
        for split, loader in [('train', data_loader), ('val', create_dataloader(val_opt))]:
            images, labels = next(iter(loader))
            if not set(labels.tolist()).issubset({0, 1}):
                raise SystemExit('Invalid binary labels in ' + split)
            print('{}: {} images; batch {}; labels {}'.format(
                split, len(loader.dataset), tuple(images.shape), sorted(set(labels.tolist()))))
        print('Data check passed. No model was trained.')
        raise SystemExit(0)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    model = Trainer(opt)

    def testmodel():
        if not run_tests:
            return
        print('*'*25);accs = [];aps = []
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        for v_id, val in enumerate(vals):
            Testopt.dataroot = '{}/{}'.format(Testdataroot, val)
            Testopt.classes = [] if multiclass[v_id] else ['']
            Testopt.no_resize = False
            Testopt.no_crop = True
            acc, ap, _, _, _, _ = validate(model.model, Testopt)
            accs.append(acc);aps.append(ap)
            print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
        print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25)
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    model.eval();testmodel();
    model.train()
    print(f'cwd: {os.getcwd()}')
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), "Train loss: {} at step: {} lr {}".format(model.loss, model.total_steps, model.lr))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

            if opt.save_latest_freq > 0 and model.total_steps % opt.save_latest_freq == 0:
                model.save_networks('latest')

        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 'changing lr at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.adjust_learning_rate()


        # Validation
        model.eval()
        acc, ap = validate(model.model, val_opt)[:2]
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        model.save_networks('latest')
        if opt.save_epoch_freq > 0 and (epoch + 1) % opt.save_epoch_freq == 0:
            model.save_networks(epoch + 1)
        testmodel()
        model.train()

    model.eval();testmodel()
    model.save_networks('last')
    train_writer.close()
    val_writer.close()
