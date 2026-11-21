# Train NPR with the Hugging Face CNNDetection archives on AutoDL

The image labels remain `0_real` (0) and `1_fake` (1). The paper's training
selection remains `car,cat,chair,horse`; no image re-encoding or model changes
are required. Downloading from Hugging Face changes archive packaging/paths.

The CNNDetection author's current download script extracts the multipart 7z
into `progan_train.zip`, then unzips that file. The previous one-step 7z
instructions alone may leave no training images.
Source: https://github.com/PeterWang512/CNNDetection/blob/master/dataset/train/download_trainset.sh

## Install and extract

Keep AutoDL's matching CUDA-enabled torch/torchvision installation. Install
missing project dependencies (the original requirements omitted tensorboardX):

```bash
cd /root/autodl-tmp/NPR-DeepfakeDetection-main
python -m pip install scipy scikit-learn numpy opencv-python Pillow tensorboardX
apt-get update
apt-get install -y p7zip-full unzip
bash prepare_autodl_data.sh /root/autodl-tmp/CNNDetection
```

Run extraction only after all seven train archive parts and `progan_val.zip`
have finished downloading. It retains archives and existing images. Check
free space with `df -h /root/autodl-tmp`: the intermediate ZIP and extracted
images require additional space beyond the downloads. Already extracted data
can be used directly; re-extraction is unnecessary.

## Check paths without importing PyTorch

```bash
python dataset_layout.py --dataroot /root/autodl-tmp/CNNDetection
```

The resolver searches at most four folder levels for `train`, `progan_train`,
`val`, or `progan_val` directories containing the selected categories and
binary labels. It rejects ambiguous layouts and never substitutes training
images for validation. For arbitrary names, pass `--train_root` and
`--val_root` to either the checker or train.py. These paths must directly
contain `car`, `cat`, `chair`, and `horse`.

## Read a batch before training

```bash
python train.py \
  --dataroot /root/autodl-tmp/CNNDetection \
  --classes car,cat,chair,horse \
  --batch_size 32 --num_threads 0 --skip_test --check_data \
  --checkpoints_dir /root/autodl-tmp/checkpoints --name data-check-
```

This builds the actual image loaders, prints image counts and batch shapes,
and exits without constructing a model or requiring a GPU. It checks one
batch per split, not every image's integrity.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataroot /root/autodl-tmp/CNNDetection \
  --classes car,cat,chair,horse \
  --name 4class-resnet-car-cat-chair-horse- \
  --batch_size 32 --num_threads 8 \
  --delr_freq 10 --lr 0.0002 --niter 50 \
  --checkpoints_dir /root/autodl-tmp/checkpoints \
  --skip_test
```

`--skip_test` skips only the cross-generator benchmark; held-out validation
still runs every epoch. To evaluate the original eight-generator suite while
training, remove `--skip_test` and supply `--test_root /path/to/test-images`,
where that directory directly contains `progan`, `stylegan`, `stylegan2`,
`biggan`, `cyclegan`, `stargan`, `gaugan`, and `deepfake`. Missing test data is
reported before training. Without an explicit test root, the original
`<dataroot>/test` layout is used when present; otherwise testing is disabled.

The network, transforms, optimizer, seed, and learning-rate schedule are
unchanged. Added checkpoint writes honor the existing save frequency options
and save `model_epoch_latest.pth` at every epoch end. These retain the original
weights-only format; they are not full optimizer/RNG resume checkpoints, and
this patch does not repair the original `--continue_train` resume workflow.

## Local checks for this patch

```bash
python -m unittest discover -s tests -p test_dataset_layout.py -v
```

Folder resolution is tested using synthetic directory trees. Full image-loader
and GPU training verification must run on AutoDL with the extracted data and
the project's PyTorch dependencies installed.
