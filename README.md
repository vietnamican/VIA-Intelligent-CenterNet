# via-trafficsign-detection

## Progress

- [x] Data.

- [x] Backbone.

    - [x] VGG-like.

    - [x] MobilenetV2.

- [x] Pretrained

    - [x] VGG-like.

    - [ ] MobilenetV2.

- [x] Training code.

- [x] Inference code.

- [ ] Demo code.

    - [x] On images.

    - [ ] On video.

## Installaition
### Requirements
- python>=3.7
- torch==1.7.0
- torchvision==0.8.1

### Install packages
Please execute below command to install essential packages
```
    $ conda install --file requirements.txt
```

## Dataset
Please access to this [link](https://github.com/makerhanoi/via-datasets) and follow instructions to download via-trafficsign dataset. Extract and put it in the project root directory.

### Train on your own dataset
See [this](readme/train_custom_dataset.md)
## Directory structure
```
VIA-Intelligent-CenterNet
├── assets
|     ├── result.png
|
├── models                      # source code for model
|     ├── ...
|
├── readme
|     ├── ...
|
├── samples
|     ├── ...
|
├── via-trafficsign
|     ├── images
│     |     ├── train
│     |     |     ├── 00001.jpg
|     |     |     ├── ...
|     |     |     ├── 10292.jpg
│     |     ├── val
│     |     |     ├── 00001.jpg
|     |     |     ├── ...
|     |     |     ├── 00588.jpg
|     |
|     ├── labels
|     |     ├── train   
│     |     |     ├── 00001.txt
|     |     |     ├── ...
|     |     |     ├── 10292.txt
│     |     ├── val
│     |     |     ├── 00001.txt
|     |     |     ├── ...
|     |     |     ├── 00588.txt
|     
├── .gitignore
├── config.py
├── datasets.py
├── inference.py
├── README.md
├── requirements.txt
├── train.py
├── utils.py
```
## Training
```
    $ python train.py --train-image-dir=$TRAIN_IMAGE_DIR \
                      --train-label-dir=$TRAIN_LABEL_DIR \
                      --val-image-dir=$VAL_IMAGE_DIR \
                      --val-label-dir=$VAL_LABEL_DIR
```
Example
```
    $ python train.py --train-image-dir='via-trafficsign/images/train' \
                      --train-label-dir='via-trafficsign/labels/train' \
                      --val-image-dir='via-trafficsign/images/val' \
                      --val-label-dir='via-trafficsign/labels/val'
```
## Testing
```
    $ python inference.py --val-image-dir=$VAL_IMAGE_DIR \
                          --val-label-dir=$VAL_LABEL_DIR \
                          --checkpoint=$CHECKPOINT \
                          --outdir=$OUTDIR
```
Example
```
    $ python inference.py --val-image-dir='via-trafficsign/images/val' \
                          --val-label-dir='via-trafficsign/labels/val' \
                          --checkpoint='archives/centernet_vgg.ckpt' \
                          --outdir='result'
```

## Demo (Unavailable now)
```
    $ python demo.py --image-path=$IMAGE_PATH \
                     --checkpoint=$CHECKPOINT \
                     --outpath=$OUTPATH
```
Example
```
    $ python demo.py --image-path='samples/1.jpg' \
                     --checkpoint='archives/centernet_vgg.ckpt' \
                     --outpath='result/1.jpg'
```
## Pretrained model
Backbone | Parameters | Matmuls | Pretrained
| --- | ---: | ---: | :--- |
VGG-like | 2.5M | 24.47G | [Link](https://github.com/vietnamican/VIA-Intelligent-CenterNet/releases/tag/v0.1.1)
Mobilenetv2 | 2.0M | 24.78G | N/A 

Download pretrained-models and put it in ```archives/``` directory. 

## Result
![alt text](assets/result.png)

