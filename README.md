## Requirement

+ Create a conda virtual environment and activate it.

```
conda create --name ood2023 python=3.8.13
conda activate ood2023
```

+ Install Pytorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```
conda install pytorch==1.13.1 torchvision==0.14.1 -c pytorch
```

+ Install the dependent libraries.

```
pip install -r requirements.txt
```

## Data Preparation

+ Download the data set from the official website and organize it into the following data structure.

```
dataset/OODCV2023
    train
	Images
            aeroplane/
	    ...
            train/
    	labels.csv
    phase1-test-images
    phase2-test-images

```

## Getting Started

### train

+ We use one Nvidia Tesla 4090 (24G) GPU for training. You need to modify the params in the `configs/beitv2.yaml` file or `configs/eva02.yaml` before training, like: `data`  `model` or pretrained weight path in `pretrained_cfg`. Run different bash files for different models.

```
./train_eva02.sh 0 or ./train_beitv2.sh 0
```

### Test

- You need to modify the `checkpoint_list` in the `eval_tta.py` file before testing.

```
python eval_tta.py
```
