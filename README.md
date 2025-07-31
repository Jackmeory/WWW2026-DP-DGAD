# DP-DGAD: A Generalist Dynamic Graph Anomaly Detector with Dynamic Prototypes
This repository is the official implementation of paper "DP-DGAD: A Generalist Dynamic Graph Anomaly Detector with Dynamic Prototypes"
	![](https://github.com/Jackmeory/KDD2026-DP-DGAD/blob/main/pipeline.png)

## Requirements
To install requirements:
```Python
pip install -r requirements.txt 
```
## Pretrain on source datasets
```Python
python train_source.py 
```
## Update and test on target datasets
```Python
python infer_target.py 
```
