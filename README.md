# DP-DGAD: A Generalist Dynamic Graph Anomaly Detector with Dynamic Prototypes
This repository is the official implementation of paper "DP-DGAD: A Generalist Dynamic Graph Anomaly Detector with Dynamic Prototypes"
	![](https://github.com/Jackmeory/KDD2026-DP-DGAD/blob/main/pipeline.png)

## Requirements
To install requirements:
```Python
pip install -r requirements.txt 
```
## Pretrain on source datasets
To train DP-DGAD on source datasests, please run the following code
```Python
python train_source.py 
```
## Update and test on target datasets
To update DP-DGAD on target datasests, please run the following code
```Python
python infer_target.py 
```
## Additional Notes
This code take reference from [GeneralDyG](https://github.com/YXNTU/GeneralDyG?tab=readme-ov-file), AAAI 2025; [ARC](https://github.com/yixinliu233/ARC?tab=readme-ov-file), NeurIPS 2024; [AnomalyGFM](https://github.com/mala-lab/anomalygfm), KDD 2025. We want to express our sincere thanks to above authors for their well structured open-source code!
