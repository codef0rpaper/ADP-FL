# ADP-FL

The code implement of paper "Adaptive Differential Privacy via Gradient Components in Medical Federated learning"

## Abstract

The integration of AI in the healthcare sector has marked significant advancements, and Federated Learning has further facilitated the amalgamation of Federated Medical Imaging. However, this integration has also sparked concerns regarding data privacy. Incorporating Differential Privacy into gradients effectively mitigates privacy leaks but at the cost of impacting model accuracy. Current research delves into DP within FL, with a focus on strategies for privacy budget allocation and noise addition. Nevertheless, the dynamic privacy requirements and resource optimization for actual medical applications are often overlooked, leading to resource wastage. This study introduces an innovative algorithm based on gradient component for adaptive noise scale optimization and privacy budget allocation, thereby enhancing privacy management while maintaining model accuracy. Our findings reveal that, compared to traditional DP techniques, our approach achieves an average accuracy improvement of 3.47% in RSNA-ICH Acc and an enhancement of up to 172.33% in Dice score for Prostate MRI Dice under various privacy budgets, demonstrating the substantial efficacy of our method in the domain of federated medical imaging.

## Overview

![](https://github.com/codef0rpaper/ADP-FL/blob/main/image/overview.png)

<div align=center>  <img src="https://github.com/codef0rpaper/ADP-FL/blob/main/image/miccai.drawio.png" width=50%></div>

## Requirements

python >= 3.11

```bash
pip install -r requirements.txt
```

## Repo Structure

```
.
├── README.md
├── dataset // folder for dataset
│   ├── Prostate
│   │   ├── I2CVB.json
│   │   ├── MSD.json
│   │   ├── NCI_ISBI_3T.json
│   │   ├── NCI_ISBI_DX.json
│   │   ├── Promise12.json
│   │   └── ProstateX.json
│   ├── RSNA-ICH
│   │   └── binary_25k
│   │       ├── df_binary25k.csv
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── validate.csv
│   └── dataset.py 
├── fed // Class for center and client
│   ├── global_trainer.py // class for center server
│   └── local_trainer.py // class for distributed client
├── fed_main.py // main function
├── image // image in README
│   ├── miccai.drawio.png
│   └── overview.png
├── nets // folder for models
│   ├── __init__.py
│   └── models.py // include Unets and DenseNet121
├── requirements.txt 
├── requirements_cpu.txt
├── run_exp.sh
└── utils
    ├── __init__.py
    ├── datasets.py // method of dataset processing
    ├── loss.py // Loss Function
    ├── nova_utils.py // setup function
    ├── util.py // setup function
    └── workflow.py // setup function
```

## Usage

```bash
bash run_exp.sh dpsgd
```
