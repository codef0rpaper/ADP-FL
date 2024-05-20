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

### Packages in requirements

```
absl-py==1.4.0
attrs==23.2.0
cachetools==5.3.2
certifi==2022.12.7
charset-normalizer==2.1.1
dm-tree==0.1.8
dp-accounting==0.4.3
filelock==3.9.0
fsspec==2023.4.0
google-auth==2.27.0
google-auth-oauthlib==1.2.0
grpcio==1.60.1
idna==3.4
Jinja2==3.1.2
joblib==1.3.2
Markdown==3.5.2
MarkupSafe==2.1.3
MedPy==0.4.0
monai==1.3.0
mpmath==1.3.0
networkx==3.2.1
numpy==1.26.3
oauthlib==3.2.2
opacus==1.4.0
opencv-python==4.9.0.80
opt-einsum==3.3.0
pandas==2.2.0
pillow==10.2.0
protobuf==4.23.4
pyarrow==15.0.0
pyasn1==0.5.1
pyasn1-modules==0.3.0
python-dateutil==2.8.2
pytz==2024.1
requests==2.28.1
requests-oauthlib==1.3.1
rsa==4.9
scikit-learn==1.4.0
scipy==1.12.0
SimpleITK==2.3.1
six==1.16.0
sympy==1.12
tensorboard==2.15.1
tensorboard-data-server==0.7.2
threadpoolctl==3.2.0
torch==2.2.0
torchaudio==2.2.0
torchvision==0.17.0
typing_extensions==4.8.0
tzdata==2023.4
urllib3==1.26.13
Werkzeug==3.0.1
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

### Files Description
- `./fed_main.py` : The script orchestrates the training process for a federated learning model, coordinating between multiple clients and a central server to aggregate updates, handle data splitting and loading, manage logging, and ensure proper model training and evaluation across different sites using PyTorch.

- `./fed/global_trainer.py` : The script coordinates the training process for a federated learning model. It manages the interaction between multiple clients and a central server, handles data splitting and loading, and ensures proper model training and evaluation across different sites using PyTorch, all while logging progress and metrics. The script includes detailed handling of differential privacy, adaptive training rounds, and different aggregation methods like FedSGD, FedAdam, and FedRMSprop.

- `./fed/local_trainer.py` : The script is a detailed implementation of a federated learning local update procedure using PyTorch. This script focuses on training local models with differentially private mechanisms to ensure privacy during the federated learning process. 

- `./nets/models.py` : The script provided defines two neural network models in PyTorch: UNet and DenseNet.

- `./utils/datasets.py` : The script provides a class for loading and processing medical image datasets, including the ability to load and preprocess images from different sources, such as the ProstateX dataset, the RSNA Intracranial Hemorrhage dataset, and the NCI-ISBI Challenge 2013 dataset. The script also includes methods for data augmentation and normalization.

- `./utils/loss.py` : The script provides a class for computing the loss function for the federated learning model. The script includes the ability to compute the cross-entropy loss and the Dice loss, which are commonly used in medical image segmentation tasks.

- `./utils/nova_utils.py` : The script provides a class for handling the Nova environment, including the ability to set up the environment, load the dataset, and train the model.

- `./utils/util.py` : The script provides several utility functions and setups commonly used in federated learning (FL) and deep learning experiments. These include logging setup, argument parsing, and various metric calculations. 

## Usage

```bash
bash run_exp.sh dpsgd
```
