# Federated Learning for Aircraft Health Monitoring

This project implements a Federated Learning framework for RUL and Fault Detection using CMPASS dataset. It enables multiple airline companies with differnt sensor configurations to collaborattively train a shared model using client-specific encoders, FedProx, client-side norm clipping and Krum aggregation without sharing raw sensor data. 


# Project Directory

```

├──data/
|   ├──train_FD001.txt
|   ├──test_FD001.txt
|   └─RUL_FD001.txt
│
├── data_utils.py
├── model.py
├── client.py
├── server.py
├── krum.py
├── test_utils.py
├── federated_learning.ipynb
│
├── encoder_client_0.pth 
├── encoder_client_1.pth 
├── encoder_client_2.pth 
├── encoder_client_3.pth 
│
├── global_shared_model.pth 
│
├── scaler.pkl 
│
├── client_0_predictions.csv 
├── client_1_predictions.csv 
├── client_2_predictions.csv 
├── client_3_predictions.csv 
│
└── README.md
```

# Quick Start

## Step 1 : Clone Repository
```
git clone https://github.com/Sujit34/fl-aircraft-health-monitoring.git
```
## Step 2 : Install Anaconda
```
Download Anaconda from https://www.anaconda.com/download
Install Anaconda.
Restart terminal.
```

## Step 3 : Create Environment using the Anacaonda Navigator
## Step 4 : Install dependencies using the Anaconda Navigator
```
dependencies:
  - python=3.10
  - numpy
  - pandas  
  - scikit-learn
  - joblib
  - pytorch  
  - torch  
  - flower
```
## Step 5 : Execution
```
Launch Jupyter and upload the data folder, all the *.py files and federated_learning.ipynb file in the root directory.  Then execute every cell of federated_learning.ipynb file.

It will save follwoing files in the root directory :

Trained weight of client's private encoder
├── encoder_client_0.pth 
├── encoder_client_1.pth 
├── encoder_client_2.pth 
├── encoder_client_3.pth

Final trained parameters of the shared model
├── global_shared_model.pth

Trained normalization model
├── scaler.pkl

Prediction result
├── client_0_predictions.csv
├── client_1_predictions.csv
├── client_2_predictions.csv
├── client_3_predictions.csv

```



