# Federated Learning for Aircraft Health Monitoring
```
This project implements a Federated Learning framework for RUL and Fault Detection using CMPASS dataset. It enables multiple airline companies with differnt sensor configurations to collaborattively train a shared model using client-specific encoders, FedProx, client-side norm clipping and Krum aggregation without sharing raw sensor data. 
```

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
├── encoder_client_0.pth (trained weight of client's private encoder)
├── encoder_client_1.pth (trained weight of client's private encoder)
├── encoder_client_2.pth (trained weight of client's private encoder)
├── encoder_client_3.pth (trained weight of client's private encoder)
│
├── global_shared_model.pth (contiains final trained parameters of the shared model)
│
├── scaler.pkl (trained normalization model)
│
├── client_0_predictions.csv (contians prediction data)
├── client_1_predictions.csv (contians prediction data)
├── client_2_predictions.csv (contians prediction data)
├── client_3_predictions.csv (contians prediction data)
│
└── README.md
```

# Step 1 : Install Anaconda

Download

https://www.anaconda.com/download

Install Anaconda.

Restart terminal.

---

# Step 2 : Create Environment using the Anacaonda Navigator
# Step 3 : Install dependencies using the Anaconda Navigator
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



