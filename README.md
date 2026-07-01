# Federated Learning for Aircraft Health Monitoring

# Project Directory

```
│
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



