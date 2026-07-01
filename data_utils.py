import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib


SENSOR_COLS = [f"s{i}" for i in range(1, 22)]


def load_cmapss(path):
    columns = (
        ["engine_id", "cycle"]
        + [f"setting{i}" for i in range(1, 4)]
        + SENSOR_COLS
    )

    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = columns

    return df


def add_train_rul_and_fault(df, fault_threshold=30):
    max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.columns = ["engine_id", "max_cycle"]

    df = df.merge(max_cycle, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["fault"] = (df["RUL"] <= fault_threshold).astype(int)

    df.drop(columns=["max_cycle"], inplace=True)

    return df


def add_test_rul_and_fault(df, rul_path, fault_threshold=30):
    final_rul = pd.read_csv(rul_path, sep=r"\s+", header=None)
    final_rul.columns = ["final_RUL"]

    engine_ids = sorted(df["engine_id"].unique())

    rul_map = {
        engine_id: final_rul.iloc[i]["final_RUL"]
        for i, engine_id in enumerate(engine_ids)
    }

    max_cycle = df.groupby("engine_id")["cycle"].max().to_dict()

    rul_values = []

    for _, row in df.iterrows():
        eid = row["engine_id"]
        rul = max_cycle[eid] - row["cycle"] + rul_map[eid]
        rul_values.append(rul)

    df["RUL"] = rul_values
    df["fault"] = (df["RUL"] <= fault_threshold).astype(int)

    return df


def normalize_train(df):
    scaler = MinMaxScaler()
    df[SENSOR_COLS] = scaler.fit_transform(df[SENSOR_COLS])

    joblib.dump(scaler, "scaler.pkl")

    return df


def normalize_test(df):
    scaler = joblib.load("scaler.pkl")
    df[SENSOR_COLS] = scaler.transform(df[SENSOR_COLS])

    return df


def split_clients(df):
    sensor_counts = [21, 10, 15, 18]

    engine_ids = df["engine_id"].unique()
    np.random.shuffle(engine_ids)

    splits = np.array_split(engine_ids, 4)

    clients = {}

    for i, ids in enumerate(splits):
        client_df = df[df["engine_id"].isin(ids)].copy()

        sensors = SENSOR_COLS[:sensor_counts[i]]

        clients[i] = {
            "df": client_df,
            "sensors": sensors,
            "input_dim": sensor_counts[i]
        }

    return clients


def create_sequences(df, sensors, seq_len=30):
    X, y_rul, y_fault = [], [], []

    for eid in df["engine_id"].unique():
        engine_df = df[df["engine_id"] == eid]

        sensor_values = engine_df[sensors].values
        rul_values = engine_df["RUL"].values
        fault_values = engine_df["fault"].values

        for i in range(len(engine_df) - seq_len):
            X.append(sensor_values[i:i + seq_len])
            y_rul.append(rul_values[i + seq_len])
            y_fault.append(fault_values[i + seq_len])

    X = np.array(X, dtype=np.float32)
    y_rul = np.array(y_rul, dtype=np.float32).reshape(-1, 1)
    y_fault = np.array(y_fault, dtype=np.float32).reshape(-1, 1)

    return X, y_rul, y_fault


def create_dataloader(client_data, batch_size=32, seq_len=30):
    X, y_rul, y_fault = create_sequences(
        client_data["df"],
        client_data["sensors"],
        seq_len
    )

    dataset = TensorDataset(
        torch.tensor(X),
        torch.tensor(y_rul),
        torch.tensor(y_fault)
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader