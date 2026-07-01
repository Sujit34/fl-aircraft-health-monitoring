import torch
import pandas as pd

from data_utils import (
    load_cmapss,
    add_test_rul_and_fault,
    normalize_test,
    split_clients,
    create_dataloader
)

from model import ClientEncoder, SharedLSTMModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_client(client_id, client_data):

    input_dim = client_data["input_dim"]

    testloader = create_dataloader(
        client_data,
        batch_size=32,
        seq_len=30
    )

    encoder = ClientEncoder(input_dim=input_dim).to(DEVICE)
    shared_model = SharedLSTMModel().to(DEVICE)

    encoder.load_state_dict(
        torch.load(
            f"encoder_client_{client_id}.pth",
            map_location=DEVICE
        )
    )

    shared_model.load_state_dict(
        torch.load(
            "global_shared_model.pth",
            map_location=DEVICE
        )
    )

    encoder.eval()
    shared_model.eval()

    true_rul = []
    pred_rul = []

    true_fault = []
    pred_fault = []

    with torch.no_grad():

        for X, y_rul, y_fault in testloader:

            X = X.to(DEVICE)

            latent = encoder(X)

            rul, fault = shared_model(latent)

            true_rul.extend(y_rul.numpy().flatten())
            pred_rul.extend(rul.cpu().numpy().flatten())

            true_fault.extend(y_fault.numpy().astype(int).flatten())

            pred_fault.extend(
                (fault.cpu().numpy().flatten() >= 0.5).astype(int)
            )

    results = pd.DataFrame({
        "True_RUL": true_rul,
        "Pred_RUL": pred_rul,
        "True_Fault": true_fault,
        "Pred_Fault": pred_fault
    })

    results.to_csv(
        f"client_{client_id}_predictions.csv",
        index=False
    )

    return results


def run_test(test_path,rul_path):

    df = load_cmapss(test_path)

    df = add_test_rul_and_fault(df,rul_path)

    df = normalize_test(df)

    clients = split_clients(df)

    results = {}

    for cid, client_data in clients.items():
        results[cid] = test_client(cid, client_data)

    return results