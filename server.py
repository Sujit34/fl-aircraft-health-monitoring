import flwr as fl
import torch

from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

from krum import krum
from model import SharedLSTMModel


class KrumStrategy(fl.server.strategy.FedAvg):
    def __init__(self, num_malicious=1, **kwargs):
        super().__init__(**kwargs)
        self.num_malicious = num_malicious
        self.krum_history = []

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        updates = []

        for _, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            updates.append(weights)

        selected_weights, selected_index = krum(
            updates,
            num_malicious=self.num_malicious
        )

        self.krum_history.append({
            "round": server_round,
            "selected_client_update_index": selected_index
        })

        self.save_global_model(selected_weights)

        return ndarrays_to_parameters(selected_weights), {
            "selected_client_update_index": selected_index
        }

    def save_global_model(self, weights):
        model = SharedLSTMModel()
        state_dict = model.state_dict()

        new_state_dict = {}

        for key, value in zip(state_dict.keys(), weights):
            new_state_dict[key] = torch.tensor(value)

        model.load_state_dict(new_state_dict)
        torch.save(model.state_dict(), "global_shared_model.pth")