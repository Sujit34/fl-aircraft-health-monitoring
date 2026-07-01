import flwr as fl
import torch
import torch.nn as nn

from model import ClientEncoder, SharedLSTMModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class HeterogeneousFLClient(fl.client.NumPyClient):
    def __init__(self, client_id, input_dim, trainloader, mu=0.01):
        self.client_id = client_id
        self.input_dim = input_dim
        self.trainloader = trainloader
        self.mu = mu

        self.encoder = ClientEncoder(input_dim=input_dim).to(DEVICE)
        self.shared_model = SharedLSTMModel().to(DEVICE)

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.shared_model.parameters()),
            lr=0.001
        )

        self.rul_loss_fn = nn.MSELoss()
        self.fault_loss_fn = nn.BCELoss()

    def get_parameters(self, config=None):
        return [
            val.cpu().numpy()
            for _, val in self.shared_model.state_dict().items()
        ]

    def set_parameters(self, parameters):
        state_dict = self.shared_model.state_dict()

        new_state_dict = {}

        for key, value in zip(state_dict.keys(), parameters):
            new_state_dict[key] = torch.tensor(value)

        self.shared_model.load_state_dict(new_state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        global_params = [
            p.clone().detach()
            for p in self.shared_model.parameters()
        ]

        self.encoder.train()
        self.shared_model.train()

        epochs = 1
        max_norm = 1.0
        total_loss = 0.0

        for _ in range(epochs):
            for X, y_rul, y_fault in self.trainloader:
                X = X.to(DEVICE)
                y_rul = y_rul.to(DEVICE)
                y_fault = y_fault.to(DEVICE)

                self.optimizer.zero_grad()

                z = self.encoder(X)
                pred_rul, pred_fault = self.shared_model(z)

                rul_loss = self.rul_loss_fn(pred_rul, y_rul)
                fault_loss = self.fault_loss_fn(pred_fault, y_fault)

                task_loss = rul_loss + fault_loss

                prox_term = 0.0

                for local_param, global_param in zip(
                    self.shared_model.parameters(),
                    global_params
                ):
                    prox_term += torch.norm(local_param - global_param) ** 2

                loss = task_loss + (self.mu / 2) * prox_term

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) +
                    list(self.shared_model.parameters()),
                    max_norm=max_norm
                )

                self.optimizer.step()

                total_loss += loss.item()

        
        torch.save(
            self.encoder.state_dict(),
            f"encoder_client_{self.client_id}.pth"
        )

        return (
            self.get_parameters(),
            len(self.trainloader.dataset),
            {
                "client_id": self.client_id,
                "input_dim": self.input_dim,
                "fedprox_loss": total_loss
            }
        )