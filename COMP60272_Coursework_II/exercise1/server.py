"""
Server implementation for Federated Averaging.
"""
import torch
from collections import OrderedDict


class Server:
    """
    Server that coordinates federated learning using Federated Averaging.
    """
    def __init__(self, model, clients):
        """
        Initialize the server.
        
        Args:
            model: Global model (PyTorch model)
            clients: List of Client objects
        """
        self.model = model
        self.clients = clients
        self.total_samples = sum(client.get_dataset_size() for client in clients)

    def aggregate(self, client_updates, eta=1.0):
        """
        Aggregate client updates following standard FL: w_G^{t+1} = w_G^t + η Δw_agg^{t+1}

        FedAvg: Δw_agg^{t+1} = Σ (|D_i|/|D|) Δw_i^{t+1}

        Args:
            client_updates: List of (client_id, update_delta) tuples
                          where update_delta = Δw_i^{t+1} = w_i^{t+1} - w_G^t
            eta: Learning rate/step size for global model update (default: 1.0)

        Returns:
            Updated global model state dict: w_G^{t+1} = w_G^t + η Δw_agg^{t+1}
        """
        # TODO: Implement server aggregation
        # Get current global model: w_G^t
        w_G_t = self.model.state_dict()

        # Get dataset sizes for weighting
        client_sizes = {client.client_id: client.get_dataset_size() for client in self.clients}
        total_samples = sum(client_sizes[client_id] for client_id, _ in client_updates)

        # Weighted aggregation of updates: Δw_agg^{t+1} = Σ (|D_i|/|D|) Δw_i^{t+1}
        delta_w_agg = OrderedDict()
        for key in w_G_t.keys():
            # Initialise as a zero tensor of the same size as the global model
            delta_w_agg[key] = torch.zeros_like(w_G_t[key])

        for client_id, update_delta in client_updates:
            weight = client_sizes[client_id] / total_samples
            for key in delta_w_agg.keys():
                delta_w_agg[key] += weight * update_delta[key]

        # Update global model: w_G^{t+1} = w_G^t + η Δw_agg^{t+1}
        w_G_t_plus_1 = OrderedDict()
        for key in w_G_t.keys():
            w_G_t_plus_1[key] = w_G_t[key] + eta * delta_w_agg[key]

        return w_G_t_plus_1
    
    def update_global_model(self, aggregated_state):
        """
        Update the global model with aggregated state.
        
        Args:
            aggregated_state: Aggregated model state dict
        """
        self.model.load_state_dict(aggregated_state)
    
    def broadcast_model(self):
        """
        Broadcast the current global model to all clients.
        """
        for client in self.clients:
            client.set_model(self.model)
    
    def get_model(self):
        """
        Get the current global model.
        
        Returns:
            Global model
        """
        return self.model

