"""
Server implementation for Secure Federated Averaging.
"""
import torch
from collections import OrderedDict
from secure_aggregation import SecureAggregator


class SecureServer:
    """
    Server that coordinates federated learning with secure aggregation.
    
    This implements the secure aggregation mechanism introduced in Exercise 2.
    With secure aggregation, the server only sees the sum of updates,
    not individual client updates. This protects privacy but enables
    poisoning attacks (as demonstrated in Exercise 3).
    """
    def __init__(self, model, clients, use_secure_aggregation=True):
        """
        Initialize the server.
        
        Args:
            model: Global model (PyTorch model)
            clients: List of Client objects
            use_secure_aggregation: Whether to use secure aggregation
        """
        self.model = model
        self.clients = clients
        self.use_secure_aggregation = use_secure_aggregation
        self.total_samples = sum(client.get_dataset_size() for client in clients)
        
        if use_secure_aggregation:
            self.secure_aggregator = SecureAggregator()
    
    def aggregate(self, client_updates, round_num=0, eta=1.0):
        """
        Aggregate client updates following standard FL: w_G^{t+1} = w_G^t + η Δw_agg^{t+1}.
        If secure aggregation is enabled: use secure_aggregator.mask_state_dict(...) per client,
        then secure_aggregator.aggregate_state_dicts(...). Otherwise: weighted FedAvg.
        See TODOs in the method body for what you need to implement.
        
        Args:
            client_updates: List of (client_id, update_delta) tuples
                          where update_delta = Δw_i^{t+1} = w_i^{t+1} - w_G^t
            round_num: Current communication round (for logging)
            eta: Learning rate/step size for global model update (default: 1.0)
        
        Returns:
            Updated global model state dict: w_G^{t+1} = w_G^t + η Δw_agg^{t+1}
        """
        # Get current global model: w_G^t
        w_G_t = self.model.state_dict()
        
        if self.use_secure_aggregation:
            # TODO (Exercise 3): Implement secure aggregation so the server only sees the sum of updates.
            # Use the secure_aggregator (self.secure_aggregator) and follow these steps:
            #
            # Step 1: Compute FedAvg weights for each client.
            #   - num_clients = len(client_updates)
            #   - For each (client_id, _) in client_updates: weight = self.clients[client_id].get_dataset_size() / self.total_samples
            #
            # Step 2: For each client, build the weighted update and mask it (simulating client-side masking).
            #   - For each (client_id, update_delta) in client_updates:
            #     - weighted_update = OrderedDict: for each key in update_delta, weighted_update[key] = weight * update_delta[key]
            #     - Call: self.secure_aggregator.mask_state_dict(weighted_update, client_id, num_clients, round_num)
            #     - Append the returned masked state dict to a list (e.g. masked_updates).
            #
            # Step 3: Aggregate the masked updates (server only sees masked; masks cancel when summed).
            #   - Call: aggregated_update = self.secure_aggregator.aggregate_state_dicts(masked_updates)
            #
            # Step 4: Update global model: w_G^{t+1} = w_G^t + eta * aggregated_update
            #   - Build w_G_t_plus_1: for each key in w_G_t, w_G_t_plus_1[key] = w_G_t[key] + eta * aggregated_update[key]
            #   - return w_G_t_plus_1
            #
            num_clients = len(client_updates)
            masked_updates = []

            # Step 1 & 2: Weight and apply a mask
            for client_id, update_delta in client_updates:
                client_obj = next(c for c in self.clients if c.client_id == client_id)
                weight = client_obj.get_dataset_size() / self.total_samples

                weighted_update = OrderedDict()
                for key in update_delta.keys():
                    weighted_update[key] = weight * update_delta[key]

                # The simulation client adds an encryption mask before sending
                masked_update = self.secure_aggregator.mask_state_dict(
                    weighted_update, client_id, num_clients, round_num
                )
                masked_updates.append(masked_update)

            # Step 3: the masks cancel each other out during aggregation
            aggregated_update = self.secure_aggregator.aggregate_state_dicts(masked_updates)

            # Step 4: Update the global model
            w_G_t_plus_1 = OrderedDict()
            for key in w_G_t.keys():
                w_G_t_plus_1[key] = w_G_t[key] + eta * aggregated_update[key]

            return w_G_t_plus_1
        else:
            # TODO (Exercise 3): Standard weighted aggregation (FedAvg) when secure aggregation is disabled.
            # Implement: Δw_agg^{t+1} = Σ_i (|D_i|/|D|) Δw_i^{t+1}, then w_G^{t+1} = w_G^t + η Δw_agg^{t+1}.
            #
            # Step 1: For each client_id in client_updates, get weight = self.clients[client_id].get_dataset_size() / self.total_samples.
            #
            # Step 2: Build aggregated_update (OrderedDict, same keys as update_delta). Initialize to zeros, then for each (client_id, update_delta) add weight * update_delta[key] to aggregated_update[key].
            #
            # Step 3: Build w_G_t_plus_1: for each key, w_G_t_plus_1[key] = w_G_t[key] + eta * aggregated_update[key]. Return w_G_t_plus_1.
            #
            aggregated_update = OrderedDict()
            for key in w_G_t.keys():
                aggregated_update[key] = torch.zeros_like(w_G_t[key])

            for client_id, update_delta in client_updates:
                client_obj = next(c for c in self.clients if c.client_id == client_id)
                weight = client_obj.get_dataset_size() / self.total_samples

                for key in update_delta.keys():
                    aggregated_update[key] += weight * update_delta[key]

            w_G_t_plus_1 = OrderedDict()
            for key in w_G_t.keys():
                w_G_t_plus_1[key] = w_G_t[key] + eta * aggregated_update[key]

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

