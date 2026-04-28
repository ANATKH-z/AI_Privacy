"""
Client implementation for Federated Learning.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from collections import OrderedDict


class Client:
    """
    Represents a client in the federated learning system.
    """
    def __init__(self, client_id, dataset, batch_size=32, learning_rate=0.01):
        """
        Initialize a client.
        
        Args:
            client_id: Unique identifier for the client
            dataset: Local dataset (PyTorch Subset)
            batch_size: Batch size for local training
            learning_rate: Learning rate for SGD
        """
        self.client_id = client_id
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        self.model = None
        self.global_model_state = None  # Store global model w_G^t
    
    def set_model(self, model):
        """
        Set the global model for this client.
        
        Args:
            model: PyTorch model (will be copied)
        """
        self.model = deepcopy(model)
        # Store global model state w_G^t for computing update
        self.global_model_state = model.state_dict()

    def train_local(self, num_epochs=1):
        """
        Perform local training following standard FL: Δw_i^{t+1} = w_i^{t+1} - w_G^t

        Args:
            num_epochs: Number of local training epochs

        Returns:
            Update delta: Δw_i^{t+1} = w_i^{t+1} - w_G^t
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")

        # Todo: Implement local training
        # Store global model at start: w_G^t
        w_G_t = deepcopy(self.global_model_state)

        # Get local model after training: w_i^{t+1}
        device = next(self.model.parameters()).device
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            for data, target in self.data_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        w_i_t_plus_1 = self.model.state_dict()

        # Compute update: Δw_i^{t+1} = w_i^{t+1} - w_G^t
        update_delta = OrderedDict()
        for key in w_G_t.keys():
            update_delta[key] = w_i_t_plus_1[key] - w_G_t[key]

        return update_delta
    
    def get_dataset_size(self):
        """
        Get the size of the client's local dataset.
        
        Returns:
            Number of samples in the dataset
        """
        return len(self.dataset)

