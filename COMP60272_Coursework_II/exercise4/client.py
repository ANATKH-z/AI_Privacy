"""
Client implementation with ZKP support for Exercise 4.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from collections import OrderedDict
from typing import Optional, Tuple
from attack import AttackStrategy
from zkp import ZKPProver, ZKPProverRust
import os


class Client:
    """
    Represents a client in the federated learning system with ZKP support.
    Can be either benign or malicious.
    """
    def __init__(self, client_id, dataset, batch_size=32, learning_rate=0.01, 
                 is_malicious=False, attack_strategy: Optional[AttackStrategy] = None,
                 use_zkp=False, zkp_bound=30.0, zkp_norm_type='L2'):
        """
        Initialize a client.
        
        Args:
            client_id: Unique identifier for the client
            dataset: Local dataset (PyTorch Subset)
            batch_size: Batch size for local training
            learning_rate: Learning rate for SGD
            is_malicious: Whether this client is malicious
            attack_strategy: Attack strategy to use if malicious
            use_zkp: Whether to generate ZKP proofs
            zkp_bound: Bound B for ZKP (||Δ_i||_p ≤ B)
            zkp_norm_type: Type of norm for ZKP ('L1', 'L2', 'Linf')
        """
        self.client_id = client_id
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        self.model = None
        self.is_malicious = is_malicious
        self.attack_strategy = attack_strategy
        self.global_model_state = None
        
        # ZKP support
        self.use_zkp = use_zkp
        if use_zkp:
            # Check if Rust implementation should be used
            # Can be controlled via environment variable or default to simplified
            use_rust_zkp = os.getenv('USE_RUST_ZKP', 'false').lower() == 'true'
            
            if use_rust_zkp:
                try:
                    self.zkp_prover = ZKPProverRust(norm_type=zkp_norm_type, bound=zkp_bound)
                    print(f"  Client {client_id}: Using Rust ZKP implementation (Groth16)")
                except Exception as e:
                    print(f"  Client {client_id}: Failed to load Rust ZKP, falling back to simplified: {e}")
                    self.zkp_prover = ZKPProver(norm_type=zkp_norm_type, bound=zkp_bound)
            else:
                self.zkp_prover = ZKPProver(norm_type=zkp_norm_type, bound=zkp_bound)
        else:
            self.zkp_prover = None
    
    def set_model(self, model):
        """
        Set the global model for this client.
        
        Args:
            model: PyTorch model (will be copied)
        """
        self.model = deepcopy(model)
        self.global_model_state = model.state_dict()
    
    def train_local(self, num_epochs=1, round_num=0):
        """
        Perform local training for specified number of epochs.
        Following standard FL: Δw_i^{t+1} = w_i^{t+1} - w_G^t
        
        Args:
            num_epochs: Number of local training epochs
            round_num: Current communication round (for attack strategies)
        
        Returns:
            update_delta: OrderedDict representing Δw_i^{t+1} = w_i^{t+1} - w_G^t
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        w_G_t = OrderedDict()
        for key, value in self.global_model_state.items():
            w_G_t[key] = value.clone()

        # TODO (Exercise 4): Local training + benign update — same as Exercise 1 (`train_local`).
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

        benign_update = OrderedDict()
        for key in w_G_t.keys():
            benign_update[key] = w_i_t_plus_1[key] - w_G_t[key]

        if benign_update is None:
            raise NotImplementedError(
                "Implement local training and benign_update (same logic as exercise1/client.py)."
            )

        if self.is_malicious and self.attack_strategy is not None:
            malicious_update = self.attack_strategy.craft_update(
                benign_update=benign_update,
                global_model=w_G_t,
                round_num=round_num
            )
            return malicious_update
        return benign_update
    
    def generate_zkp_proof(self, update: OrderedDict) -> Tuple[dict, float]:
        """
        Generate ZKP proof for an update.
        
        Args:
            update: Model update (state dict)
        
        Returns:
            Tuple of (proof, proof_generation_time)
        """
        if not self.use_zkp or self.zkp_prover is None:
            return None, 0.0
        
        return self.zkp_prover.generate_proof(update)
    
    def get_dataset_size(self):
        """
        Get the size of the client's local dataset.
        
        Returns:
            Number of samples in the dataset
        """
        return len(self.dataset)

