"""
Server implementation with ZKP-based defense for Exercise 4.
"""
import torch
import time
from collections import OrderedDict
from secure_aggregation import SecureAggregator
from zkp import ZKPVerifier, ZKPVerifierRust
import os


class SecureServerWithZKP:
    """
    Server that coordinates federated learning with secure aggregation and ZKP defense.
    
    This extends Exercise 3's secure server by adding ZKP-based input validation
    to filter out malicious updates while maintaining privacy.
    """
    def __init__(self, model, clients, use_secure_aggregation=True, 
                 use_zkp=False, zkp_bound=30.0, zkp_norm_type='L2'):
        """
        Initialize the server.
        
        Args:
            model: Global model (PyTorch model)
            clients: List of Client objects
            use_secure_aggregation: Whether to use secure aggregation
            use_zkp: Whether to use ZKP for input validation
            zkp_bound: Bound B for ZKP (||Δ_i||_p ≤ B)
            zkp_norm_type: Type of norm for ZKP ('L1', 'L2', 'Linf')
        """
        self.model = model
        self.clients = clients
        self.use_secure_aggregation = use_secure_aggregation
        self.use_zkp = use_zkp
        self.total_samples = sum(client.get_dataset_size() for client in clients)
        
        if use_secure_aggregation:
            self.secure_aggregator = SecureAggregator()
        
        if use_zkp:
            # Check if Rust implementation should be used
            # Can be controlled via environment variable or default to simplified
            use_rust_zkp = os.getenv('USE_RUST_ZKP', 'false').lower() == 'true'
            
            if use_rust_zkp:
                try:
                    self.zkp_verifier = ZKPVerifierRust(norm_type=zkp_norm_type, bound=zkp_bound)
                    print("  Server: Using Rust ZKP implementation (Groth16)")
                except Exception as e:
                    print(f"  Server: Failed to load Rust ZKP, falling back to simplified: {e}")
                    self.zkp_verifier = ZKPVerifier(norm_type=zkp_norm_type, bound=zkp_bound)
            else:
                self.zkp_verifier = ZKPVerifier(norm_type=zkp_norm_type, bound=zkp_bound)
        else:
            self.zkp_verifier = None
    
    def aggregate(self, client_updates_with_proofs, round_num=0, eta=1.0):
        """
        Aggregate client updates with optional ZKP filtering.
        Flow: (1) If ZKP enabled, call zkp_verifier.filter_updates(...) to get valid (client_id, update_delta) list.
              (2) Then aggregate over that list: either secure aggregation (mask_state_dict + aggregate_state_dicts)
              or standard FedAvg as backup. See TODOs in the method body for what you need to implement.
        Following standard FL: w_G^{t+1} = w_G^t + η Δw_agg^{t+1}.
        
        Args:
            client_updates_with_proofs: List of (client_id, update_delta, proof) tuples
                          where update_delta = Δw_i^{t+1} = w_i^{t+1} - w_G^t
            round_num: Current communication round (for logging)
            eta: Learning rate/step size for global model update (default: 1.0)
        
        Returns:
            Updated global model state dict: w_G^{t+1} = w_G^t + η Δw_agg^{t+1}
        """
        # Get current global model: w_G^t
        w_G_t = self.model.state_dict()
        
        # Filter updates using ZKP if enabled
        if self.use_zkp and self.zkp_verifier is not None:
            # TODO (Exercise 4): Filter out updates that fail ZKP verification.
            # Use zkp_verifier.filter_updates() to get only
            # (client_id, update_delta) pairs whose proof verifies ||Δ_i||_p ≤ B. Replace the line below.
            valid_updates = self.zkp_verifier.filter_updates(client_updates_with_proofs)
            client_updates = valid_updates
            
            # Handle case where all updates are rejected
            if len(client_updates) == 0:
                print(f"  ⚠️  WARNING: All updates rejected by ZKP in round {round_num}")
                print(f"  → Using previous model state (no aggregation)")
                # Return current model state unchanged
                return w_G_t
        else:
            # No ZKP filtering, use all updates
            client_updates = [(cid, update_delta) for cid, update_delta, _ in client_updates_with_proofs]
        
        # Handle empty updates list
        if len(client_updates) == 0:
            return w_G_t
        
        if self.use_secure_aggregation:
            # TODO (Exercise 4): Implement secure aggregation (same idea as Exercise 3; ex4 may have fewer clients after ZKP filter).
            # Use self.secure_aggregator. After ZKP filtering, client_updates may be a subset; use the *index* in this list for masking so masks cancel.
            #
            # Step 1: num_clients = len(client_updates). Compute weight per client: client.get_dataset_size() / self.total_samples.
            #
            # Step 2: For each client in client_updates, build weighted update and mask it.
            #   - Use enumerate(client_updates) so you have (idx, (client_id, update_delta)). Use idx (not client_id) for masking so that when ZKP drops some clients, masks still cancel.
            #   - weighted_update[key] = weight * update_delta[key] for each key.
            #   - Call: self.secure_aggregator.mask_state_dict(weighted_update, idx, num_clients, round_num)
            #   - Append result to masked_updates.
            #
            # Step 3: Call: aggregated_update = self.secure_aggregator.aggregate_state_dicts(masked_updates)
            #
            # Step 4: w_G^{t+1} = w_G^t + eta * aggregated_update, then return w_G_t_plus_1.
            #
            if self.use_secure_aggregation:
                # TODO (Exercise 4): Implement secure aggregation (same idea as Exercise 3; ex4 may have fewer clients after ZKP filter).
                num_clients = len(client_updates)
                masked_updates = []

                for idx, (client_id, update_delta) in enumerate(client_updates):
                    client_obj = next(c for c in self.clients if c.client_id == client_id)
                    weight = client_obj.get_dataset_size() / self.total_samples

                    weighted_update = OrderedDict()
                    for key in update_delta.keys():
                        weighted_update[key] = weight * update_delta[key]

                    # 必须使用 idx 掩码，保证被过滤后剩下的客户端索引连续，使得掩码依然能两两抵消
                    masked_update = self.secure_aggregator.mask_state_dict(
                        weighted_update, idx, num_clients, round_num
                    )
                    masked_updates.append(masked_update)

                aggregated_update = self.secure_aggregator.aggregate_state_dicts(masked_updates)

                w_G_t_plus_1 = OrderedDict()
                for key in w_G_t.keys():
                    w_G_t_plus_1[key] = w_G_t[key] + eta * aggregated_update[key]

                return w_G_t_plus_1
        else:
            # TODO (Exercise 4): Standard weighted aggregation (FedAvg) when secure aggregation is disabled.
            # Same as Exercise 3: Δw_agg^{t+1} = Σ_i (|D_i|/|D|) Δw_i^{t+1}, then w_G^{t+1} = w_G^t + η Δw_agg^{t+1}.
            # Step 1: Compute weight per client (get_dataset_size() / self.total_samples).
            # Step 2: aggregated_update = weighted sum of update_delta; then w_G_t_plus_1 = w_G_t + eta * aggregated_update. Return w_G_t_plus_1.
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

