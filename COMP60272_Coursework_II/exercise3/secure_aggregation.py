"""
Secure Aggregation implementation for Federated Learning.

This module implements the secure aggregation mechanism introduced in Exercise 2
using pairwise cryptographic masking (simplified version of Bonawitz et al. 2017 style). 
Each client masks their update with pairwise random masks that cancel out when summed,
so the server only observes the sum of updates, not individual updates.

This is used in Exercise 3 to demonstrate poisoning attacks that exploit
the privacy-robustness trade-off of secure aggregation.
"""
import torch
from collections import OrderedDict
from typing import List, Tuple, Union


def _pair_seed(round_seed: Union[int, float], i: int, j: int) -> int:
    """Deterministic seed for pair (i, j) so both clients agree on the same mask."""
    base = int(round_seed) if isinstance(round_seed, (int, float)) else hash(round_seed)
    lo, hi = min(i, j), max(i, j)
    return (base * 10000 + lo * 100 + hi) % (2**32)


def _get_pairwise_mask_sum_tensor(
    shape: Tuple[int, ...],
    num_clients: int,
    client_id: int,
    round_seed: Union[int, float],
) -> torch.Tensor:
    """
    Compute the sum of pairwise masks for one client (for a single tensor).
    For each pair (i, j), client i has s_ij and client j has s_ji = -s_ij;
    this returns the sum of s_{client_id,j} over all j != client_id.
    """
    total = torch.zeros(shape, dtype=torch.float32)
    for j in range(num_clients):
        if j == client_id:
            continue
        seed = _pair_seed(round_seed, client_id, j)
        gen = torch.Generator().manual_seed(seed)
        r = torch.randn(shape, generator=gen, dtype=torch.float32)
        if client_id < j:
            total = total + r
        else:
            total = total - r
    return total


class SecureAggregator:
    """
    Implements secure aggregation using pairwise cryptographic masking (Exercise 2).
    
    Each client masks their update with pairwise random masks s_{i,j} (with
    s_{j,i} = -s_{i,j}). The server only receives masked updates; when it sums
    them, the masks cancel and it obtains the correct aggregate without seeing
    any individual update.
    
    This enables poisoning attacks (Exercise 3) because malicious updates
    cannot be detected individually by the server.
    """
    
    def __init__(self):
        """Initialize the secure aggregator."""
        pass
    
    def mask_state_dict(
        self,
        state_dict: OrderedDict,
        client_id: int,
        num_clients: int,
        round_seed: Union[int, float] = 0,
    ) -> OrderedDict:
        """
        Mask a client's state dict with pairwise masks before sending to server.
        In this exercise, we assume that only the client would call this;  
        the server would never see the unmasked state dict.
        
        Args:
            state_dict: Client's update (or weighted update) state dict
            client_id: This client's index in [0, num_clients)
            num_clients: Total number of clients
            round_seed: Round number or seed for deterministic mask generation
        
        Returns:
            Masked state dict: state_dict + sum of pairwise masks for this client
        """
        masked = OrderedDict()
        for key in state_dict.keys():
            t = state_dict[key]
            mask_sum = _get_pairwise_mask_sum_tensor(
                t.shape, num_clients, client_id, (round_seed, key)
            )
            # Match device/dtype of original tensor
            mask_sum = mask_sum.to(device=t.device, dtype=t.dtype)
            masked[key] = t + mask_sum
        return masked
    
    def aggregate_updates(self, updates: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate updates securely. Call with MASKED updates; the sum equals
        the sum of the original (unmasked) updates because pairwise masks cancel.
        
        Args:
            updates: List of masked update tensors from clients
        
        Returns:
            Aggregated update (sum of all updates)
        """
        if not updates:
            raise ValueError("No updates provided")
        
        aggregated = torch.zeros_like(updates[0])
        for update in updates:
            aggregated += update
        return aggregated
    
    def aggregate_state_dicts(self, state_dicts: List[OrderedDict]) -> OrderedDict:
        """
        Aggregate state dicts securely. When using pairwise masking, pass
        MASKED state dicts (each client sends mask_state_dict(...)); the sum
        of masked dicts equals the sum of originals because pairwise masks cancel.
        
        Args:
            state_dicts: List of (masked) state dictionaries from clients
        
        Returns:
            Aggregated state dict (sum of all state dicts)
        """
        if not state_dicts:
            raise ValueError("No state dicts provided")
        
        aggregated_state = OrderedDict()
        first_state = state_dicts[0]
        for key in first_state.keys():
            aggregated_state[key] = torch.zeros_like(first_state[key])
        
        for state_dict in state_dicts:
            for key in aggregated_state.keys():
                aggregated_state[key] += state_dict[key]
        
        return aggregated_state
    
    def compute_update(self, new_state: OrderedDict, old_state: OrderedDict) -> OrderedDict:
        """
        Compute the update (delta) between two model states.
        
        Args:
            new_state: New model state
            old_state: Old model state
        
        Returns:
            Update delta = new_state - old_state
        """
        update = OrderedDict()
        for key in new_state.keys():
            update[key] = new_state[key] - old_state[key]
        return update
    
    def apply_update(self, base_state: OrderedDict, update: OrderedDict) -> OrderedDict:
        """
        Apply an update to a base model state.
        
        Args:
            base_state: Base model state
            update: Update to apply
        
        Returns:
            Updated state = base_state + update
        """
        updated_state = OrderedDict()
        for key in base_state.keys():
            updated_state[key] = base_state[key] + update[key]
        return updated_state

