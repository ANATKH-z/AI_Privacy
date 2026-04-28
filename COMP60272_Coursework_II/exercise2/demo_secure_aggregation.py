"""
Demonstration of Secure Aggregation in Federated Learning for Exercise 2.

This is an optional demo to help you understand how secure aggregation works
in the context of Federated Learning using cryptographic techniques.
Exercise 2 itself only requires a written report, not code.
"""
import torch
from collections import OrderedDict
import numpy as np


def demonstrate_secure_aggregation():
    """
    Demonstrate how secure aggregation works in Federated Learning.
    
    Shows the complete FL round with and without secure aggregation,
    including how updates are computed and aggregated.
    """
    print("=" * 70)
    print("Secure Aggregation in Federated Learning - Step by Step")
    print("=" * 70)
    
    # Simulate a simple FL scenario
    num_clients = 4
    torch.manual_seed(42)  # For reproducibility
    
    print(f"\n📋 Scenario: {num_clients} clients training a simple model")
    print("   Model has 5 parameters (simplified for demonstration)")
    
    # Step 1: Initial global model
    print("\n" + "=" * 70)
    print("STEP 1: Server broadcasts global model w_G^t to all clients")
    print("=" * 70)
    w_G_t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    print(f"   Global model w_G^t: {w_G_t.tolist()}")
    
    # Step 2: Clients perform local training
    print("\n" + "=" * 70)
    print("STEP 2: Each client performs local training")
    print("=" * 70)
    print("   Each client:")
    print("   1. Receives global model w_G^t")
    print("   2. Trains locally on their private data")
    print("   3. Gets updated model w_i^{t+1}")
    print("   4. Computes update: Δw_i^{t+1} = w_i^{t+1} - w_G^t")
    
    # Simulate local training results
    client_updated_models = []
    client_updates = []
    
    print("\n   Client local training results:")
    for i in range(num_clients):
        # Simulate local training: add some gradient-like update
        local_update = torch.randn(5) * 0.1
        updated_model = w_G_t + local_update
        client_updated_models.append(updated_model)
        client_updates.append(local_update)
        print(f"   Client {i+1}:")
        print(f"     w_{i+1}^{{t+1}} = {updated_model.tolist()}")
        print(f"     Δw_{i+1}^{{t+1}} = {local_update.tolist()}")
    
    # Step 3a: WITHOUT Secure Aggregation
    print("\n" + "=" * 70)
    print("STEP 3A: WITHOUT Secure Aggregation (Standard FedAvg)")
    print("=" * 70)
    print("   Server receives individual updates from each client:")
    for i, update in enumerate(client_updates):
        update_norm = torch.norm(update).item()
        print(f"   Client {i+1} sends: Δw_{i+1}^{{t+1}} = {update.tolist()}")
        print(f"                      ||Δw_{i+1}^{{t+1}}|| = {update_norm:.4f}")
    
    print("\n   Server can inspect each update individually:")
    print("   → Server can detect if any update is suspicious!")
    print("   → Server can identify which client sent a malicious update")
    print("   → Privacy risk: server learns about each client's data")
    
    # Standard aggregation (weighted average)
    print("\n   Server aggregates updates: Δw_agg^{t+1} = (1/N) Σ Δw_i^{t+1}")
    aggregated_update = torch.zeros_like(client_updates[0])
    for update in client_updates:
        aggregated_update += update / num_clients
    print(f"   Δw_agg^{{t+1}} = {aggregated_update.tolist()}")
    print("\n   Server updates global model: w_G^{t+1} = w_G^t + η Δw_agg^{t+1}")
    w_G_t_plus_1 = w_G_t + aggregated_update  # eta = 1.0
    print(f"   w_G^{{t+1}} = {w_G_t_plus_1.tolist()}")
    
    # Step 3b: WITH Secure Aggregation
    print("\n" + "=" * 70)
    print("STEP 3B: WITH Secure Aggregation")
    print("=" * 70)
    print("   Clients send updates through secure aggregation protocol:")
    print("   → Updates are encrypted/masked before sending")
    print("   → Server only receives the SUM of all updates")
    
    # Compute sum of updates
    sum_of_updates = sum(client_updates)
    print(f"\n   Server receives: Σ Δw_i^{{t+1}} = {sum_of_updates.tolist()}")
    print(f"   Server CANNOT see individual updates Δw_1^{{t+1}}, Δw_2^{{t+1}}, ...")
    
    # Apply aggregated update
    aggregated_update_secure = sum_of_updates / num_clients
    print(f"\n   Server computes: Δw_agg^{{t+1}} = (1/{num_clients}) Σ Δw_i^{{t+1}}")
    print(f"                  = {aggregated_update_secure.tolist()}")
    print(f"\n   Server updates: w_G^{{t+1}} = w_G^t + η Δw_agg^{{t+1}}")
    w_G_t_plus_1_secure = w_G_t + aggregated_update_secure  # eta = 1.0
    print(f"                  = {w_G_t_plus_1_secure.tolist()}")
    
    print("\n   Privacy benefit:")
    print("   ✓ Server cannot see individual client updates")
    print("   ✓ Protects information about each client's local data")
    
    # Step 4: Malicious attack scenario
    print("\n" + "=" * 70)
    print("STEP 4: Malicious Attack Scenario")
    print("=" * 70)
    
    # Create benign updates
    benign_updates = [torch.randn(5) * 0.1 for _ in range(3)]
    malicious_update = torch.ones(5) * 2.0  # Large malicious update
    
    print("   Scenario: 3 benign clients + 1 malicious client")
    print("\n   Individual updates:")
    for i, update in enumerate(benign_updates):
        print(f"   Client {i+1} (benign):  Δw_{i+1}^{{t+1}} = {update.tolist()}")
    print(f"   Client 4 (MALICIOUS): Δw_4^{{t+1}} = {malicious_update.tolist()}")
    
    # WITHOUT secure aggregation
    print("\n   WITHOUT Secure Aggregation:")
    print("   Server can inspect each update:")
    for i, update in enumerate(benign_updates):
        norm = torch.norm(update).item()
        print(f"     Client {i+1}: ||Δw_{i+1}^{{t+1}}|| = {norm:.4f} ✓ Normal")
    malicious_norm = torch.norm(malicious_update).item()
    print(f"     Client 4:  ||Δw_4^{{t+1}}|| = {malicious_norm:.4f} ⚠️  SUSPICIOUS!")
    print("   → Server can detect and reject malicious update")
    
    # WITH secure aggregation
    print("\n   WITH Secure Aggregation:")
    all_updates = benign_updates + [malicious_update]
    secure_sum = sum(all_updates)
    print(f"   Server only sees: Σ Δw_i^{{t+1}} = {secure_sum.tolist()}")
    print(f"   Server cannot see individual updates")
    print("   → Server CANNOT identify which client is malicious")
    print("   → Malicious update is hidden in the aggregate")
    
    # Equivalent aggregate example
    print("\n" + "=" * 70)
    print("STEP 5: Privacy-Robustness Trade-off Example")
    print("=" * 70)
    
    # Scenario A: All benign
    benign_sum_A = sum([torch.randn(5) * 0.1 for _ in range(4)])
    
    # Scenario B: 3 benign + 1 malicious (but malicious is smaller to compensate)
    benign_updates_B = [torch.randn(5) * 0.1 for _ in range(3)]
    # Malicious client sends smaller update to make sum similar
    malicious_compensated = benign_sum_A - sum(benign_updates_B)
    benign_sum_B = sum(benign_updates_B) + malicious_compensated
    
    print("   Two different scenarios produce the SAME aggregate:")
    print(f"   Scenario A (all benign):        Σ = {benign_sum_A.tolist()}")
    print(f"   Scenario B (3 benign + 1 malicious): Σ = {benign_sum_B.tolist()}")
    print(f"   → Server sees identical aggregates!")
    print(f"   → Server cannot distinguish between scenarios!")
    
    print("\n" + "=" * 70)
    print("SUMMARY: Privacy-Robustness Trade-off")
    print("=" * 70)
    print("\n✓ Privacy Benefits:")
    print("  • Server cannot see individual client updates")
    print("  • Protects sensitive information about client data")
    print("  • Prevents inference attacks on individual updates")
    
    print("\n✗ Robustness Costs:")
    print("  • Server cannot detect malicious updates individually")
    print("  • Malicious updates are hidden in the aggregate")
    print("  • Different update combinations can produce same aggregate")
    print("  • Makes it harder to defend against poisoning attacks")
    
    print("\n" + "=" * 70)
    print("This is why secure aggregation improves privacy")
    print("but complicates robustness in federated learning.")
    print("=" * 70)


def demonstrate_cryptographic_secure_aggregation():
    """
    Demonstrate secure aggregation using cryptographic masking technique.
    
    This implements a simplified version of the secure aggregation protocol
    using pairwise masks (similar to Bonawitz et al. 2017).
    """
    print("\n\n")
    print("=" * 70)
    print("CRYPTOGRAPHIC SECURE AGGREGATION - Detailed Implementation")
    print("=" * 70)
    
    num_clients = 4
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"\n📋 Scenario: {num_clients} clients using cryptographic secure aggregation")
    print("   Model has 5 parameters (simplified for demonstration)")
    
    # Step 1: Clients compute their updates
    print("\n" + "=" * 70)
    print("STEP 1: Clients compute their local updates")
    print("=" * 70)
    print("   Following standard FL: Δw_i^{t+1} = w_i^{t+1} - w_G^t")
    w_G_t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    client_updates = []
    
    for i in range(num_clients):
        update = torch.randn(5) * 0.1
        client_updates.append(update)
        print(f"   Client {i+1} computes: Δw_{i+1}^{{t+1}} = {update.tolist()}")
    
    # Step 2: Generate pairwise masks (cryptographic keys)
    print("\n" + "=" * 70)
    print("STEP 2: Generate pairwise masks (cryptographic keys)")
    print("=" * 70)
    print("   Each pair of clients (i, j) generates a shared random mask:")
    print("   • Client i generates: s_{i,j} (random mask)")
    print("   • Client j generates: s_{j,i} = -s_{i,j} (opposite mask)")
    print("   • When summed: s_{i,j} + s_{j,i} = 0 (masks cancel out)")
    
    # Generate pairwise masks
    pairwise_masks = {}
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            # Generate random mask
            mask = torch.randn(5) * 0.5
            pairwise_masks[(i, j)] = mask
            pairwise_masks[(j, i)] = -mask  # Opposite mask
            print(f"   Pair ({i+1}, {j+1}): s_{i+1},{j+1} = {mask.tolist()}")
            print(f"              s_{j+1},{i+1} = {(-mask).tolist()}")
    
    # Step 3: Clients mask their updates
    print("\n" + "=" * 70)
    print("STEP 3: Clients mask their updates before sending")
    print("=" * 70)
    print("   Each client i masks their update:")
    print("   • Masked update: Δw'_i^{t+1} = Δw_i^{t+1} + Σ_{j≠i} s_{i,j}")
    print("   • The mask is the sum of all pairwise masks with other clients")
    
    masked_updates = []
    for i in range(num_clients):
        # Sum all pairwise masks for client i
        total_mask = torch.zeros(5)
        for j in range(num_clients):
            if i != j:
                total_mask += pairwise_masks[(i, j)]
        
        # Mask the update
        masked_update = client_updates[i] + total_mask
        masked_updates.append(masked_update)
        
        print(f"\n   Client {i+1}:")
        print(f"     Original update:  Δw_{i+1}^{{t+1}} = {client_updates[i].tolist()}")
        print(f"     Total mask:        Σ s_{i+1},j = {total_mask.tolist()}")
        print(f"     Masked update:     Δw'_{i+1}^{{t+1}} = {masked_update.tolist()}")
        print(f"     → Sends MASKED update to server (server cannot see original)")
    
    # Step 4: Server receives only masked updates
    print("\n" + "=" * 70)
    print("STEP 4: Server receives masked updates")
    print("=" * 70)
    print("   Server receives:")
    for i, masked_update in enumerate(masked_updates):
        print(f"     Client {i+1} sends: Δw'_{i+1}^{{t+1}} = {masked_update.tolist()}")
    print("\n   ⚠️  Server CANNOT see original updates Δw_i^{t+1}!")
    print("   ⚠️  Server CANNOT identify which client sent which update!")
    
    # Step 5: Server aggregates masked updates
    print("\n" + "=" * 70)
    print("STEP 5: Server aggregates masked updates")
    print("=" * 70)
    aggregated_masked = sum(masked_updates)
    print(f"   Server computes: Σ Δw'_i^{{t+1}} = {aggregated_masked.tolist()}")
    
    # Step 6: Show that masks cancel out
    print("\n" + "=" * 70)
    print("STEP 6: Masks cancel out in aggregation")
    print("=" * 70)
    print("   When server sums all masked updates:")
    print("   Σ Δw'_i^{t+1} = Σ (Δw_i^{t+1} + Σ_{j≠i} s_{i,j})")
    print("                 = Σ Δw_i^{t+1} + Σ_{all pairs} (s_{i,j} + s_{j,i})")
    print("                 = Σ Δw_i^{t+1} + Σ_{all pairs} 0")
    print("                 = Σ Δw_i^{t+1}")
    
    # Verify masks cancel
    total_mask_sum = torch.zeros(5)
    for i in range(num_clients):
        for j in range(num_clients):
            if i != j:
                total_mask_sum += pairwise_masks[(i, j)]
    
    print(f"\n   Verification:")
    print(f"   Sum of all masks: {total_mask_sum.tolist()}")
    print(f"   → Should be approximately [0, 0, 0, 0, 0] (masks cancel)")
    
    # Step 7: Final result
    print("\n" + "=" * 70)
    print("STEP 7: Final aggregated update")
    print("=" * 70)
    true_sum = sum(client_updates)
    print(f"   True sum of updates:     Σ Δw_i^{{t+1}} = {true_sum.tolist()}")
    print(f"   Aggregated masked sum:  Σ Δw'_i^{{t+1}} = {aggregated_masked.tolist()}")
    print(f"   → They are equal! Masks successfully canceled out.")
    print(f"   → Server gets the correct aggregate WITHOUT seeing individual updates!")
    print(f"\n   Server updates global model: w_G^{{t+1}} = w_G^t + η (Σ Δw_i^{{t+1}}/N)")
    
    # Step 8: Privacy guarantee
    print("\n" + "=" * 70)
    print("STEP 8: Privacy Guarantee")
    print("=" * 70)
    print("   What server learned:")
    print("   ✓ Sum of all updates: Σ Δw_i^{t+1}")
    print("   ✗ Individual updates: Δw_1^{t+1}, Δw_2^{t+1}, Δw_3^{t+1}, Δw_4^{t+1} (HIDDEN)")
    print("   ✗ Which client sent which update (HIDDEN)")
    print("\n   Privacy achieved through cryptographic masking!")
    
    # Step 9: Malicious attack scenario
    print("\n" + "=" * 70)
    print("STEP 9: Malicious Attack with Secure Aggregation")
    print("=" * 70)
    
    # Create malicious update
    benign_updates = [torch.randn(5) * 0.1 for _ in range(3)]
    malicious_update = torch.ones(5) * 2.0
    
    print("   Scenario: 3 benign + 1 malicious client")
    print("   All clients use secure aggregation protocol")
    
    # Generate new masks for this scenario
    all_updates = benign_updates + [malicious_update]
    new_pairwise_masks = {}
    for i in range(4):
        for j in range(i + 1, 4):
            mask = torch.randn(5) * 0.5
            new_pairwise_masks[(i, j)] = mask
            new_pairwise_masks[(j, i)] = -mask
    
    # Mask all updates
    masked_all = []
    for i in range(4):
        total_mask = torch.zeros(5)
        for j in range(4):
            if i != j:
                total_mask += new_pairwise_masks[(i, j)]
        masked_all.append(all_updates[i] + total_mask)
    
    print("\n   Server receives masked updates:")
    for i, masked in enumerate(masked_all):
        print(f"     Client {i+1}: Δw'_{i+1}^{{t+1}} = {masked.tolist()}")
    
    print("\n   Server aggregates:")
    final_aggregate = sum(masked_all)
    print(f"     Σ Δw'_i^{{t+1}} = {final_aggregate.tolist()}")
    
    print("\n   ⚠️  Server CANNOT identify which client is malicious!")
    print("   ⚠️  Malicious update is hidden in the aggregate!")
    print("   → This is the robustness cost of secure aggregation")
    
    print("\n" + "=" * 70)
    print("SUMMARY: Cryptographic Secure Aggregation")
    print("=" * 70)
    print("\n🔐 Cryptographic Technique: Pairwise Masking")
    print("   • Clients generate pairwise random masks")
    print("   • Masks cancel out when aggregated")
    print("   • Server only sees masked updates")
    print("   • Server cannot learn individual updates")
    
    print("\n✓ Privacy Benefits:")
    print("   • Individual updates are cryptographically protected")
    print("   • Server cannot infer client data from updates")
    print("   • Strong privacy guarantees")
    
    print("\n✗ Robustness Costs:")
    print("   • Server cannot detect malicious updates")
    print("   • Malicious updates hidden in aggregate")
    print("   • Enables poisoning attacks")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    # demonstrate_secure_aggregation()
    demonstrate_cryptographic_secure_aggregation()


