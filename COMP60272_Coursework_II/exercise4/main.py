"""
Main script for Exercise 4: ZKP-based Defense in Secure Federated Learning.

This exercise extends Exercise 3 by adding ZKP-based input validation:
clients prove ||Δ_i^(t+1)||_p ≤ B without revealing the update.

TODO (Exercise 4 - submission):
------------------------------
• Submit solution_4.csv with columns: round, time_no_zkp, time_zkp, comm_no_zkp, comm_zkp, comm_proofs (optional)
  (comm in bytes). This script writes these; ensure you run with and without ZKP
  to fill both sides.
• Submit a 1-page PDF report: ZKP design, parameter choices (e.g. bound B),
  effectiveness against attack types, trade-off of B, and overhead (time/comm).
"""
import torch
import csv
import argparse
import numpy as np
import time
import sys
from collections import OrderedDict
from model import get_model
from data_utils import load_mnist_data, distribute_data_iid, evaluate_model
from client import Client
from server import SecureServerWithZKP
from attack import AccuracyDegradationAttack, TargetedMisclassificationAttack


def compute_update_norm(update: OrderedDict, norm_type: str = 'L2') -> float:
    """
    Compute the norm of an update.
    
    Args:
        update: Model update (state dict)
        norm_type: Type of norm ('L1', 'L2', 'Linf')
    
    Returns:
        Norm value
    """
    total_norm = 0.0
    
    for key, tensor in update.items():
        if norm_type == 'L1':
            total_norm += torch.abs(tensor).sum().item()
        elif norm_type == 'L2':
            total_norm += (tensor ** 2).sum().item()
        elif norm_type == 'Linf':
            total_norm = max(total_norm, torch.abs(tensor).max().item())
    
    if norm_type == 'L2':
        total_norm = total_norm ** 0.5
    
    return total_norm


def measure_communication_size(client_updates_with_proofs, use_zkp=False):
    """
    Measure communication size for updates and optionally proofs.
    
    Args:
        client_updates_with_proofs: List of (client_id, update_delta, proof) tuples
                          where update_delta = Δw_i^{t+1} = w_i^{t+1} - w_G^t
        use_zkp: Whether ZKP proofs are included in the total
    
    Returns:
        Total communication size in bytes
    """
    total_size = 0
    
    for client_id, update_delta, proof in client_updates_with_proofs:
        # Size of model update delta (state dict)
        for key, tensor in update_delta.items():
            total_size += tensor.numel() * 4
        
        if use_zkp and proof is not None:
            import pickle
            try:
                total_size += len(pickle.dumps(proof))
            except Exception:
                total_size += 1024
    
    return total_size


def measure_proof_size(client_updates_with_proofs):
    """
    Measure total size of ZKP proofs only (in bytes).
    
    Args:
        client_updates_with_proofs: List of (client_id, update_delta, proof) tuples
    
    Returns:
        Total proof size in bytes (0 if no proofs)
    """
    import pickle
    total = 0
    for _client_id, _update_delta, proof in client_updates_with_proofs:
        if proof is not None:
            try:
                # Prefer raw proof_data size if present (what is actually sent on wire)
                proof_data = proof.get('proof_data') if isinstance(proof, dict) else None
                if proof_data is not None:
                    total += len(proof_data) if isinstance(proof_data, (bytes, bytearray)) else len(pickle.dumps(proof_data))
                else:
                    total += len(pickle.dumps(proof))
            except Exception:
                total += 1024
    return total


def run_round_with_zkp(
    server, clients, test_loader, device, round_num, local_epochs,
    use_zkp=False, zkp_bound=30.0
):
    """
    Run a single communication round with or without ZKP.
    
    Returns:
        Tuple of (accuracy, round_time, comm_size, comm_updates_only, comm_proofs)
        where comm_updates_only is the size of updates without proofs (baseline),
        and comm_proofs is the size of ZKP proofs only.
    """
    round_start = time.time()
    
    # Step 1: Clients perform local training (or craft attacks)
    client_updates_with_proofs = []
    total_proof_time = 0.0
    
    for client in clients:
        # Train locally following standard FL: Δw_i^{t+1} = w_i^{t+1} - w_G^t
        # Client returns update delta: Δw_i^{t+1} = w_i^{t+1} - w_G^t
        update_delta = client.train_local(num_epochs=local_epochs, round_num=round_num)
        
        # Generate ZKP proof for the update delta if enabled
        # ZKP proves: ||Δw_i^{t+1}||_p ≤ B where Δw_i^{t+1} = w_i^{t+1} - w_G^t
        proof = None
        proof_time = 0.0
        if use_zkp:
            # TODO (Exercise 4): Generate ZKP proof for this client's update_delta.
            # The proof should attest that ||Δw_i^{t+1}||_p ≤ B without revealing the update.
            # Use generate_zkp_proof() in client.py if the client has a prover configured,
            # or implement your own proof generation. Return (proof, proof_time) with proof_time in seconds.
            proof, proof_time = client.generate_zkp_proof(update_delta)
            total_proof_time += proof_time
        
        client_updates_with_proofs.append((client.client_id, update_delta, proof))
    
    # Debug: Print update norms for first round or when attacks are present
    if round_num == 0 or any(client.is_malicious for client in clients):
        malicious_norms = []
        benign_norms = []
        for client_id, update_delta, _ in client_updates_with_proofs:
            norm = compute_update_norm(update_delta, norm_type='L2')
            if clients[client_id].is_malicious:
                malicious_norms.append(norm)
            else:
                benign_norms.append(norm)
        
        if malicious_norms:
            avg_malicious = np.mean(malicious_norms)
            avg_benign = np.mean(benign_norms) if benign_norms else 0.0
            if round_num == 0:
                print(f"  Round {round_num + 1} update norms:")
            print(f"    Average benign norm: {avg_benign:.6f}")
            print(f"    Average malicious norm: {avg_malicious:.6f}")
            if avg_benign > 0:
                ratio = avg_malicious / avg_benign
                print(f"    Malicious/Benign ratio: {ratio:.2f}x")
            else:
                print(f"    Malicious/Benign ratio: inf")
    
    # Step 2: Server aggregates updates using SECURE AGGREGATION with ZKP filtering
    # Following standard FL: Δw_agg^{t+1} = F(Δw_i^{t+1}), w_G^{t+1} = w_G^t + η Δw_agg^{t+1}
    w_G_t_plus_1 = server.aggregate(client_updates_with_proofs, round_num=round_num, eta=1.0)
    
    # Step 3: Server updates global model: w_G^{t+1} = w_G^t + η Δw_agg^{t+1}
    server.update_global_model(w_G_t_plus_1)
    
    # Step 4: Broadcast updated model to clients
    server.broadcast_model()
    
    # Evaluate on test set
    accuracy = evaluate_model(server.get_model(), test_loader, device)
    
    round_time = time.time() - round_start
    
    # Measure communication size (with proof if use_zkp, else updates only)
    comm_size = measure_communication_size(client_updates_with_proofs, use_zkp=use_zkp)
    comm_updates_only = measure_communication_size(client_updates_with_proofs, use_zkp=False)
    comm_proofs = measure_proof_size(client_updates_with_proofs)
    
    return accuracy, round_time, comm_size, comm_updates_only, comm_proofs


def run_comparison(
    num_clients=20,
    num_rounds=2,
    local_epochs=1,
    batch_size=32,
    learning_rate=0.01,
    malicious_ratio=0.2,
    attack_type='accuracy_degradation',
    attack_strength=2.0,
    zkp_bound=30.0,
    zkp_norm_type='L2',
    use_zkp_only=False,
    seed=42,
    output_file='solution_4.csv'
):
    """
    Run comparison between FL without ZKP and with ZKP.
    
    When use_zkp_only=True, runs only the ZKP experiment (skips no-ZKP run).
    Measures computational and communication overhead of ZKP.
    """
    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    print(f"\n{'='*70}")
    print("EXERCISE 4: ZKP-based Defense in Secure Federated Learning")
    print("="*70)
    print(f"Number of clients: {num_clients}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Malicious ratio: {malicious_ratio}")
    print(f"Attack type: {attack_type}")
    print(f"ZKP bound: {zkp_bound} (||Δ_i||_{zkp_norm_type} ≤ {zkp_bound})")
    print("="*70)
    
    # Load and distribute data
    print("\nLoading MNIST data...")
    train_dataset, test_dataset = load_mnist_data()
    client_datasets = distribute_data_iid(train_dataset, num_clients, seed=seed)
    
    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False
    )
    
    # Initialize global model
    print("\nInitializing global model...")
    global_model = get_model().to(device)
    
    # Create clients (benign and malicious)
    print("\nCreating clients...")
    num_malicious = int(num_clients * malicious_ratio)
    malicious_client_ids = np.random.choice(num_clients, num_malicious, replace=False)
    print(f"Malicious clients: {sorted(malicious_client_ids)}")
    print(f"Attack type: {attack_type}, Attack strength: {attack_strength}")
    
    results = []
    
    if not use_zkp_only:
        # Run without ZKP
        print("\n" + "="*70)
        print("Running WITHOUT ZKP (baseline - attacks enabled, no defense)")
        print("="*70)
        
        model_no_zkp = get_model().to(device)
        clients_no_zkp = []
        
        for i in range(num_clients):
            is_malicious = i in malicious_client_ids
            attack_strategy = None
            
            if is_malicious:
                if attack_type == 'accuracy_degradation':
                    attack_strategy = AccuracyDegradationAttack(attack_strength=-attack_strength)
                elif attack_type == 'targeted':
                    attack_strategy = TargetedMisclassificationAttack(
                        target_class=0, attack_strength=attack_strength
                    )
                print(f"  Client {i}: MALICIOUS ({attack_type}, strength={attack_strength})")
            else:
                print(f"  Client {i}: benign")
            
            client = Client(
                client_id=i,
                dataset=client_datasets[i],
                batch_size=batch_size,
                learning_rate=learning_rate,
                is_malicious=is_malicious,
                attack_strategy=attack_strategy,
                use_zkp=False
            )
            clients_no_zkp.append(client)
        
        server_no_zkp = SecureServerWithZKP(
            model_no_zkp, clients_no_zkp,
            use_secure_aggregation=True,
            use_zkp=False
        )
        server_no_zkp.broadcast_model()
        
        for round_num in range(num_rounds):
            accuracy, round_time, comm_size, comm_updates_only, comm_proofs = run_round_with_zkp(
                server_no_zkp, clients_no_zkp, test_loader, device,
                round_num, local_epochs, use_zkp=False
            )
            
            results.append({
                'round': round_num + 1,
                'time_no_zkp': round_time,
                'time_zkp': 0.0,
                'comm_no_zkp': comm_size,
                'comm_zkp': 0,
                'comm_proofs': 0
            })
            
            print(f"Round {round_num + 1}/{num_rounds}: accuracy={accuracy:.4f}, "
                  f"time={round_time:.4f}s, comm={comm_size/1e6:.2f}MB")
    else:
        # Pre-initialize results when only running ZKP
        results = [
            {'round': r + 1, 'time_no_zkp': 0.0, 'time_zkp': 0.0, 'comm_no_zkp': 0, 'comm_zkp': 0, 'comm_proofs': 0}
            for r in range(num_rounds)
        ]
    
    # Run with ZKP
    print("\n" + "="*70)
    print("Running WITH ZKP (defense enabled - attacks filtered)")
    print("="*70)
    
    # Reinitialize for ZKP run
    model_zkp = get_model().to(device)
    clients_zkp = []
    
    for i in range(num_clients):
        is_malicious = i in malicious_client_ids
        attack_strategy = None
        
        if is_malicious:
            if attack_type == 'accuracy_degradation':
                # AccuracyDegradationAttack reverses gradient direction
                attack_strategy = AccuracyDegradationAttack(attack_strength=-attack_strength)
            elif attack_type == 'targeted':
                attack_strategy = TargetedMisclassificationAttack(
                    target_class=0, attack_strength=attack_strength
                )
        
        client = Client(
            client_id=i,
            dataset=client_datasets[i],
            batch_size=batch_size,
            learning_rate=learning_rate,
            is_malicious=is_malicious,
            attack_strategy=attack_strategy,
            use_zkp=True,
            zkp_bound=zkp_bound,
            zkp_norm_type=zkp_norm_type
        )
        clients_zkp.append(client)
    
    server_zkp = SecureServerWithZKP(
        model_zkp, clients_zkp,
        use_secure_aggregation=True,
        use_zkp=True,
        zkp_bound=zkp_bound,
        zkp_norm_type=zkp_norm_type
    )
    server_zkp.broadcast_model()
    
    for round_num in range(num_rounds):
        accuracy, round_time, comm_size, comm_updates_only, comm_proofs = run_round_with_zkp(
            server_zkp, clients_zkp, test_loader, device,
            round_num, local_epochs, use_zkp=True, zkp_bound=zkp_bound
        )
        
        results[round_num]['time_zkp'] = round_time
        results[round_num]['comm_zkp'] = comm_size
        results[round_num]['comm_proofs'] = comm_proofs
        if use_zkp_only:
            results[round_num]['comm_no_zkp'] = comm_updates_only
        
        proof_str = f"{comm_proofs/1e3:.2f}KB" if comm_proofs < 1e6 else f"{comm_proofs/1e6:.2f}MB"
        print(f"Round {round_num + 1}/{num_rounds}: accuracy={accuracy:.4f}, "
              f"time={round_time:.4f}s, comm={comm_size/1e6:.2f}MB (proofs: {proof_str})")
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['round', 'time_no_zkp', 'time_zkp',
                                                'comm_no_zkp', 'comm_zkp', 'comm_proofs'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nTraining completed! Results saved to {output_file}")
    print(f"\nOverhead Summary:")
    avg_time_no_zkp = np.mean([r['time_no_zkp'] for r in results])
    avg_time_zkp = np.mean([r['time_zkp'] for r in results])
    avg_comm_no_zkp = np.mean([r['comm_no_zkp'] for r in results])
    avg_comm_zkp = np.mean([r['comm_zkp'] for r in results])
    avg_comm_proofs = np.mean([r['comm_proofs'] for r in results])
    
    print(f"  Average time (no ZKP): {avg_time_no_zkp:.4f}s")
    print(f"  Average time (with ZKP): {avg_time_zkp:.4f}s")
    if avg_time_no_zkp > 0:
        print(f"  Time overhead: {(avg_time_zkp/avg_time_no_zkp - 1)*100:.2f}%")
    else:
        print(f"  Time overhead: N/A (no baseline run)")
    print(f"  Average comm (no ZKP, updates only): {avg_comm_no_zkp/1e6:.2f}MB")
    if avg_comm_proofs >= 1e6:
        print(f"  Average comm (proofs only): {avg_comm_proofs/1e6:.2f}MB")
    elif avg_comm_proofs > 0:
        print(f"  Average comm (proofs only): {avg_comm_proofs/1e3:.2f}KB")
    else:
        print(f"  Average comm (proofs only): 0.00MB (no proofs generated – check ZKP prover is enabled)")
    print(f"  Average comm (with ZKP, total): {avg_comm_zkp/1e6:.2f}MB")
    if avg_comm_no_zkp > 0:
        comm_overhead_pct = (avg_comm_zkp / avg_comm_no_zkp - 1) * 100
        # Use more precision when overhead is tiny (e.g. proof adds a few KB)
        if 0 < comm_overhead_pct < 0.01:
            print(f"  Comm overhead: {comm_overhead_pct:.4f}% (proofs add {avg_comm_proofs/1e3:.2f} KB)")
        elif comm_overhead_pct >= 0.01:
            print(f"  Comm overhead: {comm_overhead_pct:.2f}%")
        else:
            print(f"  Comm overhead: {comm_overhead_pct:.2f}%")
    else:
        print(f"  Comm overhead: N/A (no baseline)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ZKP-based Defense in Secure Federated Learning'
    )
    parser.add_argument('--num_clients', type=int, default=20,
                        help='Number of clients (default: 20)')
    parser.add_argument('--num_rounds', type=int, default=2,
                        help='Number of communication rounds (default: 20)')
    parser.add_argument('--local_epochs', type=int, default=1,
                        help='Number of local training epochs (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for local training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--malicious_ratio', type=float, default=0.2,
                        help='Fraction of malicious clients (default: 0.2)')
    parser.add_argument('--attack_type', type=str, default='accuracy_degradation',
                        choices=['accuracy_degradation', 'targeted'],
                        help='Type of attack (default: accuracy_degradation)')
    parser.add_argument('--attack_strength', type=float, default=2.0,
                        help='Strength of the attack (default: 2.0)')
    parser.add_argument('--zkp_bound', type=float, default=30.0,
                        help='ZKP bound B (||Δ_i||_p ≤ B) (default: 10.0)')
    parser.add_argument('--zkp_norm_type', type=str, default='L2',
                        choices=['L1', 'L2', 'Linf'],
                        help='ZKP norm type (default: L2)')
    parser.add_argument('--use_zkp', action='store_true',
                        help='Run only with ZKP defense (default: run both with and without ZKP)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='solution_4.csv',
                        help='Output CSV filename (default: solution_4.csv)')
    
    args = parser.parse_args()
    
    run_comparison(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        malicious_ratio=args.malicious_ratio,
        attack_type=args.attack_type,
        attack_strength=args.attack_strength,
        zkp_bound=args.zkp_bound,
        zkp_norm_type=args.zkp_norm_type,
        use_zkp_only=args.use_zkp,
        seed=args.seed,
        output_file=args.output
    )

