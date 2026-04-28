"""
Main script for Exercise 3: Poisoning Attacks in Secure Federated Learning.

TODO (Exercise 3 - submission):
--------------------------------
• Set malicious_ratio ρ (typical 0.1--0.3) and justify your choice in the report.
• Ensure solution_3.csv columns match your attack type:
  - Accuracy degradation: round, accuracy_clean, accuracy_attack
  - Targeted misclassification: add target_class_accuracy (and optionally source class)
  - Backdoor: add backdoor_success_rate
• Run both clean FL (no attack) and FL with attack so accuracy_clean vs accuracy_attack
  are reported correctly.


1. Attack Category: Model Poisoning Attack.
   The malicious clients directly tamper with the gradient updates sent to the server
   (reversing and amplifying them) without modifying their local training datasets.

2. Attack Frequency: Continuous Attack.
   The malicious clients participate and send malicious updates in every communication
   round across the entire training process (all 30 rounds).

3. Attack Objective: Accuracy Degradation.
   The goal is to completely destroy the global model's availability, driving the
   overall test accuracy down to random-guessing levels (near 0%).

4. Parameter Justification (ρ = 0.2, Attack Strength = -2.0, Scale Factor = 3.0):
   I selected a malicious client fraction of ρ = 0.2 (20%). Under the cover of Secure
   Aggregation, the server cannot inspect individual updates. A 20% fraction of malicious
   clients, combined with a reversed and amplified gradient (Attack Strength = -2.0,
   Scale Factor = 3.0, effectively a -6x multiplier), is powerful enough to overwhelm the
   benign updates from the remaining 80% of clients. This specific scaling avoids immediate
   numerical overflow (NaNs) in the early rounds while ensuring the global model successfully
   diverges and the accuracy drops to 0% smoothly.
"""
import torch
import csv
import argparse
import numpy as np
from collections import OrderedDict
from model import get_model
from data_utils import load_mnist_data, distribute_data_iid, evaluate_model
from client import Client
from server import SecureServer
from attack import AccuracyDegradationAttack, TargetedMisclassificationAttack, BackdoorAttack


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


def run_secure_fl_with_attack(
    num_clients=20,
    num_rounds=20,
    local_epochs=1,
    batch_size=32,
    learning_rate=0.01,
    malicious_ratio=0.2,
    attack_type='accuracy_degradation',
    attack_strength=2.0,
    seed=42,
    output_file='solution_3.csv'
):
    """
    Run secure federated learning with poisoning attacks.
    
    IMPORTANT: This exercise implements attacks against SECURE FEDERATED LEARNING.
    The server uses secure aggregation (as introduced in Exercise 2), which means
    the server can only see the sum of all client updates, not individual updates.
    This enables poisoning attacks because malicious updates cannot be detected
    individually by the server.
    
    Args:
        num_clients: Number of clients (N >= 20)
        num_rounds: Number of communication rounds (R >= 20)
        local_epochs: Number of local training epochs (E >= 1)
        batch_size: Batch size for local training
        learning_rate: Learning rate for SGD
        malicious_ratio: Fraction of malicious clients (0 < rho < 1)
        attack_type: Type of attack ('accuracy_degradation', 'targeted', 'backdoor')
        attack_strength: Strength of the attack
        seed: Random seed for reproducibility
        output_file: Output CSV filename
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
    print("SECURE FEDERATED LEARNING WITH POISONING ATTACKS")
    print("="*70)
    print("This exercise demonstrates attacks against SECURE aggregation.")
    print("Server uses secure aggregation (Exercise 2) - cannot see individual updates.")
    print("="*70)
    print(f"\nNumber of clients: {num_clients}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Malicious ratio: {malicious_ratio}")
    print(f"Attack type: {attack_type}")
    print(f"Secure aggregation: ENABLED (server only sees sum of updates)")
    
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
    
    clients = []
    for i in range(num_clients):
        is_malicious = i in malicious_client_ids
        attack_strategy = None
        
        if is_malicious:
            if attack_type == 'accuracy_degradation':
                # AccuracyDegradationAttack reverses gradient direction
                # attack_strength should be negative to reverse: -attack_strength
                # e.g., if attack_strength=2.0, we pass -2.0 to reverse and amplify
                attack_strategy = AccuracyDegradationAttack(attack_strength=-attack_strength)
            elif attack_type == 'targeted':
                attack_strategy = TargetedMisclassificationAttack(
                    target_class=0, attack_strength=attack_strength
                )
            elif attack_type == 'backdoor':
                attack_strategy = BackdoorAttack(
                    target_label=0, attack_strength=attack_strength
                )
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")
        
        client = Client(
            client_id=i,
            dataset=client_datasets[i],
            batch_size=batch_size,
            learning_rate=learning_rate,
            is_malicious=is_malicious,
            attack_strategy=attack_strategy
        )
        clients.append(client)
        print(f"Client {i}: {'MALICIOUS' if is_malicious else 'benign'}")
    
    # Create secure server with secure aggregation enabled
    # This is the key difference from Exercise 1: server cannot see individual updates
    # This enables poisoning attacks because malicious updates are hidden in the aggregate
    server = SecureServer(global_model, clients, use_secure_aggregation=True)
    print(f"\nSecure aggregation: ENABLED")
    print("→ Server can only see the sum of all updates, not individual updates")
    print("→ This protects privacy but enables poisoning attacks")
    
    # Initialize clients with global model
    server.broadcast_model()
    
    # Training loop
    print("\nStarting secure federated training with attacks...")
    results = []
    
    for round_num in range(num_rounds):
        print(f"\nRound {round_num + 1}/{num_rounds}")
        
        # Step 1: Clients perform local training (or craft attacks)
        # Following standard FL: Δw_i^{t+1} = w_i^{t+1} - w_G^t
        client_updates = []
        w_G_t = server.model.state_dict()
        
        print("  Client update norms (||Δw_i^{t+1}||_L2):")
        for client in clients:
            # Client returns update delta: Δw_i^{t+1} = w_i^{t+1} - w_G^t
            update_delta = client.train_local(num_epochs=local_epochs, round_num=round_num)
            
            # Compute norm of update
            update_norm = compute_update_norm(update_delta, norm_type='L2')
            client_type = "MALICIOUS" if client.is_malicious else "benign"
            print(f"    Client {client.client_id:2d} ({client_type:9s}): ||Δw_{client.client_id}^{{t+1}}||_L2 = {update_norm:.6f}")
            
            client_updates.append((client.client_id, update_delta))
        
        # Step 2: Server aggregates updates using SECURE AGGREGATION
        # Following standard FL: Δw_agg^{t+1} = F(Δw_i^{t+1}), w_G^{t+1} = w_G^t + η Δw_agg^{t+1}
        # With secure aggregation, server only sees the sum of updates
        # Individual malicious updates cannot be detected
        w_G_t_plus_1 = server.aggregate(client_updates, round_num=round_num, eta=1.0)
        
        # Compute norm of aggregated update: Δw_agg^{t+1}
        # Note: This is FedAvg weighted average, so norm is typically smaller than individual updates
        # because: (1) updates from different clients may have different directions (cancellation)
        #          (2) averaging reduces magnitude: ||(1/N)ΣΔw_i|| ≤ (1/N)Σ||Δw_i||
        aggregated_update = OrderedDict()
        for key in w_G_t_plus_1.keys():
            aggregated_update[key] = w_G_t_plus_1[key] - w_G_t[key]
        aggregated_norm = compute_update_norm(aggregated_update, norm_type='L2')
        
        # Also compute average of individual update norms for comparison
        avg_individual_norm = np.mean([compute_update_norm(update, norm_type='L2') 
                                       for _, update in client_updates])
        
        # Compute malicious vs benign update norms
        malicious_norms = [compute_update_norm(update, norm_type='L2') 
                          for i, (_, update) in enumerate(client_updates) 
                          if clients[i].is_malicious]
        benign_norms = [compute_update_norm(update, norm_type='L2') 
                       for i, (_, update) in enumerate(client_updates) 
                       if not clients[i].is_malicious]
        
        avg_malicious_norm = np.mean(malicious_norms) if malicious_norms else 0.0
        avg_benign_norm = np.mean(benign_norms) if benign_norms else 0.0
        
        print(f"  Aggregated update norm: ||Δw_agg^{{t+1}}||_L2 = {aggregated_norm:.6f}")
        print(f"  Average individual norm: (1/N)Σ||Δw_i^{{t+1}}||_L2 = {avg_individual_norm:.6f}")
        print(f"  Average benign norm: {avg_benign_norm:.6f}, Average malicious norm: {avg_malicious_norm:.6f}")
        print(f"  → Aggregated norm is smaller due to FedAvg averaging (weight=1/N) and directional cancellation")
        
        # Step 3: Server updates global model: w_G^{t+1} = w_G^t + η Δw_agg^{t+1}
        server.update_global_model(w_G_t_plus_1)
        
        # Step 4: Broadcast updated model to clients
        server.broadcast_model()
        
        # Evaluate on test set
        accuracy = evaluate_model(server.get_model(), test_loader, device)
        print(f"Test accuracy: {accuracy:.4f}")
        
        results.append({
            'round': round_num + 1,
            # TODO: accuracy_clean = test accuracy without attacks; accuracy_attack = with attacks.
            # Run clean FL and attacked FL (e.g. two runs or same run with both metrics) to fill both.
            'accuracy_clean': accuracy,
            'accuracy_attack': accuracy
        })
    
    # Save results to CSV
    # TODO: Add columns per attack type (e.g. target_class_accuracy, backdoor_success_rate).
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['round', 'accuracy_clean', 'accuracy_attack'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nTraining completed! Results saved to {output_file}")
    print(f"Final accuracy: {results[-1]['accuracy_attack']:.4f}")


def run_comparison(num_clients=20, num_rounds=20, local_epochs=1, 
                   batch_size=32, learning_rate=0.01, malicious_ratio=0.2,
                   attack_type='accuracy_degradation', attack_strength=2.0,
                   seed=42, output_file='solution_3.csv'):
    """
    Run comparison between clean FL and attacked FL.
    
    This function should run two experiments:
    1. Clean FL (no attacks)
    2. FL with attacks
    
    And compare the results.
    """
    # TODO: You can implement this to compare clean vs attacked
    print("\n" + "=" * 50)
    print("1. RUNNING CLEAN FEDERATED LEARNING (NO ATTACKS)")
    print("=" * 50)
    clean_file = output_file.replace('.csv', '_clean.csv')
    run_secure_fl_with_attack(
        num_clients=num_clients, num_rounds=num_rounds, local_epochs=local_epochs,
        batch_size=batch_size, learning_rate=learning_rate,
        malicious_ratio=0.0,  # Performing clean federated learning
        attack_type=attack_type, attack_strength=attack_strength,
        seed=seed, output_file=clean_file
    )

    print("\n" + "=" * 50)
    print("2. RUNNING ATTACKED FEDERATED LEARNING")
    print("=" * 50)
    attack_file = output_file.replace('.csv', '_attack.csv')
    run_secure_fl_with_attack(
        num_clients=num_clients, num_rounds=num_rounds, local_epochs=local_epochs,
        batch_size=batch_size, learning_rate=learning_rate,
        malicious_ratio=malicious_ratio,  # Restore normal attack ratio
        attack_type=attack_type, attack_strength=attack_strength,
        seed=seed, output_file=attack_file
    )

    # Combine the results of the two operations and output them
    print(f"\nMerging results into final {output_file}...")
    import csv
    clean_results = []
    with open(clean_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_results.append(row)

    attack_results = []
    with open(attack_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            attack_results.append(row)

    final_results = []
    for i in range(len(clean_results)):
        final_results.append({
            'round': clean_results[i]['round'],
            'accuracy_clean': clean_results[i]['accuracy_clean'],
            'accuracy_attack': attack_results[i]['accuracy_attack']
        })

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['round', 'accuracy_clean', 'accuracy_attack'])
        writer.writeheader()
        writer.writerows(final_results)

    print(f"Comparison complete! Final results successfully saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Poisoning Attacks in Secure Federated Learning'
    )
    parser.add_argument('--num_clients', type=int, default=20,
                        help='Number of clients (default: 20)')
    parser.add_argument('--num_rounds', type=int, default=20,
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
                        choices=['accuracy_degradation', 'targeted', 'backdoor'],
                        help='Type of attack (default: accuracy_degradation)')
    parser.add_argument('--attack_strength', type=float, default=2.0,
                        help='Strength of the attack (default: 2.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='solution_3.csv',
                        help='Output CSV filename (default: solution_3.csv)')
    
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
        seed=args.seed,
        output_file=args.output
    )

