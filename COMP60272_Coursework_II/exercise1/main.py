"""
Main script for Exercise 1: Federated Averaging on MNIST.
"""
import torch
import csv
import argparse
from model import get_model
from data_utils import load_mnist_data, distribute_data_iid, evaluate_model
from client import Client
from server import Server


def run_federated_learning(
    num_clients=10,
    num_rounds=20,
    local_epochs=1,
    batch_size=32,
    learning_rate=0.01,
    iid=True,
    seed=42,
    output_file='solution_1.csv'
):
    """
    Run Federated Averaging on MNIST.
    
    Args:
        num_clients: Number of clients (N >= 10)
        num_rounds: Number of communication rounds (R >= 20)
        local_epochs: Number of local training epochs (E >= 1)
        batch_size: Batch size for local training
        learning_rate: Learning rate for SGD
        iid: Whether to use IID data distribution
        seed: Random seed for reproducibility
        output_file: Output CSV filename
    """
    # Set random seeds for reproducibility
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
    print(f"Number of clients: {num_clients}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Local epochs: {local_epochs}")
    print(f"Data distribution: {'IID' if iid else 'Non-IID'}")
    
    # Load and distribute data
    print("\nLoading MNIST data...")
    train_dataset, test_dataset = load_mnist_data()
    
    if iid:
        client_datasets = distribute_data_iid(train_dataset, num_clients, seed=seed)
    else:
        from data_utils import distribute_data_non_iid
        client_datasets = distribute_data_non_iid(train_dataset, num_clients, seed=seed)
    
    print(f"Total training samples: {len(train_dataset)}")
    for i, dataset in enumerate(client_datasets):
        print(f"Client {i}: {len(dataset)} samples")
    
    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False
    )
    
    # Initialize global model
    print("\nInitializing global model...")
    global_model = get_model().to(device)
    
    # Create clients
    print("\nCreating clients...")
    clients = []
    for i in range(num_clients):
        client = Client(
            client_id=i,
            dataset=client_datasets[i],
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        clients.append(client)
    
    # Create server
    server = Server(global_model, clients)
    
    # Initialize clients with global model
    server.broadcast_model()
    
    # Training loop
    print("\nStarting federated training...")
    results = []
    
    for round_num in range(num_rounds):
        print(f"\nRound {round_num + 1}/{num_rounds}")
        
        # Step 1: Clients perform local training
        # Following standard FL: Δw_i^{t+1} = w_i^{t+1} - w_G^t
        client_updates = []
        for client in clients:
            # Client returns update delta: Δw_i^{t+1} = w_i^{t+1} - w_G^t
            update_delta = client.train_local(num_epochs=local_epochs)
            client_updates.append((client.client_id, update_delta))
        
        # Step 2: Server aggregates updates
        # Following standard FL: Δw_agg^{t+1} = F(Δw_i^{t+1}), w_G^{t+1} = w_G^t + η Δw_agg^{t+1}
        w_G_t_plus_1 = server.aggregate(client_updates, eta=1.0)
        
        # Step 3: Server updates global model: w_G^{t+1} = w_G^t + η Δw_agg^{t+1}
        server.update_global_model(w_G_t_plus_1)
        
        # Step 4: Broadcast updated model to clients
        server.broadcast_model()
        
        # Evaluate on test set
        accuracy = evaluate_model(server.get_model(), test_loader, device)
        print(f"Test accuracy: {accuracy:.4f}")
        
        results.append({
            'round': round_num + 1,
            'accuracy': accuracy
        })
    
    # Save results to CSV
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['round', 'accuracy'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nTraining completed! Results saved to {output_file}")
    print(f"Final accuracy: {results[-1]['accuracy']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Averaging on MNIST')
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
    parser.add_argument('--non_iid', action='store_true',
                        help='Use non-IID data distribution')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='solution_1.csv',
                        help='Output CSV filename (default: solution_1.csv)')
    
    args = parser.parse_args()
    
    run_federated_learning(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        iid=not args.non_iid,
        seed=args.seed,
        output_file=args.output
    )

