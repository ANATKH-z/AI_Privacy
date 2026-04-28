# COMP60272 - Security and Privacy of AI - Coursework II


This repository contains the coursework for COMP60272 Coursework II, covering Federated Learning, Secure Aggregation, Poisoning Attacks, and defenses via Zero-Knowledge Proofs, see the **coursework specification PDF**. 

This README describes the code template and how to run it.

## Overview

This coursework consists of 4 exercises:
- **Exercise 1**: Federated Averaging on MNIST (6 points) — Coding
- **Exercise 2**: Secure Aggregation — Privacy–Robustness Trade-off (4 points) — Report (written assignment)
- **Exercise 3**: Poisoning Attacks in Secure Federated Learning (7 points) — Coding
- **Exercise 4**: Defenses via Zero-Knowledge Proofs (8 points) — Coding + Report

**Total: 25 points**

Besides the assignment description, you will find this README and `TODO` comments throughout the exercises to indicate what you need to implement.
You can choose to follow these hints step by step, but you are also welcome to use your own approach --- as long as the code can run correctly and the required outputs are produced as expected (e.g., the `solution_*.csv` files).

## Installation

### Using Conda (Recommended)

All exercises use the same conda environment:

```bash
# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate comp60272-coursework2
```

### Using pip (Alternative)

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy

## Runtime and debugging

Training uses **20 clients** by default on **MNIST** with a small CNN — a typical laptop (**CPU is enough**; GPU optional) can run it. A full script (e.g. many rounds, especially Exercise 4 with ZKP) may take on the order of **roughly 10–60+ minutes** on CPU depending on your machine; ZKP adds proving/verification cost per round.

**Tip:** Use fewer rounds (e.g. `--num_rounds 5`) while debugging. For **submission**, still meet the **PDF** (e.g. Exercise 1: **R ≥ 30** rounds in your final `solution_1.csv` run).

---

# Exercise 1: Federated Averaging on MNIST

**Points**: 6  
**Type**: Programming

## Task

Implement Federated Averaging (FedAvg) for MNIST image classification.

Use at least (PDF):
- N ≥ 20 clients
- E ≥ 1 local epochs per round
- R ≥ 30 communication rounds
- Weighted averaging by dataset size; standard update rule using per-client deltas Δᵢ and global step η (eta)
- Reasonable test accuracy (PDF: typically **> 0.90** on MNIST with sufficient rounds / hyperparameters)

## Usage

Navigate to the exercise directory and run:

```bash
cd exercise1
conda activate comp60272-coursework2
python main.py
```

### Command-line Arguments

- `--num_clients`: Number of clients (default: 20, minimum: 20)
- `--num_rounds`: Number of communication rounds (default: 20; the PDF requires **R ≥ 30** for Exercise 1 submission)
- `--local_epochs`: Number of local training epochs per round (default: 1, minimum: 1)
- `--batch_size`: Batch size for local training (default: 32)
- `--learning_rate`: Learning rate for SGD (default: 0.01)
- `--non_iid`: Use non-IID data distribution (default: IID)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output`: Output CSV filename (default: solution_1.csv)

### Example

```bash
# Run with default settings (20 clients, 20 rounds, 1 local epoch)
python main.py

# Run with custom settings
python main.py --num_clients 30 --num_rounds 100 --local_epochs 2

# Run with non-IID data distribution
python main.py --non_iid
```

## Output

The script generates a CSV file (`solution_1.csv` by default) with the following format:

```csv
round,accuracy
1,0.1234
2,0.2345
...
```

## Implementation Details

### Architecture

- **Model**: Simple CNN with 2 convolutional layers and 2 fully connected layers
- **Data Distribution**: IID (equal random split) or Non-IID (sharding approach)
- **Aggregation**: Weighted averaging based on dataset sizes

### Federated Averaging Algorithm (as in the PDF)

Notation: **w** = global model parameters; **w_i** = client *i*’s local model after training; |D_i| = size of client *i*’s dataset.

1. Server broadcasts global model **w** at round *t* to all clients.
2. Each client runs local SGD for *E* epochs → local model **w_i**^(t+1).
3. Each client sends the **update** (delta) **Δ_i**^(t+1) = **w_i**^(t+1) − **w**^(t).
4. Server computes **Δ_agg**^(t+1) = Σ_i ( |D_i| / Σ_j |D_j| ) · **Δ_i**^(t+1), then **w**^(t+1) = **w**^(t) + η · **Δ_agg**^(t+1) (typically **η = 1**).
5. Repeat for *R* communication rounds.

*(see the coursework specification.)*

## File Structure

```
exercise1/
├── main.py           # Main training script
├── model.py          # Neural network model definition
├── client.py         # Client implementation
├── server.py         # Server and aggregation logic
└── data_utils.py     # Data loading and distribution utilities
```

Use the repository root `environment.yml` and `requirements.txt` for the environment.

---

# Exercise 2: Secure Aggregation - Privacy-Robustness Trade-off

**Points**: 4  
**Type**: Written Assignment (PDF report, maximum 1 page)

## Task

Explain why secure aggregation improves privacy but complicates robustness in federated learning.

## Background

Secure aggregation is a cryptographic technique that ensures the server in federated learning can only observe the **aggregate** of client updates, not individual updates. This provides strong privacy guarantees but removes the server's ability to inspect individual updates, making it harder to detect malicious behavior.

## Optional Demo Code

The `exercise2/` directory contains an optional demonstration script (`demo_secure_aggregation.py`) that illustrates the concept of secure aggregation using cryptographic techniques. This is **not required** for submission but may help you understand the topic.

### Running the Demo (Optional)

```bash
cd exercise2
conda activate comp60272-coursework2
python demo_secure_aggregation.py
```

The demo shows:
1. **Basic FL workflow**: How federated learning works with and without secure aggregation
2. **Cryptographic implementation**: How pairwise masking is used to implement secure aggregation
3. **Privacy benefits**: How individual updates are hidden from the server
4. **Robustness costs**: Why malicious updates cannot be detected individually
5. **Privacy-robustness trade-off**: The fundamental tension between privacy and security

### Demo Contents

The demo includes two parts:
- **Conceptual demonstration**: Shows the FL workflow and aggregation process
- **Cryptographic implementation**: Demonstrates pairwise masking technique (similar to Bonawitz et al. 2017)

## References

### Key Paper

**Bonawitz, K., et al. (2017). "Practical Secure Aggregation for Privacy-Preserving Machine Learning."**

- **Paper**: [https://eprint.iacr.org/2017/281.pdf](https://eprint.iacr.org/2017/281.pdf)
- **Conference**: CCS 2017
- **Abstract**: This paper presents a secure aggregation protocol for federated learning that allows a server to compute the sum of client updates without learning individual updates. The protocol uses cryptographic techniques including secret sharing and pairwise masking.

### Additional Reading

- **Bell, J. H., et al. (2020).** "Secure Single-Server Aggregation with (Poly)Logarithmic Overhead." [eprint.iacr.org/2020/704](https://eprint.iacr.org/2020/704.pdf)
- **Lycklama, H., et al. (2023).** "Rofl: Robustness of secure federated learning." IEEE S&P. [arxiv.org/pdf/2107.03311](https://arxiv.org/pdf/2107.03311)

## What to Submit

Submit as **report_2.pdf** (max 1 page), per PDF. Content should cover:
- How secure aggregation improves privacy
- Why it complicates robustness
- The trade-off between privacy and security

Your report should demonstrate understanding of:
- The cryptographic techniques used in secure aggregation
- Why individual updates cannot be inspected by the server
- How this enables poisoning attacks while protecting privacy

### File Structure

```
exercise2/
└── demo_secure_aggregation.py   # Optional demo (not required for submission)
```

---

# Exercise 3: Poisoning Attacks in Secure Federated Learning

**Points**: 7  
**Type**: Programming

## Connection to Exercise 2

**IMPORTANT**: This exercise builds directly on Exercise 2. Exercise 3 implements poisoning attacks against **secure federated learning**, where the server uses ** a simplified secure aggregation** (as introduced in Exercise 2).

The key point is:
- **Exercise 2**: Explained why secure aggregation improves privacy but complicates robustness
- **Exercise 3**: Demonstrates this trade-off by implementing attacks that exploit secure aggregation

With secure aggregation enabled, the server can only see the **sum of all client updates**, not individual updates. This protects privacy but enables poisoning attacks because malicious updates cannot be detected individually.

## Task

Design and implement a poisoning attack against **secure federated learning** (with secure aggregation enabled).

Clearly specify:
- Whether the attack is data poisoning or model poisoning
- Whether the attack is single-shot or continuous
- The attack objective (e.g., accuracy degradation or targeted misclassification)

## Usage

Navigate to the exercise directory and run:

```bash
cd exercise3
conda activate comp60272-coursework2
python main.py
```

### Command-line Arguments

- `--num_clients`: Number of clients (default: 20)
- `--num_rounds`: Number of communication rounds (default: 20)
- `--local_epochs`: Number of local training epochs per round (default: 1)
- `--batch_size`: Batch size for local training (default: 32)
- `--learning_rate`: Learning rate for SGD (default: 0.01)
- `--malicious_ratio`: Fraction of malicious clients (default: 0.2)
- `--attack_type`: Type of attack - `accuracy_degradation`, `targeted`, or `backdoor` (default: accuracy_degradation)
- `--attack_strength`: Strength of the attack (default: 2.0)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output`: Output CSV filename (default: solution_3.csv)

### Example

```bash
# Run with default settings
python main.py

# Run with accuracy degradation attack
python main.py --attack_type accuracy_degradation --attack_strength 3.0

# Run with targeted misclassification attack
python main.py --attack_type targeted --attack_strength 5.0 --malicious_ratio 0.3
```

## Output

The script generates a CSV file (`solution_3.csv` by default) with the following format:

```csv
round,accuracy_clean,accuracy_attack
1,0.1234,0.1100
2,0.2345,0.2000
...
```

For **targeted** or **backdoor** attacks, the coursework specification PDF specifies **additional columns** (`target_class_accuracy`, `backdoor_success_rate`, etc.).

**Note:** You need a **clean** run (no attack) and an **attack** run to fill `accuracy_clean` and `accuracy_attack` correctly.

## Implementation Details

### Secure Aggregation (Connection to Exercise 2)

**This exercise uses secure aggregation as introduced in Exercise 2.**

The implementation uses secure aggregation where the server only sees the sum of all client updates, not individual updates. This is similar to the secure aggregation mechanism discussed in Exercise 2:
- **Privacy benefit**: Server cannot see individual client updates
- **Robustness cost**: Server cannot detect malicious updates individually
- **Attack enabler**: Malicious updates are hidden in the aggregate, enabling poisoning attacks

This demonstrates the privacy-robustness trade-off explained in Exercise 2.

### Attack Types

1. **Accuracy Degradation Attack**: Aims to reduce overall model accuracy by reversing gradient directions
2. **Targeted Misclassification Attack**: Aims to cause specific misclassifications
3. **Backdoor Attack**: Implants a backdoor that triggers on specific patterns

### Your Tasks

You need to:

1. **Specify attack type**: Choose data poisoning or model poisoning
2. **Specify attack pattern**: Choose single-shot or continuous
3. **Specify attack objective**: Accuracy degradation, targeted misclassification, or backdoor
4. **Implement attack**: Complete the attack strategy in `attack.py`
5. **Run comparison**: Implement comparison between clean and attacked FL
6. **Generate results**: Produce `solution_3.csv` with both clean and attack accuracies

## File Structure

```
exercise3/
├── main.py                  # Main training script
├── model.py                 # Neural network model definition
├── client.py                # Client implementation (supports malicious clients)
├── server.py                # Secure server with secure aggregation
├── secure_aggregation.py   # Secure aggregation implementation
├── attack.py                # Attack strategy implementations
└── data_utils.py            # Data loading and distribution utilities
```

Use the repository root `environment.yml` and `requirements.txt` for the environment.

## Notes

- **`client.py` (`train_local`)**: Local SGD + benign update **Δ** is left as a **TODO** (same idea as Exercise 1). Implement Exercise 1 first, then reuse that logic in Ex3/Ex4 `client.py` so runs work.
- The provided attack implementations are **templates/placeholders**. You must implement your own attack strategies.
- The comparison between clean and attacked FL must be implemented by you.

---

# Exercise 4: Defenses via Zero-Knowledge Proofs

**Points**: 8  
**Type**: Programming + Report

## Connection to Exercise 3

**IMPORTANT**: This exercise extends Exercise 3 by adding Zero-Knowledge Proof (ZKP) based defense mechanisms. Exercise 4 implements ZKP-based input validation to filter out malicious updates while maintaining the privacy guarantees of secure aggregation.

The key idea:
- **Exercise 3**: Demonstrated poisoning attacks against secure aggregation
- **Exercise 4**: Implements ZKP defense to filter malicious updates without breaking privacy

## Task

Propose a ZKP-based input validation mechanism for secure FL and evaluate its overhead (see PDF for full bullet list). In summary you should:

- Use the same MNIST FL setup as Exercises 1–3; **N ≥ 20**; **R ≥ 30** for the full setting (**fewer rounds are acceptable for overhead evaluation**)
- Include benign and malicious clients (fraction **ρ** (rho); default **0.2** in template)
- Implement ZKP validation that **‖Δ_i‖_p ≤ B** (p-norm of the update bounded by **B**) without revealing the update; compare with/without ZKP; test effectiveness vs attacks (e.g. from Exercise 3)

Each client proves: there exists an update **Δ_i** such that **‖Δ_i‖_p ≤ B**.

This allows the server to enforce constraints (e.g., update norm bounds) without learning the updates themselves, thus filtering out malicious updates that exceed the bound.

## Installation

You can **first run the exercise without ZKP** (or with the template’s simplified code) to understand the workflow.

The PDF requires a **ZKP-based** mechanism that verifies **‖Δ_i‖_p ≤ B** without revealing **Δ_i**. This template’s Python “simplified” code is **not** sufficient for that requirement. For submission, use **Groth16** (Option A) or a **proper self-implemented ZKP** (Option B).

**Two implementation paths that satisfy the PDF:**

1. **Use the provided Groth16** (recommended, easiest) - Complete Groth16 implementation included
2. **Implement your own ZKP** (advanced) - Use any library or method you prefer

### Option A: Use Provided Groth16 (Recommended)

We provide a complete Groth16 zk-SNARK implementation in `exercise4/zkp_rust/` (uses arkworks in `exercise4/groth16/`). To use it:

```bash
# 1. Activate conda environment
conda activate comp60272-coursework2

# 2. Install Rust (if not already installed)
conda install -c conda-forge rust
# Or from https://rustup.rs/

# 3. Install maturin
pip install maturin

# 4. Build the Rust module
cd exercise4/zkp_rust
maturin develop

# 5. Enable Groth16 in Exercise 4
export USE_RUST_ZKP=true
cd ..
python main.py
```

**The Groth16 project is already included** - no need to download separately.

### Option B: Implement Your Own ZKP

You can implement your own ZKP solution using any library or method you prefer (see the Exercise 4 section in this document).

## Usage

Navigate to the exercise directory and run:

```bash
cd exercise4
conda activate comp60272-coursework2
python main.py
```

### Quick Start Guide

**Local exploration only (not valid for submission):** run the template without building Groth16 — useful to see the FL + defence loop; **do not** submit this as your Exercise 4 ZKP solution.

```bash
cd exercise4
conda activate comp60272-coursework2
python main.py
```

**What you must submit (Groth16):**
```bash
cd exercise4/zkp_rust
maturin develop
cd ..
export USE_RUST_ZKP=true
python main.py
```

**Note**: 
- Build Groth16 with `maturin develop` in `exercise4/zkp_rust/` and set `USE_RUST_ZKP=true` for a **submittable** Exercise 4.
- Alternatively, implement your **own** ZKP (Option B) and document it in your report and code.

### Command-line Arguments

- `--num_clients`: Number of clients (default: 20)
- `--num_rounds`: Number of communication rounds (default: 20; PDF Ex4: **R ≥ 30** in general, **fewer rounds allowed** for overhead evaluation)
- `--local_epochs`: Number of local training epochs per round (default: 1)
- `--batch_size`: Batch size for local training (default: 32)
- `--learning_rate`: Learning rate for SGD (default: 0.01)
- `--malicious_ratio`: Fraction of malicious clients (default: 0.2, as in PDF)
- `--attack_type`: `accuracy_degradation`, `targeted`, or `backdoor` (PDF default: accuracy degradation)
- `--attack_strength`: Strength of the attack (default: 2.0)
- `--zkp_bound`: ZKP bound B (||Δ_i||_p ≤ B) (default: 30.0)
- `--zkp_norm_type`: ZKP norm type - `L1`, `L2`, or `Linf` (default: L2)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output`: Output CSV filename (default: solution_4.csv)

### Example

```bash
# Run with default settings
python main.py

# Run with custom ZKP bound
python main.py --zkp_bound 50.0 --zkp_norm_type L2

# Run with different attack
python main.py --attack_type targeted --attack_strength 3.0
```

## Output

The PDF requires **`solution_4.csv`** with **exactly these five columns** per row (no extra columns in the submitted file):

```csv
round,time_no_zkp,time_zkp,comm_no_zkp,comm_zkp
1,2.31,3.95,4800000,4900000
2,2.28,3.92,4800000,4900000
...
```

### Field definitions (same as PDF)

- `round`: Communication round index  
- `time_no_zkp`: Wall-clock time (seconds) **per round** without ZKPs  
- `time_zkp`: Wall-clock time (seconds) **per round** with ZKPs  
- `comm_no_zkp`: Communication volume (bytes) **per round** without ZKPs  
- `comm_zkp`: Communication volume (bytes) **per round** with ZKPs  

If your template code writes extra columns (e.g. proof-size breakdown), **strip or merge into `comm_zkp` before submission** so the CSV matches the PDF.

## Implementation Details

### ZKP-based Input Validation

The template sketches a simplified protocol for learning the flow. **Your submitted solution** must use **Groth16** or your **own** ZKP. In general, the intended design is:
1. **Client (Prover)**: Proves ||Δ_i||_p ≤ B without revealing Δ_i
2. **Server (Verifier)**: Verifies the proof and rejects updates that fail
3. **Privacy**: The server learns only validity of the bound (together with secure aggregation), not individual raw updates

### Defense Mechanism

- **Filtering**: Updates that fail ZKP verification are rejected
- **Privacy-preserving**: Server doesn't learn individual updates (secure aggregation maintained)
- **Overhead**: ZKP generation and verification add computational and communication costs

### Your Tasks

You need to:

1. **Understand ZKP concept**: How ZKPs allow constraint verification without revealing values
2. **Implement ZKP protocol**: Complete the ZKP proof generation and verification
3. **Measure overhead**: Compare computational and communication costs with/without ZKP
4. **Analyze effectiveness**: Evaluate how well ZKP filtering defends against attacks
5. **Write report**: Explain the ZKP mechanism and analyze overhead (max 1 page)

## File Structure

```
exercise4/
├── main.py                  # Main script with overhead measurement
├── zkp.py                   # ZKP implementation (prover and verifier)
├── client.py                # Client with ZKP support
├── server.py                # Server with ZKP verification and filtering
├── model.py                 # Neural network model definition
├── data_utils.py            # Data loading utilities
├── secure_aggregation.py   # Secure aggregation (from Exercise 2/3)
├── attack.py                # Attack strategies (from Exercise 3)
├── zkp_rust/                # Groth16 ZKP (build with: maturin develop)
├── groth16/                 # arkworks library (used by zkp_rust)
├── environment.yml          # Conda environment (optional; root env works)
└── requirements.txt         # Python dependencies (optional; root works)
```

## What to Submit

Per **PDF**:

1. **report_4.pdf** (maximum **1 page**), covering at least: ZKP design (norm type, bound **B**, prove/verify); justification of parameters; effectiveness vs attack types; trade-off of **B**; computational and communication overhead.

2. **solution_4.csv** with the **five columns** above.

3. Code under **exercise4/**.



# Notes

- All exercises share the same conda environment: `comp60272-coursework2`
- Make sure to activate the conda environment before running any exercise
- Check the coursework specification for detailed requirement
