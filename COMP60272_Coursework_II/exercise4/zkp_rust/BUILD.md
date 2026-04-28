# How to Build and Install Groth16 Rust Module

## Quick Start

1. **Ensure Rust is installed**:
   ```bash
   conda activate comp60272-coursework2
   conda install -c conda-forge rust
   # Or from https://rustup.rs/
   ```

2. **Install maturin**:
   ```bash
   pip install maturin
   ```

3. **Build and install the module**:
   ```bash
   cd exercise4/zkp_rust
   maturin develop
   ```

4. **Verify installation**:
   ```bash
   python -c "import zkp_rust; print('✓ Rust module loaded successfully')"
   python -c "import zkp_rust; print('Functions:', dir(zkp_rust))"
   ```

5. **Use in Exercise 4**:
   ```bash
   cd ../..  # Go back to exercise4 directory
   export USE_RUST_ZKP=true
   python main.py
   ```

## Detailed Steps

### Step 1: Install Rust

**Using Conda (Recommended)**:
```bash
conda activate comp60272-coursework2
conda install -c conda-forge rust
```

**Or using rustup (Official method)**:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Verify installation:
```bash
rustc --version
cargo --version
```

### Step 2: Install maturin

```bash
pip install maturin
# Or
conda install -c conda-forge maturin
```

Verify installation:
```bash
maturin --version
```

### Step 3: Build Rust Module

Enter the Rust project directory:
```bash
cd exercise4/zkp_rust
```

**Development build** (Recommended, automatically installs to current environment):
```bash
maturin develop
```

**Release build** (Optimized performance):
```bash
maturin build --release
pip install target/wheels/zkp_rust-*.whl
```

**Note**: The first build needs to download arkworks dependencies, which may take some time. If you encounter network issues, check your network connection.

### Step 4: Verify Module

```bash
python -c "import zkp_rust; print('✓ Module loaded')"
python -c "import zkp_rust; print('Available functions:', [x for x in dir(zkp_rust) if not x.startswith('_')])"
```

You should see:
```
✓ Module loaded
Available functions: ['generate_proof', 'verify_proof']
```

### Step 5: Test Functionality

```python
import zkp_rust

# Test proof generation
proof = zkp_rust.generate_proof(norm=5.0, bound=30.0, norm_type="L2")
print(f"Proof generated: {len(proof)} bytes")

# Test proof verification
is_valid = zkp_rust.verify_proof(proof, bound=30.0, norm_type="L2")
print(f"Proof valid: {is_valid}")
```

### Step 6: Use in Exercise 4

```bash
cd exercise4
export USE_RUST_ZKP=true
python main.py --num_rounds 3
```

## Common Issues

### Q: `ModuleNotFoundError: No module named 'zkp_rust'`

**A**: Make sure you've run `maturin develop` or installed the wheel file.

Check:
```bash
python -c "import sys; print(sys.path)"
# Make sure your conda environment path is in the list
```

### Q: `error: failed to compile`

**A**: Check:
1. Rust is correctly installed: `rustc --version`
2. Dependencies are correct: check `Cargo.toml`
3. Network connection: first build needs to download dependencies

### Q: `package 'zkp_rust' depends on 'ark-relations' with feature 'r1cs' but 'ark-relations' does not have that feature`

**A**: This issue has been fixed. The `Cargo.toml` has removed the non-existent `r1cs` feature. If you still encounter this error:
1. Make sure `Cargo.toml` has `ark-relations = "0.4"` without `features = ["r1cs"]`
2. Clean and rebuild: `cargo clean && maturin develop`

### Q: `Rust module does not have 'generate_proof' function`

**A**: 
1. Make sure `src/lib.rs` correctly exports the functions
2. Rebuild: `maturin develop --force`
3. Check module: `python -c "import zkp_rust; print(dir(zkp_rust))"`

### Q: Build is slow

**A**: This is normal. The first build needs to:
- Download Rust dependencies (arkworks, etc.)
- Compile all dependency libraries
- Compile your code

Subsequent builds will be faster (incremental compilation).

## Uninstall

If you need to uninstall the module:

```bash
pip uninstall zkp-rust
```

## Rebuild

If you modified the Rust code:

```bash
cd exercise4/zkp_rust
maturin develop --force
```

The `--force` flag ensures a complete recompilation.
