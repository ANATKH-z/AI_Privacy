"""
ZKP for input validation in secure FL: clients prove ||Δ_i||_p ≤ B without revealing Δ_i.

REFERENCE: Groth16 in zkp_rust — build with `cd zkp_rust && maturin develop`.
  Used by default when the module is installed (set USE_RUST_ZKP=false to force fallback).
  You may use this or implement another ZKP (e.g. Bulletproofs, PLONK).

FALLBACK: ZKPProver / ZKPVerifier below are simplified (no real zero-knowledge);
  used when zkp_rust is not available.
"""
import torch
import hashlib
import time
from collections import OrderedDict
from typing import Tuple, Optional


class ZKPProver:
    """Prover: proves ||Δ_i||_p ≤ B without revealing Δ_i (simplified implementation)."""

    def __init__(self, norm_type: str = 'L2', bound: float = 30.0):
        self.norm_type = norm_type
        self.bound = bound
    
    def compute_norm(self, update: OrderedDict) -> float:
        total_norm = 0.0
        
        for key, tensor in update.items():
            if self.norm_type == 'L1':
                total_norm += torch.abs(tensor).sum().item()
            elif self.norm_type == 'L2':
                total_norm += (tensor ** 2).sum().item()
            elif self.norm_type == 'Linf':
                total_norm = max(total_norm, torch.abs(tensor).max().item())
        
        if self.norm_type == 'L2':
            total_norm = total_norm ** 0.5
        
        return total_norm
    
    def generate_proof(self, update: OrderedDict) -> Tuple[dict, float]:
        """Generate a (simplified) proof that ||Δ_i||_p ≤ B. Returns (proof, time)."""
        start_time = time.time()
        norm = self.compute_norm(update)
        norm_commitment = self._commit_norm(norm)
        range_proof = self._generate_range_proof(norm, self.bound)
        proof = {
            'norm_commitment': norm_commitment,
            'bound': self.bound,
            'norm_type': self.norm_type,
            'range_proof': range_proof
        }
        
        proof_time = time.time() - start_time
        
        return proof, proof_time
    
    def _commit_norm(self, norm: float) -> bytes:
        """Simplified commitment (hash); real ZKP would use Pedersen or similar."""
        norm_str = f"{norm:.10f}"
        return hashlib.sha256(norm_str.encode()).digest()
    
    def _generate_range_proof(self, norm: float, bound: float) -> dict:
        """Simplified: only records norm ≤ bound (no cryptographic proof)."""
        return {'proof_type': 'simplified_range_proof', 'satisfies_bound': norm <= bound}
    
    def get_proof_size(self, proof: dict) -> int:
        """Estimate proof size in bytes."""
        import pickle
        try:
            return len(pickle.dumps(proof))
        except:
            # Fallback estimation
            size = 0
            size += len(proof.get('norm_commitment', b''))
            size += len(str(proof.get('bound', 0)).encode())
            size += len(str(proof.get('norm_type', '')).encode())
            size += len(str(proof.get('range_proof', {})).encode())
            return size


class ZKPVerifier:
    """Verifier: checks proofs that ||Δ_i||_p ≤ B (simplified implementation)."""

    def __init__(self, norm_type: str = 'L2', bound: float = 30.0):
        self.norm_type = norm_type
        self.bound = bound

    def verify_proof(self, proof: dict) -> Tuple[bool, float]:
        start_time = time.time()
        
        # Verify proof structure
        if 'norm_commitment' not in proof:
            return False, time.time() - start_time
        
        if 'range_proof' not in proof:
            return False, time.time() - start_time
        
        # Verify bound matches
        if proof.get('bound') != self.bound:
            return False, time.time() - start_time
        
        if proof.get('norm_type') != self.norm_type:
            return False, time.time() - start_time
        
        # Verify range proof
        range_proof = proof.get('range_proof', {})
        is_valid = range_proof.get('satisfies_bound', False)
        
        verification_time = time.time() - start_time
        
        return is_valid, verification_time

    def filter_updates(self, client_updates_with_proofs: list) -> list:
        """Return only (client_id, update_delta) for updates that pass verification."""
        valid_updates = []
        for client_id, update_delta, proof in client_updates_with_proofs:
            is_valid, _ = self.verify_proof(proof)
            if is_valid:
                valid_updates.append((client_id, update_delta))
            else:
                print(f"  ⚠️  Client {client_id} update rejected: ||Δw_{client_id}^{{t+1}}|| exceeds bound")
        return valid_updates


# -------- Groth16 (zkp_rust): reference rigorous implementation --------

def _load_zkp_rust():
    """Load zkp_rust module (PyO3) or CDLL; return (lib, available, use_ctypes)."""
    import os
    import platform
    _here = os.path.dirname(os.path.abspath(__file__))
    try:
        if platform.system() == "Windows":
            ext = ".dll"
        elif platform.system() == "Darwin":
            ext = ".dylib"
        else:
            ext = ".so"
        lib_path = os.path.join(_here, "zkp_rust", "target", "release", f"libzkp_rust{ext}")
        if os.path.exists(lib_path):
            import ctypes
            return ctypes.CDLL(lib_path), True, True
    except Exception:
        pass
    for name in ['zkp_rust', 'zkp_groth16', 'fl_zkp']:
        try:
            m = __import__(name)
            if hasattr(m, 'generate_proof') and hasattr(m, 'verify_proof'):
                print(f"✓ Loaded Rust ZKP module: {name}")
                return m, True, False
        except ImportError:
            continue
    if os.getenv('USE_RUST_ZKP', 'false').lower() == 'true':
        print("WARNING: Rust ZKP not found. Build with: cd zkp_rust && maturin develop")
    return None, False, False


class ZKPProverRust:
    """Groth16 prover via zkp_rust (build: cd zkp_rust && maturin develop). Falls back to ZKPProver if unavailable."""

    def __init__(self, norm_type: str = 'L2', bound: float = 30.0):
        self.norm_type = norm_type
        self.bound = bound
        self.rust_lib, self.rust_available, self.use_ctypes = _load_zkp_rust()
    
    def compute_norm(self, update: OrderedDict) -> float:
        """Same as ZKPProver.compute_norm; norm is passed to Rust."""
        total_norm = 0.0
        for key, tensor in update.items():
            if self.norm_type == 'L1':
                total_norm += torch.abs(tensor).sum().item()
            elif self.norm_type == 'L2':
                total_norm += (tensor ** 2).sum().item()
            elif self.norm_type == 'Linf':
                total_norm = max(total_norm, torch.abs(tensor).max().item())
        
        if self.norm_type == 'L2':
            total_norm = total_norm ** 0.5
        
        return total_norm
    
    def _call_rust_prover(self, update: OrderedDict):
        import ctypes
        if not self.rust_available:
            raise ImportError("Rust library is not available")
        norm = self.compute_norm(update)
        if self.use_ctypes:
            self.rust_lib.generate_proof.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_char_p]
            self.rust_lib.generate_proof.restype = ctypes.POINTER(ctypes.c_char)
            proof_ptr = self.rust_lib.generate_proof(
                ctypes.c_double(norm), ctypes.c_double(self.bound), ctypes.c_char_p(self.norm_type.encode())
            )
            return ctypes.string_at(proof_ptr)
        proof_data = self.rust_lib.generate_proof(norm, self.bound, self.norm_type)
        return bytes(proof_data) if isinstance(proof_data, list) else proof_data
    
    def generate_proof(self, update: OrderedDict) -> Tuple[dict, float]:
        """Generate Groth16 proof; fall back to ZKPProver if Rust unavailable."""
        start_time = time.time()
        if not self.rust_available:
            if __import__('os').getenv('USE_RUST_ZKP', 'false').lower() == 'true':
                print("  ⚠️  Using simplified ZKP (Rust not available)")
            return ZKPProver(norm_type=self.norm_type, bound=self.bound).generate_proof(update)
        try:
            proof_data = self._call_rust_prover(update)
            return {
                'proof_type': 'rust_arkworks_zk_snark',
                'proof_data': proof_data,
                'bound': self.bound,
                'norm_type': self.norm_type,
            }, time.time() - start_time
        except Exception as e:
            if __import__('os').getenv('USE_RUST_ZKP', 'false').lower() == 'true':
                print(f"  ⚠️  Rust ZKP failed ({e}), using simplified")
            return ZKPProver(norm_type=self.norm_type, bound=self.bound).generate_proof(update)


class ZKPVerifierRust:
    """Groth16 verifier via zkp_rust. Falls back to ZKPVerifier if unavailable."""

    def __init__(self, norm_type: str = 'L2', bound: float = 30.0):
        self.norm_type = norm_type
        self.bound = bound
        self.rust_lib, self.rust_available, self.use_ctypes = _load_zkp_rust()
    
    def verify_proof(self, proof: dict) -> Tuple[bool, float]:
        import ctypes
        start_time = time.time()
        if not self.rust_available:
            if __import__('os').getenv('USE_RUST_ZKP', 'false').lower() == 'true':
                print("  ⚠️  Using simplified ZKP verification (Rust not available)")
            return ZKPVerifier(norm_type=self.norm_type, bound=self.bound).verify_proof(proof)
        if proof.get('proof_type') != 'rust_arkworks_zk_snark' or proof.get('proof_data') is None:
            return False, time.time() - start_time
        if proof.get('bound') != self.bound or proof.get('norm_type') != self.norm_type:
            return False, time.time() - start_time
        proof_data = proof['proof_data']
        try:
            if self.use_ctypes:
                self.rust_lib.verify_proof.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_double, ctypes.c_char_p]
                self.rust_lib.verify_proof.restype = ctypes.c_bool
                is_valid = self.rust_lib.verify_proof(
                    ctypes.c_char_p(proof_data), ctypes.c_double(self.bound),
                    ctypes.c_char_p(self.norm_type.encode())
                )
            else:
                pd = bytes(proof_data) if isinstance(proof_data, list) else proof_data
                is_valid = self.rust_lib.verify_proof(pd, self.bound, self.norm_type)
            return is_valid, time.time() - start_time
        except Exception as e:
            if __import__('os').getenv('USE_RUST_ZKP', 'false').lower() == 'true':
                print(f"  ⚠️  Rust verify failed ({e}), using simplified")
            return ZKPVerifier(norm_type=self.norm_type, bound=self.bound).verify_proof(proof)
    
    def filter_updates(self, client_updates_with_proofs: list) -> list:
        """Return only (client_id, update_delta) for updates that pass verification."""
        valid_updates = []
        for client_id, update_delta, proof in client_updates_with_proofs:
            if self.verify_proof(proof)[0]:
                valid_updates.append((client_id, update_delta))
            else:
                print(f"  ⚠️  Client {client_id} update rejected: ||Δw_{client_id}^{{t+1}}|| exceeds bound")
        return valid_updates

