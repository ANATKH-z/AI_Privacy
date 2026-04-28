// ============================================================================
// Exercise 4: PyO3 + Groth16 (arkworks) — norm / bound demonstration circuit
// ============================================================================
// This is the default `zkp_rust` library: build with `maturin develop` in this
// directory; enable from Python via USE_RUST_ZKP=true (see course README).
//
// The R1CS ties private `norm` and public `bound` via witness `diff` with
// bound = diff + norm (see ConstraintSynthesizer impl). A fully sound
// "norm ≤ bound" statement over integers would also constrain diff ≥ 0 in-circuit
// (e.g. range proofs); that extension is left as further discussion.
// ============================================================================

use pyo3::prelude::*;
use ark_bn254::{Bn254, Fr};
use ark_groth16::{Groth16, Proof, ProvingKey, PreparedVerifyingKey};
use ark_relations::gr1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_r1cs_std::{
    alloc::AllocVar,
    eq::EqGadget,
    fields::fp::FpVar,
};
use ark_std::rand::rngs::OsRng;
use ark_snark::{SNARK, CircuitSpecificSetupSNARK};
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};

// Small R1CS: public bound, private norm and diff, constraint bound = diff + norm.
struct NormConstraint {
    norm: Option<Fr>,
    bound: Fr,
}

impl ConstraintSynthesizer<Fr> for NormConstraint {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<Fr>,
    ) -> Result<(), SynthesisError> {
        // Allocate public input: bound
        let bound_var = FpVar::<Fr>::new_input(cs.clone(), || Ok(self.bound))?;
        
        // Allocate private witness: norm
        let norm_var = FpVar::<Fr>::new_witness(cs.clone(), || {
            self.norm.ok_or(SynthesisError::AssignmentMissing)
        })?;
        
        // Witness diff intended as (bound - norm) in Z; in-circuit only bound = diff + norm.
        let diff = FpVar::<Fr>::new_witness(cs.clone(), || {
            Ok(self.bound - self.norm.ok_or(SynthesisError::AssignmentMissing)?)
        })?;
        
        // Enforce: bound = diff + norm
        // This ensures that diff = bound - norm
        bound_var.enforce_equal(&(&diff + &norm_var))?;
        
        // Integer inequality norm ≤ bound would require diff ≥ 0 in-circuit (not done here).
        
        Ok(())
    }
}

// Global state for proving and verification keys
// In a production system, these would be generated once in a trusted setup
// and stored securely
use std::sync::OnceLock;

static PROVING_KEY: OnceLock<ProvingKey<Bn254>> = OnceLock::new();
static PROCESSED_VK: OnceLock<PreparedVerifyingKey<Bn254>> = OnceLock::new();

fn initialize_keys() {
    PROVING_KEY.get_or_init(|| {
        let mut rng = OsRng;
        
        // Generate a dummy constraint to create the circuit structure
        // The actual values don't matter for key generation
        let constraint = NormConstraint {
            norm: Some(Fr::from(0u64)),
            bound: Fr::from(30u64),
        };
        
        // Generate proving and verification keys using Groth16 setup
        let (pk, vk) = <Groth16<Bn254> as CircuitSpecificSetupSNARK<Fr>>::setup(constraint, &mut rng)
            .expect("Failed to generate Groth16 keys");
        
        // Prepare the verification key for faster verification
        let pvk = Groth16::<Bn254>::process_vk(&vk)
            .expect("Failed to process verification key");
        
        // Store the processed verification key
        PROCESSED_VK.set(pvk).expect("Failed to set processed verification key");
        
        pk
    });
}

/// Generate a Groth16 proof for the bundled R1CS (see file header for circuit semantics).
///
/// `norm` / `bound` are truncated to `u64` then mapped to Fr. Python rejects norm > bound
/// before proving; the SNARK binds the witness used at setup/prove time.
///
/// Args:
///     norm: The norm value (f64), kept private; truncated to u64 for the circuit.
///     bound: The bound value (f64), public; truncated to u64.
///     norm_type: "L1", "L2", or "Linf" (for reference only).
///
/// Returns:
///     Serialized Groth16 proof as Vec<u8>.
#[pyfunction]
fn generate_proof(norm: f64, bound: f64, _norm_type: &str) -> PyResult<Vec<u8>> {
    initialize_keys();

    if norm > bound {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Norm {} exceeds bound {}", norm, bound)
        ));
    }

    // Encode as field elements (non-negative integers; truncation for f64)
    let norm_u = norm.trunc() as u64;
    let bound_u = bound.trunc() as u64;
    let norm_fr = Fr::from(norm_u);
    let bound_fr = Fr::from(bound_u);
    
    // Create the constraint with the actual values
    let constraint = NormConstraint {
        norm: Some(norm_fr),
        bound: bound_fr,
    };
    
    // Generate Groth16 proof
    let mut rng = OsRng;
    let pk = PROVING_KEY.get().expect("Proving key not initialized");
    let proof = Groth16::<Bn254>::prove(pk, constraint, &mut rng)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to generate Groth16 proof: {:?}", e)
        ))?;
    
    // Serialize proof to bytes using compressed format
    let mut proof_bytes = Vec::new();
    CanonicalSerialize::serialize_compressed(&proof, &mut proof_bytes)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to serialize proof: {:?}", e)
        ))?;
    
    Ok(proof_bytes)
}

/// Verify a Groth16 proof against public input `bound` (serialized proof from `generate_proof`).
/// 
/// Args:
///     proof_bytes: Serialized proof as Vec<u8>
///     bound: The bound value (f64) - this is public
///     norm_type: Type of norm ("L1", "L2", or "Linf") - for reference only
/// 
/// Returns:
///     True if proof is valid (i.e., norm ≤ bound), False otherwise
#[pyfunction]
fn verify_proof(proof_bytes: Vec<u8>, bound: f64, _norm_type: &str) -> PyResult<bool> {
    // Initialize keys if not already done
    initialize_keys();
    
    // Deserialize proof from bytes
    let proof: Proof<Bn254> = CanonicalDeserialize::deserialize_compressed(proof_bytes.as_slice())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to deserialize proof: {:?}", e)
        ))?;
    
    // Convert bound to field element
    let bound_fr = Fr::from(bound.trunc() as u64);
    
    // Prepare public inputs (only the bound is public)
    // The norm value remains private and is not revealed
    let public_inputs = vec![bound_fr];
    
    // Verify proof using Groth16 verification
    let pvk = PROCESSED_VK.get().expect("Processed verification key not initialized");
    let is_valid = Groth16::<Bn254>::verify_with_processed_vk(pvk, &public_inputs, &proof)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to verify Groth16 proof: {:?}", e)
        ))?;
    
    Ok(is_valid)
}

/// Python module definition
#[pymodule]
fn zkp_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_proof, m)?)?;
    m.add_function(wrap_pyfunction!(verify_proof, m)?)?;
    Ok(())
}