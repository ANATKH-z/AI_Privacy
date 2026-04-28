# Groth16 / `zkp_rust` troubleshooting

This note is for when **`maturin develop`** or **`cargo build`** fails. The template is **intended to work** with the bundled `exercise4/groth16` + `zkp_rust`; failures are usually environment or dependency skew.

**Coursework:** your submission must use a **real ZKP** (this Groth16 path or your own). The Python “simplified” prover is **not** an acceptable final solution—see the main README and coursework PDF.

---

## Current configuration (default)

`zkp_rust/Cargo.toml` is set up as:

- **`ark-groth16`** from the **local path** `../groth16`
- **arkworks** crates pinned via **git** (and `[patch.crates-io]`) to match `groth16`

So you do **not** need to switch to crates.io unless you hit compile errors and want to try a stable stack.

---

## If the build fails

Typical causes:

1. **Git dependencies** moved ahead; `groth16` and `zkp_rust` must stay aligned.
2. **Rust toolchain** too old; use a recent stable Rust (`rustup update`).

**Try in order:**

1. From repo root:
   ```bash
   cd exercise4/groth16 && cargo build
   cd ../zkp_rust && maturin develop
   ```
2. If `groth16` fails: update `groth16` / lockfiles per that crate’s README, then mirror the same arkworks versions in `zkp_rust/Cargo.toml`.

### Alternative: published crates.io versions

If git-based arkworks keeps breaking on your machine, you can try **published** `0.5.x` lines for `ark-groth16`, `ark-bn254`, etc., in both `groth16` and `zkp_rust` so they **match**. This is more maintenance but avoids moving git commits.

---

## Not for submission

A **placeholder** Rust/Python path (signatures only, no sound ZKP) is useful only to debug the FL loop locally. **Do not** submit that as Exercise 4—markers expect Groth16 (built from this project) or a **documented self-implemented ZKP**.
