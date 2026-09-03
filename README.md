# RHS-RCMTR — Multimodal SAR/optical segmentation research code (R6 SAR-DE)

Configuration-driven PyTorch research code for **remote-sensing multimodal
representation learning (SAR–optical alignment)**. One model factory and one
training entry point serve every experimental row: frozen-baseline static
fusion, the verified R5 candidate (OA-SCRT), and the R6 SAR-DE route
(SAR representation enhancement + evidence-conditioned residual correction).

Real data and pretrained weights never leave the cloud host; local tests run
on generated tensors only, and the target test split stays sealed until an
authorized final-test run.

## What is inside (R6 SAR-DE, 2026-09)

- `sar_encoder_v2` — four-layer residual multi-scale single-band SAR encoder
  (random init, no external weights).
- Distillation pre-training protocol (two-run chain): run 1 produces a
  pre-training anchor; runs 2 (baseline vs candidate) load it as their common
  initialization.
- `R6-C1-EVSCRT` — evidence-conditioned bounded residual correction router
  (`EvScrtRouter`): optical-anchor output + gated SAR residual, where the gate
  is supervised by a soft box `(1 - u_o) <= gate <= q_s` on P1-audited
  reliability evidence (`evidence_response_gate_loss`).
- Training-time degradation injection (`r6_train_inject_v1`) so the gate learns
  to respond to low-quality evidence during training.
- Gradient-balance hook (`sar_grad_scale`) shared by both comparison arms.
- Frozen decision gates (`r6_gates.py`): G1 pre-training gate, G2 rapid gate,
  G3 formal gate, q-permutation detectability, paired bootstrap — pure
  functions, locally unit-tested.
- Claim–implementation alignment registry (`CLAIM_ALIGNMENT_TABLE`) with a CI
  guard (`tests/unit/test_claim_alignment.py`).
- Auditing CLIs for the cloud: `audit_sar_competence_v2.py` (G1 gate),
  `audit_q_permutation_detectability.py`, `audit_proxy_equivalence.py`,
  `audit_claim_impl_alignment.py`, `create_distill_pretrain_anchor.py`.

## Repository layout

```
configs/        runtime / experiment rows (configs/experiment/r6 for the R6 arms)
environment/    locked requirements
manifests/      clean-sync snapshot manifest + code snapshot manifest
scripts/        train / evaluate / audit entry points
src/rhs_rcmtr/  data, engine, losses, mechanisms, metrics, models, utils
tests/          local unit/integration tests (pytest, synthetic only)
review/         code-review receipts
```

## Local check

```bash
pip install -e .            # python == 3.11, torch 2.5.1 (see environment/)
pytest tests -q             # synthetic-only, no data/GPU required
```

## Cloud runs

Every real run is bound by four hashes (run / data / pretrained-audit /
code manifests) plus a `resolved_config_sha256`; the entry point fails closed
unless all bindings match. See `manifests/clean_sync_manifest.json` for the
exact code snapshot.
