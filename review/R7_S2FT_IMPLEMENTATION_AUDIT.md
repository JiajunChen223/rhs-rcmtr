# R7-S2FT implementation cross-audit

Status: **DRAFT / NOT RELEASED**  
Branch: `r7-s2ft-implementation`  
Scientific-evidence status: **false until real-data runs and the cloud-only DINO audit pass**

## Frozen scientific contract

R7 is an architecture-level internal delta. The parent optimization/data protocol is not allowed to move. `configs/experiment/r7/baseline.yaml` and `r7_s2ft_full.yaml` therefore share the same split, preprocessing, augmentation, sampler, optimizer, learning rate, formal/rapid horizons, seed semantics, batch budget, SAR gradient scale, and loss/degradation graph. Their intended only row-level difference is `enabled_mechanism_ids`.

R7 uses the existing `r6_train_inject_v1` only because it is part of the frozen parent protocol. Reliability evidence is not consumed by the R7 model forward and has no R7 loss term.

## Independent review roles

### 1. Architecture implementation review

Checks:
- exactly one shared frozen DINO transformer is registered;
- single-band SAR has its own trainable patch projection initialized from DINO RGB patch filters;
- sensor adapters are modality-specific and identity initialized;
- DINO is kept in eval mode but is **not** wrapped in `torch.no_grad`, so gradients can pass through frozen blocks to trainable adapters/SAR patch projection;
- DINO blocks are executed lock-step because intermediate GLCI outputs must feed subsequent foundation blocks;
- GLCI uses a 3x3 geo-local stencil plus bounded learned offset and starts as identity;
- SDR is semantic conditioned and starts as identity.

Cross-review fixes already applied:
1. Removed a registered `source_patch_embed` reference from the SAR patch transplant because it duplicated the shared DINO module in the parameter/state tree.
2. Replaced the high-cost direct 3x3 offset predictor on 2C channels with a 1x1 bottleneck followed by a zero-initialized 3x3 offset head.
3. Added frozen-block activation checkpointing for the trainable-through-frozen-foundation path.

Static result: **PASS**. Executable result: **requires pytest / cloud audit**.

### 2. Scientific-logic review

Checks:
- R7 v1 adds no auxiliary scientific loss: segmentation objective remains the parent objective;
- reliability, honesty, CRM, and evidence-response supervision do not enter the R7 model;
- compatibility `route_weights` exist only for the legacy engine API and are explicitly tagged `compatibility_constant_not_model_output`;
- zero-initialized adapters/GLCI/SDR prevent random innovation branches from corrupting the foundation representation at initialization;
- explicit modality masks are reapplied after every shared transformer block, after adapters, and after GLCI so an absent modality cannot regenerate nonzero patch tokens through CLS/register/bias interactions.

Static result: **PASS**. Performance claim: **NOT TESTED**.

### 3. Protocol/fairness review

Machine-check target in `test_s2ft_candidate_keeps_parent_protocol_and_loss_graph`:
- `assert_training_object_parity(parent, candidate)`;
- matched protocol budget hashes equal;
- exact loss dictionaries equal;
- exact loss-graph signatures equal;
- only `enabled_mechanism_ids` differs at the experiment-row level.

Initialization isolation:
- `router.*` and `innovation.*` are excluded from common trainable tensors;
- R7's architecture trainables are reset only through `S2FTSegmenter.reset_innovation_parameters` under `innovation_init_seed`;
- the reset path does not recurse into the frozen DINO model;
- the existing immutable common-init anchor still binds the run to the parent protocol, but R7 correctly reports zero trainable common tensors because it is a full architecture delta rather than a router-only delta.

Static result: **PASS**. Runtime receipt: **PENDING**.

### 4. Adversarial failure review

Guarded failure modes:
- accidental `torch.no_grad` around the shared foundation;
- duplicate DINO registration through SAR transplant source reference;
- hidden reliability dependence;
- fake SAR-off caused by patch/position/CLS/bias leakage;
- innovation reset corrupting pretrained DINO;
- unconstrained global cross-modal attention;
- unbounded deformable offsets;
- legacy honesty/counterfactual loss consuming R7 compatibility weights;
- accidental parent loss/protocol drift.

Static result: **PASS** for implemented guards.

## Required executable checks before release

The PR must remain draft until all of the following are executed successfully:

1. `python -m compileall -q src tests`
2. `pytest -q`
3. Real-checkpoint optical equivalence audit:
   `python scripts/audit_r7_optical_foundation_equivalence.py ...`
4. The equivalence receipt must satisfy `max_abs_error <= 1e-5` and zero SAR feature leakage under SAR-off.
5. The real model parameter audit must show every DINO parameter frozen and all R7 trainables under the declared innovation subtree.
6. A synthetic/cloud smoke run must complete with the parent loss graph and no R7 honesty/evidence/counterfactual term.

No result produced before these checks may be described as scientific evidence.

## First real experiment order after release

The first real run is the R7 full candidate against the frozen R7 parent under the matched 6-epoch rapid protocol. No hyperparameter rescue is authorized from the implementation branch. If architectural ablations are needed, they must be added as explicit predeclared rows rather than ad-hoc edits to the full candidate.
