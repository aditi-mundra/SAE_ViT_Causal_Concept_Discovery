# SAE-Based Causal Concept Discovery in Vision Transformers

_Experiment `smoke`, run `layer_9__topk_256_k8`. All numbers below were produced by this code; nothing is hand-entered._


## Abstract
We train sparse autoencoders (topk, width 256) on patch activations of vit_base_patch16_224 at block 9 using 10 cifar100 images (seed 42). On held-out validation patches the SAE reaches explained variance -0.134 with mean L0 8.0 and 17.6% dead features. Concept candidates are chosen from validation data only and tested by ablation, amplification, steering and frequency-matched random controls on a disjoint causal-evaluation split.

## Introduction & Motivation
ViT activations are polysemantic: single neurons mix many visual concepts (superposition). Sparse autoencoders re-express activations in an overcomplete sparse basis whose directions may be interpretable. Heatmap methods show where the model looks, not what is represented or whether it matters.

## Problem Statement & Research Questions
Can SAEs decompose ViT representations into interpretable concepts, and do controlled interventions show that these concepts causally influence predictions? (RQ1-RQ10 are answered below with measured values.)

## Hypotheses (declared before running)
H1 SAE features more interpretable than neurons; H2 later layers more semantic; H3 ablation > random ablation; H4 amplification raises the aligned logit for some features; H5 local sometimes > global; H6 some concepts are redundantly represented. Verdict rules are documented in `causal/hypotheses.py`.

## Related Work
ViT (Dosovitskiy et al.), Grad-CAM (Selvaraju et al.), superposition/toy models (Elhage et al.), dictionary learning for LLM interpretability (Bricken et al.; Cunningham et al.), Top-K SAEs (Gao et al.), activation steering and causal scrubbing, CLIP (Radford et al.). See References.

## Methodology
ViT -> block-L patch activations (CLS kept separately) -> normalisation -> SAE -> feature discovery (top-100 patches, class purity/entropy, spatial concentration, CLIP labels) -> candidate selection (validation only) -> intervention x' = x + (f'_i - f_i) W_dec[i] preserving SAE error -> logit-space effects -> random controls, bootstrap, paired tests, Holm correction.

## Experimental Setup
Model `vit_base_patch16_224`, dataset `cifar100` (10 images), split (0.6, 0.2, 0.2) (SAE-train / feature-val / causal-eval), normalisation `standardize`, SAE `topk` width 256 k=8 lambda=0.003 lr=0.001 epochs=1.

## Results: SAE quality (RQ1, RQ2)
- mse: 1.099
- explained_variance: -0.1339
- l0: 8
- dead_pct_eval: 17.58
- mean_feature_density: 0.03125
- n_vectors: 392
- dead_pct_train_last_epoch: 0

## Layer / Width / SAE-type Comparison (RQ4)
| run_tag | layer | sae_type | width | k | mse | explained_variance | l0 | dead_pct_eval | median_purity | median_class_entropy_norm | median_spatial_conc10 | median_top_patch_purity | median_interp_score | pct_highly_specific | pct_polysemantic | neuron_median_interp_score | clip_mean_score | n_candidates | mean_ablation_logit_effect | median_ablation_logit_effect | mean_random_effect | mean_amp_delta_logit | causal_minus_random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| layer_9__topk_256_k8 | 9 | topk | 256 | 8 | 1.099 | -0.134 | 8.000 | 17.578 | 0.948 | 0.044 | 1.000 | 1.000 | 0.919 | 100.000 | 0.000 | 0.648 |  | 3 | 0.001 | 0.001 | 0.000 | 0.001 | 0.001 |

## Causal Analysis (RQ5-RQ8)
| feature_id | concept | n_images | abl_mean | abl_ci_lo | abl_ci_hi | abl_dz | abl_rand_rand_mean | abl_rand_p_empirical | abl_p_holm_paired | amp_mean | dose_spearman_rho | causal_rank | interp_rank | combined_rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | shark | 2 | 0.001 | 0.001 | 0.001 | 10.019 | 0.002 | 0.667 | 1.000 | 0.001 | 1.000 | 2.000 | 1.000 | 2.000 |
| 1 | shark | 2 | 0.006 | 0.003 | 0.009 | 1.339 | -0.002 | 0.333 | 1.000 | 0.005 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | seal | 1 | -0.003 | -0.003 | -0.003 |  | 0.001 | 0.500 | 1.000 | -0.002 | -0.900 | 3.000 | 1.000 | 2.000 |

## Spatial causal effects (RQ9)
| feature_id | position | n_active_patches | spearman_activation_vs_effect | sum_single_patch_effects | global_effect | max_patch_effect | top_effect_patch |
|---|---|---|---|---|---|---|---|
| 0 | 3 | 1 |  | 0.001 | 0.001 | 0.001 | 182 |

## Feature Interaction (RQ10)
| effect_a | effect_b | effect_ab | interaction | ci_lo | ci_hi | p_t | p_wilcoxon | category | n | feature_a | feature_b | decoder_cosine |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 | 0.006 | 0.007 | 0.000 | 0.000 | 0.000 | 0.471 | 0.500 | synergistic | 2 | 0 | 1 | -0.046 |
|  |  |  |  |  |  |  |  | insufficient_data | 1 | 0 | 2 | -0.025 |
|  |  |  |  |  |  |  |  | insufficient_data | 1 | 1 | 2 | 0.070 |

## Grad-CAM Comparison
Grad-CAM: where is the model focusing? SAE feature: what internal feature is active? Intervention: does changing it change the prediction?
| feature_id | position | spearman_gradcam_vs_feature | spearman_gradcam_vs_ablation | spearman_feature_vs_ablation |
|---|---|---|---|---|
| 0 | 3 | -0.037 | -0.037 | 1.000 |

## Hypotheses: verdicts
- **H1**: Supported
- **H2**: Not tested
- **H3**: Not supported
- **H4**: Not supported
- **H5**: Partially supported (per-modified-patch effect only)
- **H6**: Not supported

## Automatic warnings
- SAE reconstruction is poor (explained variance -0.134 < 0.70)
- feature 0: ablation effect not distinguishable from random (p_emp=0.667)
- feature 1: ablation effect not distinguishable from random (p_emp=0.333)
- feature 2: ablation effect not distinguishable from random (p_emp=0.500)

## Figures
- `experiments\smoke\layer_9__topk_256_k8\plots\ablation_amplification.png`
- `experiments\smoke\layer_9__topk_256_k8\plots\causal_effect_distribution.png`
- `experiments\smoke\layer_9__topk_256_k8\plots\dose_response.png`
- `experiments\smoke\layer_9__topk_256_k8\plots\feature_clusters.png`
- `experiments\smoke\layer_9__topk_256_k8\plots\feature_interaction.png`
- `experiments\smoke\layer_9__topk_256_k8\plots\feature_quality.png`
- `experiments\smoke\layer_9__topk_256_k8\plots\gradcam_vs_sae_feature0.png`
- `experiments\smoke\layer_9__topk_256_k8\plots\local_vs_global.png`
- `experiments\smoke\layer_9__topk_256_k8\plots\random_controls.png`
- `experiments\smoke\layer_9__topk_256_k8\plots\sae_training.png`
- `experiments\smoke\layer_9__topk_256_k8\plots\sae_vs_neurons.png`
- `experiments\smoke\layer_9__topk_256_k8\plots\steering_curves.png`

## Limitations
SAE reconstruction error is not zero; feature splitting and absorption can blur concepts; candidate selection uses proxy metrics; effects are measured for the model's own predicted class (ImageNet head) on CIFAR-100 images (distribution shift); single seed unless repeated; CLIP labels are interpretation, not evidence; ablation at one layer ignores compensating pathways.

## Future Work
DINOv2, ViT-L, ImageNet-1K, CLIP vision encoder, video/multimodal transformers, automated concept discovery, causal feature circuits, feature graphs, cross-layer feature tracking.

## Conclusion
Conclusions must rest on all three levels: representation (feature exists), interpretation (independent evidence for what it encodes), and causality (controlled intervention beats random controls). See hypothesis verdicts above.

## References
Dosovitskiy 2021 (ViT); Selvaraju 2017 (Grad-CAM); Elhage 2022 (Toy Models of Superposition); Bricken 2023 (Towards Monosemanticity); Cunningham 2023 (Sparse Autoencoders Find Highly Interpretable Features); Gao 2024 (Scaling and evaluating sparse autoencoders); Radford 2021 (CLIP); Geiger 2021 / Chan 2022 (causal abstraction / scrubbing).