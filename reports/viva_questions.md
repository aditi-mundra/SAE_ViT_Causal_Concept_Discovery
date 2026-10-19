# Viva questions and short answers

## Basic (ViT)
1. **What is a ViT?** A transformer that splits an image into patches, embeds them as tokens and processes them with attention + MLP blocks.
2. **Patch size/count here?** 16x16 px, 14x14 = 196 patches (+ CLS) for 224x224 input.
3. **What does attention do?** Mixes information between tokens using learned query-key similarity.
4. **What is the residual stream?** The running token vectors that each block adds to; we hook its output.
5. **Why remove CLS for patch analysis?** CLS summarises the whole image; patch concepts need per-location vectors.
6. **Why freeze the ViT?** We interpret the given model; training it would change what we study.
7. **Which dimension is a token?** 768 for ViT-B/16.

## SAE
8. **What is an SAE?** An autoencoder with a wide sparse hidden layer trained to reconstruct activations.
9. **Why overcomplete?** Superposition packs more concepts than dimensions; a wide basis can separate them.
10. **L1 sparsity?** Adds lambda*||f||_1 so most features are zero.
11. **Top-K SAE?** Keeps only the K largest features per vector; sparsity is exact and no L1 shrinkage.
12. **Why unit-norm decoder?** Stops the model cheating by shrinking activations and growing decoder weights.
13. **L0?** Average number of active features per vector.
14. **Explained variance?** 1 - residual energy / total variance; how much structure is reconstructed.
15. **Dead feature?** One that never activates; wasted capacity.
16. **Standard vs L1 vs Top-K here?** Standard = ReLU without penalty (dense baseline), L1 = penalised, Top-K = hard sparsity.

## Interpretability
17. **Polysemantic?** One unit responds to several unrelated concepts.
18. **Monosemantic?** One unit ~ one concept.
19. **Superposition?** Representing more features than dimensions using near-orthogonal directions.
20. **Class purity?** max_c P(c|feature).
21. **Class entropy?** Spread of a feature over classes; low = specific.
22. **Spatial Concentration Score?** Share of the mean activation map in its top 10% patches.
23. **Why CLIP?** An independent text-aligned lens on what top patches show.
24. **Why not use CLIP as causal evidence?** It describes images, it does not test the model.
25. **Why compare to neurons?** To test whether SAE features are actually more interpretable.

## Causality
26. **Ablation?** Setting a feature to zero and measuring the output change.
27. **Why preserve SAE error?** Replacing x with the reconstruction would change the model on its own.
28. **Amplification?** Scaling a feature up (alpha>1).
29. **Steering?** Adding alpha times a decoder direction.
30. **Counterfactual?** What the output would be had the feature been different.
31. **Why random controls?** Any perturbation changes logits; we need a null.
32. **Why frequency-matched controls?** Rare vs common features differ in how often they matter.
33. **Why logits?** Softmax saturates and hides changes.
34. **Causal effect formula?** CE = z_y(x) - z_y(x with feature ablated).
35. **Can one experiment prove causality?** No; it shows influence of the feature direction at that layer, not necessarily its natural role.
36. **Local vs global intervention?** Local edits only strongly active patches; global all active patches.

## Statistics
37. **Paired t-test?** Tests whether mean per-image differences are zero.
38. **Wilcoxon?** Non-parametric paired test on signs/ranks.
39. **Bootstrap CI?** Resample images with replacement to get a CI of the mean.
40. **Effect size (d_z)?** Mean difference / SD of differences.
41. **Empirical p?** (1 + #random >= candidate) / (R + 1).
42. **Why Holm?** Controls family-wise error across many features.
43. **Why not p-values alone?** Tiny effects can be significant; magnitude and controls matter.

## Experiments
44. **Layer comparison?** Same SAE settings at blocks 3/6/9/11; compare quality and semantics.
45. **Width ablation?** 2048-16384 features; check reconstruction, dead features, interpretability.
46. **Dose-response?** Effect vs multiplier 0-5x; monotone trend supports influence.
47. **Interaction?** I = E_AB - E_A - E_B; negative = redundant, positive = synergistic.
48. **Data split?** 60% SAE training, 20% feature validation, 20% causal evaluation.
49. **Why a causal split?** To avoid selecting and testing on the same images.
50. **Selection bias?** Candidates are chosen on validation proxies, never on causal outcomes.

## Limitations
51. **SAE reconstruction error?** Residual part the SAE misses; preserved, but concepts may live there.
52. **Feature splitting?** One concept splits into several features as width grows.
53. **Feature absorption?** A general feature fails to fire where a specific one covers it.
54. **Distribution shift?** CIFAR upsampled images differ from ImageNet training data.
55. **Why ImageNet logits on CIFAR?** The pretrained head is ImageNet; we use the model's own predicted class.

## Implementation
56. **How do hooks work?** register_forward_hook on blocks[L] captures the output tensor.
57. **forward_from_hidden?** Runs remaining blocks, final norm and head from stored hidden states.
58. **Why memmap?** Activations are GBs; memmap streams from disk.
59. **float16 storage?** Halves disk; compute stays float32.
60. **Device handling?** cuda > mps > cpu automatically.
61. **Reproducibility?** Seeds, saved configs, deterministic splits.
