# 15-slide presentation (fill figures from `experiments/<name>/<run>/plots/`)

**1. Title** – SAE-Based Causal Concept Discovery in ViTs. Points: title, name, guide, date, one-line goal. Visual: pipeline icon strip. Say: "We ask what concepts a ViT holds and whether they cause its predictions."

**2. Motivation** – Points: ViTs are accurate but opaque; heatmaps show where, not what; safety/trust need mechanisms; interpretability needs causal tests; SAEs give sparse parts. Visual: example image + Grad-CAM. Say: "Seeing a heatmap is not understanding."

**3. Problem Statement** – Points: neurons are polysemantic; superposition; need concept units; need evidence they matter; avoid over-claiming. Visual: neuron-vs-concept sketch. Say: "Find the concepts, then test them."

**4. Research Questions** – Points: reconstruction? sparsity? interpretable vs neurons? layers? causal vs random? Visual: RQ list. Say: "Ten questions, answered from measured data."

**5. Vision Transformer** – Points: 16x16 patches; 196 tokens + CLS; blocks of attention+MLP; residual stream; we hook block outputs. Visual: Figure 2 patch grid. Say: "We read the residual stream after block L."

**6. Why ViTs Are Hard to Interpret** – Points: dense 768-d vectors; superposition; attention != explanation; no concept labels; distributed computation. Visual: neuron top-images mixed. Say: "One neuron, many meanings."

**7. Sparse Autoencoders** – Points: encode to wide space; keep few features; decode; L1 or Top-K; unit-norm decoder directions. Visual: Figure 3 SAE diagram. Say: "Few active features per patch, each a direction."

**8. Proposed Architecture** – Points: extract -> SAE -> discover -> interpret -> intervene -> validate; 60/20/20 split. Visual: Figure 1 flow. Say: "Representation, interpretation, localisation, causation, validation."

**9. Feature Discovery** – Points: top-100 patches; heatmaps; purity; spatial concentration; dead-feature checks. Visual: Figure 4+5. Say: "Each feature gets a card of evidence."

**10. Semantic Interpretation** – Points: human inspection; CLIP prompts; class enrichment; CLIP is not causal evidence; monosemantic vs polysemantic. Visual: CLIP + enrichment bars. Say: "Three independent lenses on meaning."

**11. Causal Intervention** – Points: x' = x + (f'-f)W_dec; error preserved; ablate/amplify/steer; global vs local; logit effects. Visual: intervention diagram. Say: "We change one feature and watch the logit."

**12. Experimental Results** – Points: ablation vs random with CI; dose-response; interactions; layer/width/type tables; hypothesis verdicts. Visual: Figures 7-12 (measured). Say: "Report only what the numbers show, including failures."

**13. Grad-CAM Comparison** – Points: Grad-CAM where; SAE what; intervention matters; they can disagree. Visual: 4-panel figure. Say: "Three different questions."

**14. Limitations & Future Work** – Points: reconstruction error; feature splitting/absorption; proxy selection; distribution shift; DINOv2/ViT-L/circuits. Visual: roadmap. Say: "Evidence, not proof."

**15. Conclusion** – Points: pipeline built; claims tied to 3 levels; effects vs random; hypotheses verdicts; reproducible code. Visual: summary table. Say: "A feature is credible only if representation, interpretation and causation agree."
