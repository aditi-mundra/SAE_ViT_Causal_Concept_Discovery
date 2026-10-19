# SAE-Based Causal Concept Discovery and Feature Steering in Vision Transformers

Pipeline: ViT -> block-L patch activations -> SAE (standard / L1 / Top-K) -> feature discovery -> CLIP / class / spatial
interpretation -> causal intervention (ablation, amplification, steering) -> random controls + statistics -> reports.

## Setup
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Verify the install (offline, no downloads: tiny random ViT + synthetic images)
    pytest tests -q
    python run_pipeline.py --mode selftest

## Run
    python run_pipeline.py --mode smoke                 # 10 images, real ViT-B/16
    python run_pipeline.py --mode demo                  # CIFAR-100, 500 imgs, layer 9, 1024 feats, Top-K=16
    python run_pipeline.py --mode full                  # 10k imgs, layer 9, 4096 feats
    python run_pipeline.py --mode full --sweep          # + layers 3/6/9/11, widths, SAE types, Top-K 16/32/64
    streamlit run app.py

## Stage commands
    python extract.py --mode demo --layer 9
    python train.py --mode demo --sae-width 4096 --sae-type topk --k 32
    python discover.py --mode demo [--feature 172]
    python intervene.py --mode demo --feature 172 --strength 0 [--steer] [--local 90]
    python evaluate.py --mode demo

## Outputs (experiments/<name>/<layer_L__type_width[_k]>/)
config.json, experiment_config.json, checkpoint.pt, metrics.csv, results.json, warnings.json, plots/,
discovery/ (feature_summary.csv, neuron_summary.csv, feature_top_patches.csv.gz, clusters.csv),
causal/ (candidate_causal_summary.csv, random_controls.json, dose_response.json, interactions.csv, steering.csv,
local_vs_global.csv, spatial_summary.csv, gradcam_comparison.csv, results_rows.csv), layer_summary.json.
Experiment level: comparison.csv, hypotheses.json/.md; global: experiments/results.csv; report: reports/report_<name>.md.

## Scientific design
Splits 60/20/20 (SAE-train / feature-val / causal-eval). Candidates are picked from validation, non-causal metrics only.
Effects: logit_effect = z_y(base) - z_y(intervened) for the model's baseline top-1 class. Random controls are
frequency-matched features. Intervention: x' = x + (f'_i - f_i) W_dec[i] (SAE error preserved).
Note: the pretrained head is ImageNet-1k, so effects on CIFAR-100 images are measured on the model's own predicted class.
