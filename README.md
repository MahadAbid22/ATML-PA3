# ATML PA3: Fine-Tuning, Vision-Language Grounding and Test-Time Compute

**Course:** EE-5102/CS-6304 Advanced Topics in Machine Learning (Fall 2026)  
**Author:** Muhammad Mahad Abid

This repository contains the code, configurations, split and example IDs, machine-readable results, tables and
figures for the three PA3 tasks. Each experiment is a Python module with one stage per process. A stage can be
stopped and started again, and it writes its numbers to `results/`. Every table and figure in the report is built
from those files by the `analyze` module of the task, so every reported number can be traced back to a saved file.

| Task | Topic | Package |
|---|---|---|
| 1 | Fine-tuning Qwen3-0.6B on MetaMathQA with Full FT, Full FT + L2-SP, LoRA (r=4, r=16) and OPLoRA (r=16, k=128) | `task1_finetune/` |
| 2 | Building a SigLIP and Qwen3 vision-language model and testing whether it really uses the image | `task2_vlm/` |
| 3 | Test-time compute: best-of-N selection, self-certainty, VisConf, rank-weighted consensus, live refinement and adaptive sampling | `task3_ttc/` |

## Repository structure

```text
PA3/
├── README.md, requirements.txt, .gitignore
├── configs/           # task1.yaml, task2.yaml, task3.yaml: every hyperparameter used
├── common/            # seeds and paths, chat template, LoRA/OPLoRA module, plotting and LaTeX helpers
├── scripts/           # setup_env.sh, run_all.sh
├── task1_finetune/    # data, train, export, evaluate, qualitative, diagnostics, analyze
├── task2_vlm/         # download, data, features, model, train, evaluate, analyze, vqa_norm
├── task3_ttc/         # answers, data, core, selection, signals, adaptive, live, analyze
├── data/              # saved IDs only: Task 1 training indices and probes, Task 2 selection, Task 3 splits and orders
├── results/           # JSON, JSONL, CSV and parquet results, plus the log of every stage
└── report/            # generated LaTeX tables and PDF/PNG figures
```

Not committed (see `.gitignore`): `.venv/`, `data/supplied/`, raw datasets, cached features, checkpoints and merged
models. These are written to `outputs/`, or to `$PA3_OUTPUTS` if that is set.

## Setup

You need Linux, Python 3.12 and an NVIDIA GPU with a CUDA 12.6 or newer driver. One 24 GB card is enough.

```bash
DATA_URL="<ATML-PA3-Data Google Drive folder>" bash scripts/setup_env.sh
```

The script creates `.venv` with the pinned versions in `requirements.txt` (PyTorch 2.14.1+cu126, Transformers
5.22.2, lm-evaluation-harness 0.4.13, math-verify 0.9.0) and downloads the course-supplied Task 3 rollouts into
`data/supplied/`. The rollouts are not redistributed here. Public models and datasets are fetched from the Hugging
Face Hub, cocodataset.org and visualqa.org the first time they are needed.

Hardware used: NVIDIA RTX 4090 (24 GB), AMD Ryzen 9 5900X (24 vCPU), 64 GB RAM, NVIDIA driver 595.99, Ubuntu container
on Vast.ai. All reported numbers come from one run on this machine. `results/environment.json` records the exact
environment.

## Reproducing everything

```bash
bash scripts/run_all.sh
```

`run_all.sh` runs two lanes side by side. The GPU lane runs all GPU jobs strictly one at a time, so training times and
peak memory are not affected by other GPU work. The CPU lane does the preparation that needs only CPU or network, which
is the Task 2 data selection and image download and the Task 3 offline analyses.

Each finished stage leaves a marker in `outputs/markers/`, so running the script again skips what is already done, and
training resumes from its last checkpoint. Logs go to `results/logs/`. The reported run includes the optional 49-token
Task 2 experiment. Use `OPTIONAL=0 bash scripts/run_all.sh` to skip it.

## Reproducing each task

Activate the environment first with `source .venv/bin/activate`.

### Task 1

```bash
python -m task1_finetune.data                        # 40,000 MetaMathQA examples, source order, filtered
python -m task1_finetune.diagnostics --make-probes   # fixed 32 + 32 probe set (seed 6304)
python -m task1_finetune.train --method full_ft      # full_ft | l2sp | lora_r4 | lora_r16 | oplora_r16
python -m task1_finetune.export --condition full_ft  # also full_ft_mid; adapters are merged into fresh backbones
python -m task1_finetune.evaluate --condition baseline
python -m task1_finetune.qualitative --condition baseline
python -m task1_finetune.qualitative --collect
python -m task1_finetune.diagnostics --run
python -m task1_finetune.analyze
```

Run `train`, `export`, `evaluate` and `qualitative` once for each method or condition.

### Task 2

```bash
python -m task2_vlm.download --annotations
python -m task2_vlm.data
python -m task2_vlm.download --images
python -m task2_vlm.features
python -m task2_vlm.train          # add --tokens 49 for the optional run
python -m task2_vlm.evaluate
python -m task2_vlm.analyze
```

### Task 3

```bash
python -m task3_ttc.data
python -m task3_ttc.selection
python -m task3_ttc.signals
python -m task3_ttc.adaptive
python -m task3_ttc.live
python -m task3_ttc.analyze
```

## Where the results are

| Task | Tables | Figures | Source data |
|---|---|---|---|
| 1 | `report/tables/task1_*.tex` | `report/figures/task1_*` | `results/task1/` |
| 2 | `report/tables/task2_*.tex` | `report/figures/task2_*` | `results/task2/tokens196/`, `results/task2/tokens49/` |
| 3 | `report/tables/task3_*.tex` | `report/figures/task3_*` | `results/task3/` |

## Implementation notes

The assignment fixes most of the protocol. These are the decisions it leaves open and the values I used. All of them
are in the configs.

### Task 1

Training uses the Qwen3 chat template with `enable_thinking=False`, so the prompt ends with the empty think block and
the loss covers the response plus `<|im_end|>`. Overlap with GSM8K is checked on normalised text (NFKC, lower case,
alphanumerics only) for both `query` and `original_question`. Logits are computed only at supervised positions, which
gives the same result as full-sequence cross-entropy with masking but uses less memory.

For L2-SP, the exact penalty gradient λ(θ−θ₀) is added to the unscaled gradient before clipping. This is the same as
back-propagating (λ/2)‖θ−θ₀‖² and avoids an FP32 autograd graph over all parameters. The penalty is a sum over all
unique parameters, with tied embeddings counted once, and θ₀ is held in FP32 on the GPU.

LoRA and OPLoRA share one implementation in `common/lora.py`. A uses Kaiming-uniform initialisation (a=√5) and B starts
at zero, as in PEFT. The same seed gives LoRA r=16 and OPLoRA r=16 identical initial factors. OPLoRA bases are computed
once per weight in FP32 with `torch.linalg.svd`, and the projected factors are formed on every forward pass and exported
for inference.

The cosine schedule (`get_cosine_schedule_with_warmup`) starts at LR 0 on the first update. Skipped FP16 updates still
count toward the 1,250 updates and are logged. Once a run is exported, the first- and third-quarter recovery checkpoints
and all optimizer states are deleted to save disk, after their sizes are logged.

The reported GSM8K metric is the stock `gsm8k_cot` flexible-extract exact match. I also save a diagnostic column (the
number in the last `\boxed{}`, otherwise the last number) to separate answer-format effects from correctness. It is
never used as the primary score. The log-likelihood benchmarks use batch size 16 because batch size 64 ran out of
memory. This changes throughput only.

The fixed 42-response annotation in `results/task1/qualitative/annotation_subset.csv` is scored by hand, and `analyze`
summarises it in `report/tables/task1_annotation.tex`.

### Task 2

Grounding evaluation pairs are split evenly between presence and counting (dev 12 + 12, test 32 + 32). The training
pools hold 6,000 presence and 3,000 counting pairs, built only on training images. To limit label noise, a presence
positive needs an instance covering at least 1% of the image, and for counting there can be no crowd annotation and
every counted instance must cover at least 0.25% of the image.

The instruction is "Write a short caption for this image." for captions, and the question followed by "Answer the
question using a single word or phrase." for QA. The text is the same in every image condition.

The connector is calibrated on training data only. A fixed scale makes the mean token L2 norm of the initial connector
output equal the mean norm of the text-token embeddings (`results/task2/tokens196/calibration.json`). Each stage uses a
fresh AdamW with new parameter groups, and one dynamic loss scaler is shared across stages. Weight decay excludes
biases and 1-D parameters, as in Task 1. VQA answer normalisation and soft accuracy re-implement the official rules.

### Task 3

MATH answers are grouped by math-verify equivalence, checked in both directions and without gold answers. MathVista
answers are normalised with the MathVista rules: option letters for multiple choice, and integer, float with precision
or list otherwise. LaTeX wrappers and units are removed before numeric conversion, and non-numeric outputs for numeric
questions form their own incorrect groups.

One MATH question (`test:4928`, evaluation split) has answers such as `3^{5050}` and power towers that make SymPy hang.
Its 32 answers are grouped by normalised string equality instead, and none of them matches the gold answer under either
method. It is listed in `results/task3/data_stats_math.json` under `string_fallback_questions`.

The ten orders per question come from one `default_rng(6304)` stream per dataset, drawn in sorted question-ID order, and
are saved in `data/task3/orders_*.npz`. Live generation uses Hugging Face `generate` (FP16, batch size 1), and each call
is seeded with 6304 + 100 × question index + call index.

## Attribution

VQA answer normalisation follows the official code (GT-Vision-Lab/VQA, `vqaEval.py`) and is re-implemented in
`task2_vlm/vqa_norm.py`. MathVista answer normalisation follows `normalize_extracted_answer` in lupantech/MathVista and
is re-implemented in `task3_ttc/answers.py`. The qualitative prompts in `data/task1/first_plot_questions.yaml` are taken
unchanged from the Emergent Misalignment repository (Betley et al., 2025). Benchmark scoring uses EleutherAI
lm-evaluation-harness and math equivalence uses Hugging Face math-verify.

Code was written with assistance from an LLM coding assistant, as the assignment permits.
