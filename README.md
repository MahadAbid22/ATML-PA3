# ATML PA3: Fine-Tuning, Vision-Language Grounding, and Test-Time Compute

Code, configurations, split and example IDs, and machine-readable results for EE-5102/CS-6304
Programming Assignment 3:

| Task | Topic | Package |
|---|---|---|
| 1 | Qwen3-0.6B fine-tuning on MetaMathQA: Full FT, Full FT + L2-SP, LoRA r=4/16, OPLoRA r=16 (k=128) | `task1_finetune/` |
| 2 | SigLIP–Qwen3 VLM: connector + LoRA curriculum, complementary-pair grounding analysis | `task2_vlm/` |
| 3 | Test-time compute: best-of-N selection, self-certainty, VisConf, RWC, live refinement, adaptive sampling | `task3_ttc/` |

Every experiment is a Python module with one stage per process. Stages are resumable and write
machine-readable results to `results/`. Tables (`report/tables/*.tex`, booktabs) and figures
(`report/figures/*.pdf`) are rebuilt from those files by each task's `analyze` module.

## 1. Environment

Linux, Python 3.12, an NVIDIA GPU with a CUDA ≥ 12.6 driver (a single 24 GB card is sufficient).

```bash
DATA_URL="<ATML-PA3-Data Google Drive folder>" bash scripts/setup_env.sh
```

The script creates `.venv` with the pinned versions in `requirements.txt` (PyTorch 2.14.1+cu126,
Transformers 5.22.2, lm-evaluation-harness 0.4.13, math-verify 0.9.0). It also downloads the
course-supplied Task 3 rollouts into `data/supplied/`; these are not redistributed here. Public models
and datasets are fetched from the Hugging Face Hub, cocodataset.org and visualqa.org on first use.
`results/environment.json` records the exact environment used for the reported numbers.

All reported numbers come from one run on a single rented machine: NVIDIA RTX 4090 (24 GB), AMD Ryzen 9
5900X (24 vCPU), 64 GB RAM, NVIDIA driver 595.99, Ubuntu container (Vast.ai).

## 2. Reproduce everything

```bash
bash scripts/run_all.sh          # OPTIONAL=0 skips the optional 49-token Task 2 run
```

The reported run includes the optional 49-token Task 2 experiment (`results/task2/tokens49/`).

`run_all.sh` runs two lanes:

* **GPU lane.** All GPU jobs run strictly one at a time, so reported training times and peak memory
  are not distorted by other GPU work.
* **CPU lane.** CPU- and network-only preparation runs alongside it: Task 2 data selection and image
  download, and the Task 3 offline analyses.

Each stage leaves a marker in `outputs/markers/`. Re-running the script skips finished stages, and
training resumes from its last checkpoint. Logs go to `results/logs/`.

### Individual stages

```bash
# Task 1
python -m task1_finetune.data                         # 40,000 MetaMathQA examples (source order, filtered)
python -m task1_finetune.diagnostics --make-probes    # fixed 32 + 32 probe set (seed 6304)
python -m task1_finetune.train --method full_ft       # full_ft | l2sp | lora_r4 | lora_r16 | oplora_r16
python -m task1_finetune.export --condition full_ft   # + full_ft_mid; adapters merged into fresh backbones
python -m task1_finetune.evaluate --condition baseline
python -m task1_finetune.qualitative --condition baseline && python -m task1_finetune.qualitative --collect
python -m task1_finetune.diagnostics --run
python -m task1_finetune.analyze
# Task 2
python -m task2_vlm.download --annotations && python -m task2_vlm.data && python -m task2_vlm.download --images
python -m task2_vlm.features && python -m task2_vlm.train && python -m task2_vlm.evaluate && python -m task2_vlm.analyze
# Task 3
python -m task3_ttc.data && python -m task3_ttc.selection && python -m task3_ttc.signals && python -m task3_ttc.adaptive
python -m task3_ttc.live && python -m task3_ttc.analyze
```

## 3. Repository layout

```
configs/            task1.yaml, task2.yaml, task3.yaml: every hyperparameter used
common/             utilities, chat template helper, own LoRA/OPLoRA module, plotting/LaTeX helpers
task1_finetune/     data, train, export (merge + check), evaluate (lm-eval), qualitative, diagnostics, analyze
task2_vlm/          download, data (selection/pairs), features (SigLIP cache), model, train, evaluate, analyze
task3_ttc/          answers (normalisation/grading), data, core, selection, signals, adaptive, live, analyze
scripts/            setup_env.sh, run_all.sh
data/               saved IDs: task1 training indices + probe set, task2 selection IDs, task3 splits + orders
results/            machine-readable results (JSON / JSONL / CSV / parquet), logs
report/             generated LaTeX tables and PDF/PNG figures
```

Raw datasets, cached features, checkpoints and merged models are written to `outputs/` (or
`$PA3_OUTPUTS`). They are not committed.

## 4. Implementation choices

The assignment fixes most of the protocol. These are the decisions it leaves open, with the value used
here. All of them are in the configs.

**Task 1**
* Training format: Qwen3 chat template with `enable_thinking=False`. The prompt ends with the empty
  think block, and the loss covers the response plus `<|im_end|>`. GSM8K overlap is checked by
  normalised text (NFKC, lower case, alphanumerics only) on `query` and `original_question`.
* Logits are computed only at supervised positions. This is equivalent to full-sequence CE with
  masking, but uses less memory.
* L2-SP: the exact penalty gradient λ(θ−θ₀) is added to the unscaled gradient before clipping. This is
  identical to back-propagating (λ/2)‖θ−θ₀‖² and avoids an FP32 autograd graph over all parameters.
  The penalty is a sum over all unique parameters (tied embeddings counted once), and θ₀ is held in FP32
  on the GPU.
* LoRA and OPLoRA share one implementation (`common/lora.py`). A uses Kaiming-uniform initialisation
  (a=√5) and B is zero, as in PEFT. The same seed gives LoRA-16 and OPLoRA-16 bit-identical initial
  factors. OPLoRA bases are computed once per weight in FP32 (`torch.linalg.svd`), and the projected
  factors are formed on every forward pass and exported for inference.
* The cosine schedule (`get_cosine_schedule_with_warmup`) starts at LR 0 on the first update.
  Skipped FP16 updates still count toward the 1,250 updates and are logged.
* After a run is exported, the first- and third-quarter recovery checkpoints and all optimizer states
  are deleted to save disk. Their sizes are logged first.
* Reported GSM8K metric: stock `gsm8k_cot` flexible-extract exact match. An extra *diagnostic*
  column (number in the last `\boxed{}`, else the last number) separates answer-format effects from
  correctness. It is never used as the primary score.

**Task 2**
* Grounding evaluation pairs are split evenly between presence and counting (dev 12 + 12, test 32 + 32).
  Training pools hold 6,000 presence and 3,000 counting pairs, built only on training images (the
  caption images plus the VQAv2 training images).
* Label-noise filters: a presence positive needs an instance covering ≥ 1 % of the image. For
  counting, no crowd annotation is allowed and every counted instance must cover ≥ 0.25 % of the image.
* Instructions: "Write a short caption for this image." for captions, and the question followed by
  "Answer the question using a single word or phrase." for QA. The instruction text is identical in
  every image condition.
* Connector calibration uses training data only. A fixed scale is applied so that the mean token L2
  norm of the initial connector output equals the mean norm of the text-token embeddings
  (`results/task2/tokens196/calibration.json`).
* Optimiser: a fresh AdamW per stage (new parameter groups). One dynamic loss scaler is shared across
  stages. Weight decay excludes biases and 1-D parameters, as in Task 1.
* VQA answer normalisation and soft accuracy re-implement the official VQA evaluation rules.

**Task 3**
* MATH answers are grouped by math-verify equivalence, checked in both directions, without gold
  answers. MathVista answers are normalised with MathVista's rules (option letters for multiple
  choice; integer / float-with-precision / list). LaTeX wrappers and units are stripped before numeric
  conversion. Non-numeric outputs for numeric questions form their own (incorrect) groups.
* One MATH question (`test:4928`, evaluation split) has answers such as `3^{5050}` and power towers
  that make SymPy hang. Its 32 answers are grouped by normalised string equality instead; none of them
  matches the gold answer under either method. The question is listed in
  `results/task3/data_stats_math.json` (`string_fallback_questions`).
* The ten orders per question come from one `default_rng(6304)` stream per dataset, drawn in sorted
  question-ID order. They are saved in `data/task3/orders_*.npz`.
* Live generation uses Hugging Face `generate` (FP16, batch 1). Each call is seeded with
  6304 + 100·question_index + call_index.

## 5. Attribution

* **VQA evaluation:** answer normalisation follows the official code (GT-Vision-Lab/VQA, `vqaEval.py`),
  re-implemented in `task2_vlm/vqa_norm.py`.
* **MathVista:** answer normalisation follows `normalize_extracted_answer` in lupantech/MathVista,
  re-implemented in `task3_ttc/answers.py`.
* **Qualitative prompts:** `data/task1/first_plot_questions.yaml` is taken unchanged from the Emergent
  Misalignment repository (Betley et al., 2025).
* **Libraries:** benchmark scoring uses EleutherAI lm-evaluation-harness, and math equivalence uses
  Hugging Face math-verify.
* **LLM assistance:** code was written with assistance from an LLM coding assistant, as the assignment
  permits.
