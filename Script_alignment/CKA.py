# =============================================================================
# Unified Layer-wise CKA Extraction & ARR-Ready Plotting
# Models: Llama-3.2-3B, Qwen3-4B, Gemma-2-2B | Tasks: NLI, POS 
# Environment: Google Colab
# =============================================================================

import os
import json
import math
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from google.colab import drive

drive.mount('/content/drive')

# ══════════════════════════════════════════════════════════════════════════════
# 1. ARR-READY PLOTTING CONFIGURATION & FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
TITLE_SIZE  = 36
LABEL_SIZE  = 32
TICK_SIZE   = 20
LEGEND_SIZE = 20

PAIR_STYLES = {
    "Roman–Devanagari":  {"color": "#1F77B4", "marker": "o", "label": "Roman–Devanagari"},
    "Roman–Mixed":       {"color": "#D62728", "marker": "s", "label": "Roman–Mixed"},
    "Devanagari–Mixed":  {"color": "#2CA02C", "marker": "^", "label": "Devanagari–Mixed"},
}

def apply_global_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.weight": "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal"
    })

def _format_line_axes(ax, title, xlabel, ylabel, ymin, ymax, y_interval):
    ax.set_title(title, pad=16, fontsize=TITLE_SIZE, fontweight="normal")
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontweight="normal")
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontweight="normal")
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax.set_ylim(ymin, ymax)
    if y_interval is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_interval))

    ax.grid(True, which='major', axis='both', linestyle='--', color='#D3D3D3', alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", direction="out", color="black", width=1.2, length=6)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

def plot_line(data, plot_type, model_name, task_name, save_dir, figsize=(11, 8)):
    os.makedirs(save_dir, exist_ok=True)
    apply_global_style()

    fig, ax = plt.subplots(figsize=figsize)
    handles = []

    for series_name, values in data.items():
        style  = PAIR_STYLES[series_name]
        layers = np.arange(len(values))
        ax.plot(
            layers, values, color=style["color"], marker=style["marker"],
            linewidth=2.5, markersize=8, markevery=max(1, len(values) // 10),
            zorder=3, label=style["label"],
        )
        handles.append(mpatches.Patch(
            facecolor=style["color"], label=style["label"], edgecolor='black', linewidth=0.8,
        ))

    _format_line_axes(
        ax, title=model_name, xlabel="Transformer Layer", ylabel="CKA Score",
        ymin=0.0, ymax=1.0, y_interval=0.1
    )

    safe_model = model_name.replace("/", "-").replace(" ", "_")
    base = os.path.join(save_dir, f"{plot_type}_{task_name}_{safe_model}")

    plt.subplots_adjust(bottom=0.15)
    plt.savefig(base + ".pdf", format="pdf", bbox_inches="tight")
    plt.savefig(base + ".png", format="png", bbox_inches="tight")
    print(f"  Plot successfully saved to: {base}.pdf")
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 2. CKA MATH
# ══════════════════════════════════════════════════════════════════════════════
def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    K = X @ X.T
    L = Y @ Y.T
    hsic  = (K * L).sum()
    var_K = (K * K).sum().sqrt()
    var_L = (L * L).sum().sqrt()
    if var_K == 0 or var_L == 0:
        return 0.0
    return (hsic / (var_K * var_L)).item()


# ══════════════════════════════════════════════════════════════════════════════
# 3. TASK & DATA CONFIGURATION UTILS
# ══════════════════════════════════════════════════════════════════════════════
def get_task_config(task):
    if task == "NLI":
        return {
            "keys": {
                "Roman": ("premise_roman", "hypothesis_roman"),
                "Devanagari": ("premise_orig", "hypothesis_orig"),
                "Mixed": ("premise_mixed", "hypothesis_mixed"),
            },
            "system_hint": (
                "Classify the relationship between the premise and hypothesis.\n"
                "Reply with exactly one word — either: entailment, neutral, or contradiction.\n"
                "No explanations. No punctuation. Just the one word."
            )
        }
    elif task == "POS":
        return {
            "keys": {
                "Roman": ("tokens_roman",),
                "Devanagari": ("tokens_orig",),
                "Mixed": ("tokens_mixed",),
            },
            "system_hint": (
                "You are a linguistic annotator. Assign a Universal Dependencies POS tag to each word in the provided list. "
                "Valid tags: NOUN, VERB, ADJ, ADV, PRON, DET, ADP, CCONJ, SCONJ, NUM, PART, PUNCT, INTJ, SYM, PROPN, AUX, X. "
                "Return ONLY a comma-separated list of tags in the exact same order. No explanations, no markdown."
            )
        }
    raise ValueError(f"Unknown task: {task}")

def is_valid_row(row, task, config):
    for script, keys in config["keys"].items():
        for key in keys:
            if not row.get(key):
                return False
    return True

def make_prompt(tokenizer, model_id, task, row, script, config):
    system_hint = config["system_hint"]
    
    if task == "NLI":
        p_key, h_key = config["keys"][script]
        user_content = f"Premise: {row[p_key]}\nHypothesis: {row[h_key]}"
    elif task == "POS":
        t_key = config["keys"][script][0]
        words_str = json.dumps(row[t_key], ensure_ascii=False)
        user_content = f"Words: {words_str}"

    if "gemma" in model_id.lower():
        messages = [{"role": "user", "content": f"{system_hint}\n\n{user_content}"}]
    else:
        messages = [
            {"role": "system", "content": system_hint},
            {"role": "user", "content": user_content}
        ]
        
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return tokenizer.apply_chat_template(messages, tokenize=False)


# ══════════════════════════════════════════════════════════════════════════════
# 4. EXPERIMENT PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_cka_experiment(model_id, task, data_file, cache_file, figures_dir, target_samples=300, batch_size=8, max_seq_len=512):
    print(f"\n{'='*80}")
    print(f"Starting CKA Experiment: Model={model_id} | Task={task}")
    print(f"{'='*80}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    task_config = get_task_config(task)

    # --- Load Data ---
    rows = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    valid_rows = [r for r in rows if is_valid_row(r, task, task_config)]
    print(f"Loaded {len(valid_rows)} valid samples for {task}.")

    # --- Load Model ---
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, output_hidden_states=True,
        device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True,
    ).eval()

    dummy_inputs = tokenizer(["dummy"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        dummy_outputs = model(**dummy_inputs)
    num_states = len(dummy_outputs.hidden_states)
    hidden_dim = dummy_outputs.hidden_states[0].shape[-1]
    del dummy_outputs, dummy_inputs
    torch.cuda.empty_cache()
    print(f"Detected {num_states} hidden states. Hidden dim: {hidden_dim}")

    # --- Caching & State Management ---
    reps_rom, reps_dev, reps_mix = None, None, None
    successful_samples = 0

    if os.path.exists(cache_file):
        try:
            cache = torch.load(cache_file, map_location="cpu", weights_only=False)
            if cache["reps_rom"].shape[0] == target_samples:
                print("\nResuming from valid cache...")
                reps_rom = cache["reps_rom"]
                reps_dev = cache["reps_dev"]
                reps_mix = cache["reps_mix"]
                successful_samples = cache["successful_samples"]
                print(f"  Resuming from {successful_samples} samples.")
            else:
                print(f"\nWARNING: Old cache size ({cache['reps_rom'].shape[0]}) does not match TARGET_SAMPLES ({target_samples}). Starting fresh.")
        except Exception as e:
            print(f"Error loading cache: {e}. Starting fresh.")

    if reps_rom is None:
        reps_rom = torch.zeros(target_samples, num_states, hidden_dim, dtype=torch.float16)
        reps_dev = torch.zeros(target_samples, num_states, hidden_dim, dtype=torch.float16)
        reps_mix = torch.zeros(target_samples, num_states, hidden_dim, dtype=torch.float16)

    # --- Extraction Loop ---
    @torch.no_grad()
    def extract_batch(prompts):
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len).to(model.device)
        out = model(**enc)
        result = torch.stack([layer[:, -1, :] for layer in out.hidden_states], dim=1).float().cpu().half()
        del out, enc
        return result

    remaining = valid_rows[successful_samples : successful_samples + target_samples]
    if remaining:
        print(f"\nProcessing up to {target_samples} samples (have {successful_samples} already)...")

    for b0 in tqdm(range(0, len(remaining), batch_size), desc="Batches"):
        batch = remaining[b0 : b0 + batch_size]
        bs = len(batch)

        rom_prompts = [make_prompt(tokenizer, model_id, task, r, "Roman", task_config) for r in batch]
        dev_prompts = [make_prompt(tokenizer, model_id, task, r, "Devanagari", task_config) for r in batch]
        mix_prompts = [make_prompt(tokenizer, model_id, task, r, "Mixed", task_config) for r in batch]

        h_rom = extract_batch(rom_prompts)
        h_dev = extract_batch(dev_prompts)
        h_mix = extract_batch(mix_prompts)

        slot = successful_samples
        reps_rom[slot : slot + bs] = h_rom
        reps_dev[slot : slot + bs] = h_dev
        reps_mix[slot : slot + bs] = h_mix
        successful_samples += bs

        if successful_samples % 20 < batch_size or successful_samples >= target_samples:
            torch.save({
                "reps_rom": reps_rom, "reps_dev": reps_dev, "reps_mix": reps_mix,
                "successful_samples": successful_samples,
            }, cache_file)

    # --- Compute CKA ---
    print("\nComputing CKA scores...")
    cka_rom_dev, cka_rom_mix, cka_dev_mix = [], [], []

    for layer_idx in tqdm(range(num_states), desc="Layers"):
        X_rom = reps_rom[:successful_samples, layer_idx, :].float().to(device)
        X_dev = reps_dev[:successful_samples, layer_idx, :].float().to(device)
        X_mix = reps_mix[:successful_samples, layer_idx, :].float().to(device)

        cka_rom_dev.append(linear_cka(X_rom, X_dev))
        cka_rom_mix.append(linear_cka(X_rom, X_mix))
        cka_dev_mix.append(linear_cka(X_dev, X_mix))

        del X_rom, X_dev, X_mix
        torch.cuda.empty_cache()

    # --- Plotting ---
    print("\nGenerating paper-quality plot...")
    cka_plot_data = {
        "Roman–Devanagari": cka_rom_dev,
        "Roman–Mixed":      cka_rom_mix,
        "Devanagari–Mixed": cka_dev_mix,
    }

    short_model_name = model_id.split("/")[-1]
    plot_line(
        data=cka_plot_data, plot_type="cka", model_name=short_model_name, 
        task_name=task, save_dir=figures_dir
    )

    # --- Clean up Memory ---
    del model, tokenizer, reps_rom, reps_dev, reps_mix
    gc.collect()
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    FIGURES_DIR = "/content/drive/MyDrive/Figures"
    
    # Define all models and tasks you want to run here
    EXPERIMENTS = [
        {
            "model_id": "meta-llama/Llama-3.2-3B-Instruct",
            "task": "NLI",
            "data_file": "/content/drive/MyDrive/indic_mixed_dataset.jsonl",
            "cache_file": "/content/drive/MyDrive/llama3_nli_cka_cache.pt"
        },
        {
            "model_id": "Qwen/Qwen3-4B",
            "task": "NLI",
            "data_file": "/content/drive/MyDrive/indic_mixed_dataset.jsonl",
            "cache_file": "/content/drive/MyDrive/qwen_nli_cka_cache.pt"
        },
        {
            "model_id": "google/gemma-2-2b-it",
            "task": "NLI",
            "data_file": "/content/drive/MyDrive/indic_mixed_dataset.jsonl",
            "cache_file": "/content/drive/MyDrive/gemma_nli_cka_cache.pt"
        },
        {
            "model_id": "google/gemma-2-2b-it",
            "task": "POS",
            "data_file": "/content/drive/MyDrive/hindi_pos_dataset_clean.jsonl",
            "cache_file": "/content/drive/MyDrive/gemma_pos_cka_cache.pt"
        }
    ]

    for exp in EXPERIMENTS:
        run_cka_experiment(
            model_id=exp["model_id"],
            task=exp["task"],
            data_file=exp["data_file"],
            cache_file=exp["cache_file"],
            figures_dir=FIGURES_DIR,
            target_samples=300,
            batch_size=8,
            max_seq_len=512
        )