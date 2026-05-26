# ==============================================================================
# Unified Script Identity Probe (Simple Train/Test Split) — ARR-Ready
# Models: Llama-3.2-3B, Qwen3-4B, Gemma-2-2B | Tasks: NLI, POS 
# ==============================================================================

import os, json, gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from google.colab import drive
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION & SETUP
# ══════════════════════════════════════════════════════════════════════════════
FIGURES_DIR = "/content/drive/MyDrive/Figures"
MAX_ROWS    = 1000      # 1000 per script class → perfectly balanced
SAVE_EVERY  = 50

# Mount drive once
drive.mount('/content/drive')
os.makedirs(FIGURES_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print(f"Device: {device}")

SCRIPT_NAMES = ["Devanagari", "Roman", "Mixed"]

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING (ARR-Ready Style)
# ══════════════════════════════════════════════════════════════════════════════
TITLE_SIZE, LABEL_SIZE, TICK_SIZE = 36, 32, 20
PAIR_STYLES = {
    "Devanagari": {"color": "#D62728", "marker": "o", "label": "Devanagari"},
    "Roman":      {"color": "#1F77B4", "marker": "s", "label": "Roman"},
    "Mixed":      {"color": "#2CA02C", "marker": "^", "label": "Mixed"},
}

def apply_global_style():
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset":  "stix",
        "font.weight":       "normal",
        "axes.titleweight":  "normal",
        "axes.labelweight":  "normal",
    })

def plot_line(data, model_name, task_name, save_dir):
    apply_global_style()
    fig, ax = plt.subplots(figsize=(11, 8))

    for series_name, values in data.items():
        style = PAIR_STYLES[series_name]
        ax.plot(
            np.arange(len(values)), values,
            color=style["color"], marker=style["marker"],
            linewidth=2.5, markersize=8,
            markevery=max(1, len(values) // 10), zorder=3, label=style["label"]
        )

    # Chance level reference line
    ax.axhline(33.33, color="#888888", linewidth=1.5, linestyle=":", zorder=2, label="Chance (33%)")

    # Formatting axes
    ax.set_title(f"Script Identity Probe: {model_name}", pad=16, fontsize=TITLE_SIZE)
    ax.set_xlabel("Transformer Layer", fontsize=LABEL_SIZE)
    ax.set_ylabel("Accuracy (%)", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(True, which="major", axis="both", linestyle="--", color="#D3D3D3", alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", direction="out", color="black", width=1.2, length=6)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    safe_model = model_name.replace("/", "-").replace(" ", "_")
    base_path = os.path.join(save_dir, f"script_probe_{task_name}_{safe_model}")
    
    plt.savefig(base_path + ".pdf", format="pdf", bbox_inches="tight")
    print(f"  Saved plot to {base_path}.pdf")
    plt.show()
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# DATA & PROMPT HANDLING 
# ══════════════════════════════════════════════════════════════════════════════
def get_task_config(task):
    if task == "NLI":
        return {
            "keys": {
                "Devanagari": ("premise_orig", "hypothesis_orig"),
                "Roman":      ("premise_roman", "hypothesis_roman"),
                "Mixed":      ("premise_mixed", "hypothesis_mixed"),
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
                "Devanagari": ("tokens_orig", "gold_tags_orig"),
                "Roman":      ("tokens_roman", "gold_tags_orig"),
                "Mixed":      ("tokens_mixed", "gold_tags_orig"),
            },
            "system_hint": (
                "You are a linguistic annotator. Assign a Universal Dependencies POS tag to each word in the provided list. "
                "Valid tags: NOUN, VERB, ADJ, ADV, PRON, DET, ADP, CCONJ, SCONJ, NUM, PART, PUNCT, INTJ, SYM, PROPN, AUX, X. "
                "Return ONLY a comma-separated list of tags in the exact same order. No explanations, no markdown."
            )
        }
    raise ValueError(f"Unknown task: {task}")

def is_valid(row, task, config):
    for s in SCRIPT_NAMES:
        if task == "NLI":
            p_key, h_key = config["keys"][s]
            p, h = row.get(p_key, ""), row.get(h_key, "")
            if not p or not h: return False
            text_str = p + " " + h
        else: # POS
            t_key, g_key = config["keys"][s]
            t, g = row.get(t_key, []), row.get(g_key, [])
            if not t or not g or g == "ERROR": return False
            text_str = " ".join(t)

        # Roman script pollution check
        if s == "Roman":
            dev_chars = sum(1 for c in text_str if '\u0900' <= c <= '\u097F')
            if dev_chars / max(len(text_str), 1) > 0.3: return False
    return True

def make_prompt(tokenizer, model_id, task, row, script, config):
    if task == "NLI":
        p_key, h_key = config["keys"][script]
        user_content = f"Premise: {row[p_key]}\nHypothesis: {row[h_key]}"
    else: # POS
        t_key, _ = config["keys"][script]
        words_str = json.dumps(row[t_key], ensure_ascii=False)
        user_content = f"Words: {words_str}"

    system_hint = config["system_hint"]

    # Gemma models generally merge system prompts into the user role
    if "gemma" in model_id.lower():
        messages = [{"role": "user", "content": f"{system_hint}\n\n{user_content}"}]
    else:
        messages = [
            {"role": "system", "content": system_hint},
            {"role": "user", "content": user_content}
        ]

    # Qwen/Llama apply correctly; wrap in try-except if specific kwargs throw errors
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return tokenizer.apply_chat_template(messages, tokenize=False)

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(model_id, task, data_path, cache_file):
    print(f"\n{'='*80}\nStarting Experiment: Model={model_id} | Task={task}\n{'='*80}")
    
    task_config = get_task_config(task)

    # 1. Load Data
    print("Loading and filtering data...")
    with open(data_path, encoding="utf-8") as f:
        valid_rows = [r for r in [json.loads(l) for l in f if l.strip()] if is_valid(r, task, task_config)][:MAX_ROWS]
    n = len(valid_rows)
    print(f"  {n} valid samples — {n*3} total observations across scripts.")

    # 2. Load Model
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Standardize on 4-bit config with double quant for optimal memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, output_hidden_states=True, 
        device_map="auto", trust_remote_code=True
    ).eval()

    with torch.no_grad():
        dummy = tokenizer(["dummy"], return_tensors="pt").to(device)
        dummy_out = model(**dummy)
        NUM_LAYERS = len(dummy_out.hidden_states)
        HIDDEN_DIM = dummy_out.hidden_states[0].shape[-1]
        del dummy, dummy_out
    torch.cuda.empty_cache()
    print(f"  {NUM_LAYERS} hidden states, hidden_dim={HIDDEN_DIM}")

    # 3. Extraction
    reps = {s: np.zeros((NUM_LAYERS, n, HIDDEN_DIM), dtype=np.float16) for s in SCRIPT_NAMES}
    start_idx = 0

    if os.path.exists(cache_file):
        cache = np.load(cache_file)
        start_idx = int(cache["start_idx"])
        for s in SCRIPT_NAMES: reps[s] = cache[s]
        print(f"Resuming from sample {start_idx}.")

    for sample_idx in tqdm(range(start_idx, n), desc="Extracting hidden states"):
        prompts = [make_prompt(tokenizer, model_id, task, valid_rows[sample_idx], s, task_config) for s in SCRIPT_NAMES]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

        with torch.inference_mode():
            hidden_states = model(**inputs).hidden_states

        for l_idx, l_hs in enumerate(hidden_states):
            last = l_hs[:, -1, :].float().cpu()
            for i, s in enumerate(SCRIPT_NAMES): 
                reps[s][l_idx, sample_idx] = last[i].numpy().astype(np.float16)

        del hidden_states, inputs
        torch.cuda.empty_cache()

        if (sample_idx + 1) % SAVE_EVERY == 0 or (sample_idx + 1) == n:
            np.savez(cache_file, start_idx=sample_idx + 1, **{s: reps[s] for s in SCRIPT_NAMES})

    # 4. Probing (80/20 Train/Test Split)
    print("\nRunning script identity probe per layer...")
    y_all = np.concatenate([np.zeros(n), np.ones(n), np.full(n, 2)])
    idx_train, idx_test = train_test_split(np.arange(len(y_all)), test_size=0.2, random_state=42, stratify=y_all)
    
    per_script_acc = {s: [] for s in SCRIPT_NAMES}
    overall_acc = []

    for layer_idx in tqdm(range(NUM_LAYERS), desc="Probing"):
        X_all = np.concatenate([reps[s][layer_idx] for s in SCRIPT_NAMES], axis=0).astype(np.float32)
        
        scaler = StandardScaler().fit(X_all[idx_train])
        X_train_scaled = scaler.transform(X_all[idx_train])
        X_test_scaled  = scaler.transform(X_all[idx_test])

        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", multi_class="multinomial", random_state=42)
        clf.fit(X_train_scaled, y_all[idx_train])

        preds = clf.predict(X_test_scaled)
        true = y_all[idx_test]

        overall_acc.append((preds == true).mean() * 100)

        for i, s in enumerate(SCRIPT_NAMES):
            mask = (true == i)
            per_script_acc[s].append((preds[mask] == true[mask]).mean() * 100 if mask.sum() > 0 else 0.0)

    # 5. Summary & Plotting
    print("\n── SCRIPT PROBE RESULTS ────────────────────────────────────")
    print(f"  Chance baseline:  33.3%")
    for s in SCRIPT_NAMES:
        peak, peak_layer, final = max(per_script_acc[s]), int(np.argmax(per_script_acc[s])), per_script_acc[s][-1]
        print(f"  {s:<14}  peak={peak:.1f}% (layer {peak_layer})  final={final:.1f}%")

    near_chance_layers = [l for l, a in enumerate(overall_acc) if a <= 38.33]
    if near_chance_layers:
        print(f"\n  Script invariance emergence: layer {near_chance_layers[0]}")
    else:
        print(f"\n  Script identity never fully lost (min overall = {min(overall_acc):.1f}%)")

    short_model_name = model_id.split("/")[-1]
    plot_line(per_script_acc, short_model_name, task, FIGURES_DIR)

    # Free memory before next experiment
    del model, tokenizer, reps, X_all
    gc.collect()
    torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    EXPERIMENTS = [
        {
            "model_id": "meta-llama/Llama-3.2-3B-Instruct",
            "task": "NLI",
            "data_path": "/content/drive/MyDrive/indic_mixed_dataset.jsonl",
            "cache_file": "/content/drive/MyDrive/script_probe_nli_llama3.npz"
        },
        {
            "model_id": "Qwen/Qwen3-4B",
            "task": "NLI",
            "data_path": "/content/drive/MyDrive/indic_mixed_dataset.jsonl",
            "cache_file": "/content/drive/MyDrive/script_probe_nli_qwen3.npz"
        },
        {
            "model_id": "google/gemma-2-2b-it",
            "task": "NLI",
            "data_path": "/content/drive/MyDrive/indic_mixed_dataset.jsonl",
            "cache_file": "/content/drive/MyDrive/script_probe_nli_gemma.npz"
        },
        {
            "model_id": "google/gemma-2-2b-it",
            "task": "POS",
            "data_path": "/content/drive/MyDrive/hindi_pos_dataset_clean.jsonl",
            "cache_file": "/content/drive/MyDrive/script_probe_pos_gemma.npz"
        }
    ]

    for exp in EXPERIMENTS:
        run_experiment(
            model_id=exp["model_id"],
            task=exp["task"],
            data_path=exp["data_path"],
            cache_file=exp["cache_file"]
        )