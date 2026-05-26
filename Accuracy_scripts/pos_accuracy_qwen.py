# ==============================================================================
# Unified Batched POS Tagging Evaluation — Qwen3-4B
# Includes embedded paper-quality plotting & Apple Silicon (MPS) optimizations.
# ==============================================================================

import os
import json
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ── 1. EMBEDDED PLOTTING LOGIC ────────────────────────────────────────────────
FONT_SIZE   = 20
FONT_FAMILY = "Times New Roman"

SCRIPT_ORDER  = ["Devanagari", "Roman", "Mixed"]
SCRIPT_COLORS = {
    "Devanagari": "#C63232",
    "Roman":      "#3571A7",
    "Mixed":      "#45943B",
}
BAR_WIDTH = 0.38
BAR_ALPHA  = 0.88

def apply_global_style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":          FONT_SIZE,
        "axes.titlesize":     FONT_SIZE,
        "axes.labelsize":     FONT_SIZE,
        "xtick.labelsize":    FONT_SIZE,
        "ytick.labelsize":    FONT_SIZE,
        "legend.fontsize":    FONT_SIZE - 4,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })

def plot_accuracy_bars(accuracy: dict, model_name: str, task_name: str, save_dir: str, ylabel: str = "Accuracy (%)", figsize: tuple = (8, 6)):
    os.makedirs(save_dir, exist_ok=True)
    values = [accuracy.get(s, 0.0) for s in SCRIPT_ORDER]
    colors = [SCRIPT_COLORS[s] for s in SCRIPT_ORDER]
    x = np.arange(len(SCRIPT_ORDER))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(x, values, width=BAR_WIDTH, color=colors, alpha=BAR_ALPHA, zorder=3, edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(SCRIPT_ORDER, ha="center")
    ax.set_ylim(0, 108)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f"{model_name}", pad=12)

    for spine in ax.spines.values():
      spine.set_visible(True)
      spine.set_color("#888888")
      spine.set_linewidth(0.8)

    safe_model = model_name.replace("/", "-").replace(" ", "_")
    safe_task  = task_name.replace(" ", "_")
    base       = os.path.join(save_dir, f"{safe_task}_{safe_model}")

    plt.subplots_adjust(bottom=0.25)
    plt.savefig(base + ".pdf", format="pdf", bbox_inches="tight")
    plt.savefig(base + ".png", format="png", bbox_inches="tight")
    print(f"\n✅ Plot successfully saved to: {base}.pdf")
    plt.show()
    plt.close()

# ── 2. CONFIGURATION ──────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B"
INPUT_FILE  = "hindi_pos_dataset_clean.jsonl" # Update to your exact Mac path
OUTPUT_LOGS = "pos_eval_logs_qwen.jsonl"
FIGURES_DIR = "Figures"
BATCH_SIZE  = 8   
MAX_SAMPLES = 250

# ── 3. LOAD MODEL & TOKENIZER (Apple Silicon Optimized) ───────────────────────
print(f"Loading {MODEL_ID} tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Decoder-only models MUST use left-padding for batched generation
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model to Mac RAM...")
device = torch.device("mps")

# Explicitly set EAGER attention and float16 to bypass the MPS C++ crash
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
    trust_remote_code=True
)

print("Pushing model to Apple MPS GPU...")
model = model.to(device)
model.eval()
print(f"Model ready on device: {model.device}\n")

# ── 4. BATCHED INFERENCE FUNCTION ─────────────────────────────────────────────
def get_predicted_tags_batch(token_lists, script_hint):
    system_prompt = (
        f"You are a linguistic annotator specializing in {script_hint}. "
        "Assign a Universal Dependencies POS tag to each word in the provided list. "
        "Valid tags: NOUN, VERB, ADJ, ADV, PRON, DET, ADP, CCONJ, SCONJ, NUM, PART, PUNCT, INTJ, SYM, PROPN, AUX, X. "
        "Return ONLY a comma-separated list of tags in the exact same order. No explanations, no markdown."
    )

    prompts = []
    for tokens in token_lists:
        words_str = json.dumps(tokens, ensure_ascii=False)
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Words: {words_str}"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        )

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    batch_results = []
    prompt_len = inputs.input_ids.shape[1]
    for i in range(len(token_lists)):
        raw = tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True).strip()
        tags = [re.sub(r'[^A-Z]', '', t.strip().upper()) for t in raw.split(',')]
        batch_results.append([t for t in tags if t])

    return batch_results

# ── 5. SCORING LOGIC ──────────────────────────────────────────────────────────
def score_pos_sequence(pred_tags, gold_tags):
    if not gold_tags:
        return 0, 0
    correct = sum(1 for p, g in zip(pred_tags, gold_tags) if p == g.strip().upper())
    return correct, len(gold_tags)

# ── 6. MAIN EXECUTION LOOP ────────────────────────────────────────────────────
scripts_map = {
    "Devanagari": {"tokens_key": "tokens_orig",  "hint": "Hindi in Devanagari script"},
    "Roman":      {"tokens_key": "tokens_roman", "hint": "Hindi in Roman script (ITRANS)"},
    "Mixed":      {"tokens_key": "tokens_mixed", "hint": "code-mixed Hindi (Devanagari and Roman)"}
}

totals = {s: {"correct": 0, "total": 0} for s in scripts_map}

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Cannot find input file: {INPUT_FILE}")

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    rows = [json.loads(line.strip()) for line in f]
if MAX_SAMPLES:
    rows = rows[:MAX_SAMPLES]

print(f"Processing {len(rows)} rows in batches of {BATCH_SIZE}...")

with open(OUTPUT_LOGS, 'w', encoding='utf-8') as log_f:
    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Batches"):
        batch_rows = rows[i:i + BATCH_SIZE]
        batch_log_entries = {row['id_index']: {"id_index": row['id_index']} for row in batch_rows}

        for s_name, config in scripts_map.items():
            batch_tokens = [r.get(config["tokens_key"], []) for r in batch_rows]
            golds = [r.get("gold_tags_orig", []) for r in batch_rows]

            preds = get_predicted_tags_batch(batch_tokens, config["hint"])

            for j, r in enumerate(batch_rows):
                rid = r['id_index']
                correct, total = score_pos_sequence(preds[j], golds[j])
                totals[s_name]["correct"] += correct
                totals[s_name]["total"] += total

                batch_log_entries[rid][f"{s_name}_pred"] = preds[j]
                batch_log_entries[rid][f"{s_name}_gold"] = golds[j]
                batch_log_entries[rid][f"{s_name}_acc"]  = round((correct/total)*100, 1) if total > 0 else 0

        for entry in batch_log_entries.values():
            log_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log_f.flush()

# ── 7. RESULTS & VISUALIZATION ────────────────────────────────────────────────
print("\n" + "="*55)
print(f"POS Tagging Accuracy - {MODEL_ID}")
print("="*55)

accuracy_dict = {}
for s in scripts_map.keys():
    c = totals[s]["correct"]
    t = totals[s]["total"]
    acc = (c / t * 100) if t > 0 else 0.0
    accuracy_dict[s] = acc
    print(f"{s:<14} {acc:>11.1f}%  ({c} / {t} tokens)")

print("\nGenerating paper-quality plot...")
apply_global_style()
plot_accuracy_bars(
    accuracy    = accuracy_dict,
    model_name  = "Qwen3-4B",
    task_name   = "POS",
    save_dir    = FIGURES_DIR,
    ylabel      = "Accuracy (%)"
)