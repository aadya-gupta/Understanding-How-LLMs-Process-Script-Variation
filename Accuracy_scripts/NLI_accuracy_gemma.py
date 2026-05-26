"""
Paper-Quality Accuracy Bar Charts
===================================
One chart per (model × task) combination.
X-axis: three scripts (Devanagari, Roman, Mixed)
Y-axis: Accuracy (%)
Legend: bottom-centre, bordered, white background

HOW TO USE IN YOUR EVAL SCRIPTS
---------------------------------
At the top of any eval script (NER, POS, NLI), add:

    from plot_accuracy import apply_global_style, plot_accuracy_bars
    apply_global_style()

Then after you have computed your accuracy dict, call:

    # accuracy dict format:
    # {"Devanagari": 70.9, "Roman": 75.4, "Mixed": 52.6}

    plot_accuracy_bars(
        accuracy    = accuracy,          # your computed dict
        model_name  = "Qwen3-4B",
        task_name   = "NER",
        save_dir    = "/content/drive/MyDrive/Figures",
    )

That's it. The function handles everything else.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── SHARED CONSTANTS ──────────────────────────────────────────────────────────
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

# ── GLOBAL STYLE ──────────────────────────────────────────────────────────────
def apply_global_style():
    """Call once per notebook session before generating any plots."""
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PLOTTING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_bars(
    accuracy:   dict,
    model_name: str,
    task_name:  str,
    save_dir:   str,
    ylabel:     str  = "Accuracy (%)",
    figsize:    tuple = (8, 6),
):
    """
    Parameters
    ----------
    accuracy   : {"Devanagari": float, "Roman": float, "Mixed": float}
    model_name : e.g. "Qwen3-4B"
    task_name  : e.g. "NER"
    save_dir   : directory where PDF and PNG are saved
    ylabel     : y-axis label (override for token-level POS if needed)
    figsize    : figure size in inches

    Saves
    -----
    <save_dir>/<task>_<model>.pdf
    <save_dir>/<task>_<model>.png
    """
    os.makedirs(save_dir, exist_ok=True)

    values = [accuracy.get(s, 0.0) for s in SCRIPT_ORDER]
    colors = [SCRIPT_COLORS[s]     for s in SCRIPT_ORDER]
    x      = np.arange(len(SCRIPT_ORDER))

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        x,
        values,
        width=BAR_WIDTH,
        color=colors,
        alpha=BAR_ALPHA,
        zorder=3,
        edgecolor="white",
        linewidth=0.6,
    )
    '''
   # Value labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE - 6,
            fontweight="bold",
            color="#111111",
        )
    '''
    # ── X-axis: script names, tilted 20° ──────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(
        SCRIPT_ORDER,
        #rotation=20,
        ha="center",
        #rotation_mode="anchor",
    )

    # ── Y-axis ────────────────────────────────────────────────────────────────
    ax.set_ylim(0, 108)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(f"{model_name}", pad=12)

    for spine in ax.spines.values():
      spine.set_visible(True)
      spine.set_color("#888888")
      spine.set_linewidth(0.8)

    '''# ── Legend: bottom-centre, bordered, white background ─────────────────────
    legend_patches = [
        mpatches.Patch(
            facecolor=SCRIPT_COLORS[s],
            alpha=BAR_ALPHA,
            label=s,
            edgecolor="#888888",
            linewidth=0.5,
        )
        for s in SCRIPT_ORDER
    ]
    leg = ax.legend(
        handles       = legend_patches,
        loc           = "lower center",
        bbox_to_anchor= (0.5, -0.48),   # pushed further below x-axis labels
        ncol          = 3,
        frameon       = True,
        framealpha    = 0.85,
        edgecolor     = "#888888",
        fancybox      = False,
        borderpad     = 0.8,
        handlelength  = 2.0,
        handleheight  = 1.2,
    )
    leg.get_frame().set_linewidth(0.8)'''

    # ── Save ──────────────────────────────────────────────────────────────────
    safe_model = model_name.replace("/", "-").replace(" ", "_")
    safe_task  = task_name.replace(" ", "_")
    base       = os.path.join(save_dir, f"{safe_task}_{safe_model}")

    # Extra bottom margin so the legend is fully inside the saved figure
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(base + ".pdf", format="pdf", bbox_inches="tight")
    plt.savefig(base + ".png", format="png", bbox_inches="tight")
    print(f"  Saved: {base}.pdf")
    plt.show()
    plt.close()


import json, csv, os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── applies global style ───────────────
apply_global_style()

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE   = #enter path
OUTPUT_CSV   = #enter path
FIGURES_DIR  = #enter path
MODEL_NAME   = "google/gemma-2-2b-it" # Updated for Gemma-2
NUM_SAMPLES  = 1500
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}

SCRIPTS = {
    "original": ("premise_orig",   "hypothesis_orig"),
    "roman":    ("premise_roman",  "hypothesis_roman"),
    "mixed":    ("premise_mixed",  "hypothesis_mixed"),
}

# ── PROMPT ────────────────────────────────────────────────────────────────────
def make_prompt(premise, hypothesis):
    return (
        "Classify the relationship between the premise and hypothesis.\n"
        "Reply with exactly one word — either: entailment, neutral, or contradiction.\n"
        "No explanations. No punctuation. Just the one word.\n\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Answer:"
    )

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("Loading model...")
# 1. Define the Mac GPU explicitly
device = torch.device("mps")

# 2. Load with float16 and force EAGER attention
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    attn_implementation="eager",  # <--- THIS IS THE MAGIC FIX FOR MPS
    trust_remote_code=True
)

# 3. Manually push it to the Mac GPU
model = model.to(device)
model.eval()
print(f"Model ready on device: {model.device}\n")

# ── INFERENCE ─────────────────────────────────────────────────────────────────
@torch.inference_mode()
def predict(premise, hypothesis):
    prompt = make_prompt(premise, hypothesis)
    messages = [{"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
    
    for word in ["entailment", "neutral", "contradiction"]:
        if word in raw:
            return word, raw
    return None, raw

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
samples = []
with open(INPUT_FILE, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

if NUM_SAMPLES is not None:
    samples = samples[:NUM_SAMPLES]

print(f"Evaluating {len(samples)} samples across 3 scripts...\n")

# ── EVALUATE ──────────────────────────────────────────────────────────────────
correct = {"original": 0, "roman": 0, "mixed": 0}
rows = []

for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
    gold_int = sample["label"]
    gold_str = LABEL_MAP[gold_int]

    for script, (prem_key, hyp_key) in SCRIPTS.items():
        pred, raw = predict(sample[prem_key], sample[hyp_key])
        score = 1 if pred == gold_str else 0
        if score:
            correct[script] += 1

        rows.append({
            "sample_id": sample.get("id_index", i),
            "script":    script,
            "gold":      gold_str,
            "predicted": pred if pred else f"FAIL({raw})",
            "score":     score,
        })

# ── ACCURACY ──────────────────────────────────────────────────────────────────
n = len(samples)
accuracy = {s: correct[s] / n * 100 for s in SCRIPTS}

print("\n── RESULTS ──────────────────────────")
for s, acc in accuracy.items():
    print(f"  {s:>10}: {acc:.1f}%  ({correct[s]}/{n})")

# ── SAVE CSV ──────────────────────────────────────────────────────────────────
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["sample_id", "script", "gold", "predicted", "score"])
    writer.writeheader()
    writer.writerows(rows)
print(f"\nResults saved to: {OUTPUT_CSV}")

# ── PLOT ──────────────────────────────────────────────────────────────────────
acc_display = {
    "Devanagari": accuracy["original"],
    "Roman":      accuracy["roman"],
    "Mixed":      accuracy["mixed"],
}

plot_accuracy_bars(
    accuracy   = acc_display,
    model_name = "Gemma-2-2B", # Updated for plot title/save logic
    task_name  = "NLI",
    save_dir   = FIGURES_DIR,
)
