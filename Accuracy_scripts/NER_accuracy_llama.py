# ==============================================================================
# Unified NER Accuracy Evaluation — Llama-3.2-3B-Instruct
# Includes embedded paper-quality plotting & Apple Silicon (MPS) optimizations.
# Only plots Loose Accuracy (Exact/Substring + Token Overlap F1 >= 0.5)
# ==============================================================================

import os
import json
import re
import unicodedata
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
    print(f"\nPlot successfully saved to: {base}.pdf")
    plt.show()
    plt.close()


# ── 2. CONFIGURATION ──────────────────────────────────────────────────────────
MODEL_ID        = "meta-llama/Llama-3.2-3B-Instruct"
ANNOTATION_FILE = "/Users/aadya/Documents/script_interp/comi_lingua_annotations.jsonl" # Update path if needed
OUTPUT_LOGS     = "ner_eval_logs_llama.jsonl"
FIGURES_DIR     = "Figures"
RESUME          = True   
MAX_SAMPLES     = 1000     

# ── 3. LOAD MODEL & TOKENIZER (Apple Silicon Optimized) ───────────────────────
print(f"Loading {MODEL_ID} tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
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


# ── 4. NER INFERENCE & PARSING FUNCTIONS ──────────────────────────────────────
def get_entity(text, script_hint):
    hints = {
        "roman_hindi": (
            "The text is Hindi written in Roman script. "
            "Extract the single most prominent named entity. "
            "Return ONLY the entity name as it appears in the text. "
            "No explanation, no punctuation."
        ),
        "devanagari": (
            "The text is Hindi in Devanagari script. "
            "Extract the single most prominent named entity. "
            "Return ONLY the entity name in Devanagari, exactly as it appears. "
            "No explanation, no punctuation."
        ),
        "mixed": (
            "The text is code-mixed Hindi (Devanagari + Roman). "
            "Extract the single most prominent named entity. "
            "Return ONLY the entity name exactly as it appears in the text. "
            "No explanation, no punctuation."
        ),
    }

    messages = [
        {"role": "system", "content": hints[script_hint]},
        {"role": "user", "content": f"Text: {text}"}
    ]

    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text_input], return_tensors="pt").to(device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=terminators
        )

    raw = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    return parse_output(raw)

def parse_output(raw):
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    text = raw.split('\n')[0].strip()

    for pat in [
        r'(?i)^(the\s+)?(main\s+)?(named\s+)?entity\s*(name)?\s*[:is]+\s*',
        r'(?i)^(the\s+)?answer\s+is\s*:?\s*',
        r'(?i)^entity\s*:\s*',
    ]:
        text = re.sub(pat, '', text).strip()

    text = re.sub(r'\*+', '', text)
    text = text.strip('\'"''""').strip()
    text = re.split(r'[,;]', text)[0].strip()

    return text

# ── 5. SCORING LOGIC ──────────────────────────────────────────────────────────
def normalize(text):
    return unicodedata.normalize('NFC', text).lower().strip()

def score(pred, gold):
    if not pred or not gold: return 0
    p = normalize(pred)
    g = normalize(gold)

    if p == g: return 2
    if len(p) > 2 and p in g: return 2
    if len(g) > 2 and g in p: return 2

    p_tokens = set(p.split())
    g_tokens = set(g.split())
    overlap = p_tokens & g_tokens
    if overlap:
        precision = len(overlap) / len(p_tokens)
        recall    = len(overlap) / len(g_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 >= 0.5: return 1

    return 0

def accuracy(scores, min_score=1):
    valid = [s for s in scores if s >= 0]
    if not valid: return 0.0, 0
    return sum(1 for s in valid if s >= min_score) / len(valid) * 100, len(valid)

# ── 6. DATA LOADING & EXECUTION LOOP ──────────────────────────────────────────
rows = []
if not os.path.exists(ANNOTATION_FILE):
    raise FileNotFoundError(f"Cannot find input file: {ANNOTATION_FILE}")

with open(ANNOTATION_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            rows.append(json.loads(line.strip()))
        except:
            continue

if MAX_SAMPLES:
    rows = rows[:MAX_SAMPLES]
    print(f"Loaded {len(rows)} annotated rows (Limited).")
else:
    print(f"Loaded {len(rows)} annotated rows.")

completed_ids = set()
if RESUME and os.path.exists(OUTPUT_LOGS):
    with open(OUTPUT_LOGS, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                completed_ids.add(json.loads(line.strip())['id'])
            except:
                continue
    print(f"Resuming — {len(completed_ids)} rows already done.")

scripts = {
    "Roman":      ("rom_text", "gold_rom", "roman_hindi"),
    "Devanagari": ("dev_text", "gold_dev", "devanagari"),
    "Mixed":      ("mix_text", "gold_mix", "mixed"),
}

results = {s: [] for s in scripts}
skipped = 0

log_mode = 'a' if RESUME else 'w'
with open(OUTPUT_LOGS, log_mode, encoding='utf-8') as log_f:
    for row in tqdm(rows, desc="Evaluating NER"):
        row_id = row.get('id')

        if RESUME and row_id in completed_ids:
            continue

        log_entry = {"id": row_id}
        
        for script_name, (text_col, gold_col, hint) in scripts.items():
            text = row.get(text_col, '')
            gold = row.get(gold_col, '')

            if not text or not gold or gold == 'ERROR':
                log_entry[f"{script_name}_pred"]  = "SKIP"
                log_entry[f"{script_name}_gold"]  = gold
                log_entry[f"{script_name}_score"] = -1
                continue

            if script_name == "Roman":
                devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
                if devanagari_chars / max(len(text), 1) > 0.3:
                    log_entry[f"{script_name}_pred"]  = "SKIP_BAD_DATA"
                    log_entry[f"{script_name}_gold"]  = gold
                    log_entry[f"{script_name}_score"] = -1
                    skipped += 1
                    continue

            pred  = get_entity(text, hint)
            s     = score(pred, gold)

            results[script_name].append(s)
            log_entry[f"{script_name}_pred"]  = pred
            log_entry[f"{script_name}_gold"]  = gold
            log_entry[f"{script_name}_score"] = s

        log_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        log_f.flush()

print(f"\nDone. Skipped {skipped} rows with bad Roman data.")

# ── 7. RESULTS & VISUALIZATION ────────────────────────────────────────────────
print("\n" + "="*55)
print(f"NER Accuracy — {MODEL_ID}")
print("="*55)
print(f"{'Script':<14} {'Loose %':>9}  {'Strict %':>10}  {'N':>5}")
print("-"*55)

accuracy_dict_loose = {}

for s in ["Devanagari", "Roman", "Mixed"]:
    loose,  n = accuracy(results[s], min_score=1)
    strict, _ = accuracy(results[s], min_score=2)
    accuracy_dict_loose[s] = loose
    print(f"{s:<14} {loose:>8.1f}%  {strict:>9.1f}%  {n:>5}")
print("="*55)

print("\nGenerating paper-quality plot (Loose Accuracy Only)...")
apply_global_style()
plot_accuracy_bars(
    accuracy    = accuracy_dict_loose,
    model_name  = "Llama-3.2-3B",
    task_name   = "NER",
    save_dir    = FIGURES_DIR,
    ylabel      = "Accuracy (%)"
)