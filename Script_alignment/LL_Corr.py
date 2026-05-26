# =============================================================================
# Unified Logit Lens & Pearson Correlation Extraction (ARR-Ready)
# Models: Llama-3.2-3B, Gemma-2-2B, Qwen3-4B | Tasks: NER, POS, NLI
# Environment: Google Colab
# =============================================================================

import os
import json
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
# 1. ARR-READY LINE PLOTTING CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
TITLE_SIZE  = 36
LABEL_SIZE  = 32
TICK_SIZE   = 20
LEGEND_SIZE = 20

PAIR_STYLES = {
    "Devanagari": {"color": "#D62728", "marker": "o", "label": "Devanagari"},
    "Roman":      {"color": "#1F77B4", "marker": "s", "label": "Roman"},
    "Mixed":      {"color": "#2CA02C", "marker": "^", "label": "Mixed"},
    "Roman–Devanagari":  {"color": "#1F77B4", "marker": "o", "label": "Roman–Devanagari"},
    "Roman–Mixed":       {"color": "#D62728", "marker": "s", "label": "Roman–Mixed"},
    "Devanagari–Mixed":  {"color": "#2CA02C", "marker": "^", "label": "Devanagari–Mixed"},
}

YLABELS = {
    "logit_lens":  "Accuracy (%)",
    "correlation": "Pearson Correlation (ρ)"
}

YRANGES = {
    "logit_lens":  (0.0, 100.0, 10.0),
    "correlation": (-1.0, 1.0, 0.25)
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

    ymin, ymax, y_interval = YRANGES[plot_type]
    _format_line_axes(ax, title=model_name, xlabel="Transformer Layer", ylabel=YLABELS[plot_type],
                      ymin=ymin, ymax=ymax, y_interval=y_interval)

    safe_model = model_name.replace("/", "-").replace(" ", "_")
    base = os.path.join(save_dir, f"{plot_type}_{task_name}_{safe_model}")

    plt.subplots_adjust(bottom=0.15)
    plt.savefig(base + ".pdf", format="pdf", bbox_inches="tight")
    plt.savefig(base + ".png", format="png", bbox_inches="tight")
    print(f"  Plot successfully saved to: {base}.pdf")
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 2. UNIFIED CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
# USER SELECTIONS: Change these variables to run different configurations
SELECTED_MODEL = "Qwen/Qwen3-4B" # Options: "meta-llama/Llama-3.2-3B-Instruct", "google/gemma-2-2b-it", "Qwen/Qwen3-4B"
SELECTED_TASK  = "NLI"             # Options: "NER", "POS", "NLI"
MAX_ROWS       = 300
FIGURES_DIR    = "/content/drive/MyDrive/Figures"

# Task-Specific Configurations
TASKS = {
    "NER": {
        "dataset_path": "/content/drive/MyDrive/comi_lingua_annotations.jsonl",
        "scripts_config": {
            "Roman":      {"text": "rom_text", "gold": "gold_rom", "hint": "The text is Hindi written in Roman script. Extract the single most prominent named entity. Return ONLY the entity name as it appears in the text. No explanation, no punctuation."},
            "Devanagari": {"text": "dev_text", "gold": "gold_dev", "hint": "The text is Hindi in Devanagari script. Extract the single most prominent named entity. Return ONLY the entity name in Devanagari, exactly as it appears. No explanation, no punctuation."},
            "Mixed":      {"text": "mix_text", "gold": "gold_mix", "hint": "The text is code-mixed Hindi (Devanagari + Roman). Extract the single most prominent named entity. Return ONLY the entity name exactly as it appears in the text. No explanation, no punctuation."}
        }
    },
    "POS": {
        "dataset_path": "/content/drive/MyDrive/hindi_pos_dataset_clean.jsonl",
        "scripts_config": {
            "Roman":      {"text": "tokens_roman", "gold": "gold_tags_orig", "hint": "Hindi in Roman script (ITRANS)"},
            "Devanagari": {"text": "tokens_orig",  "gold": "gold_tags_orig", "hint": "Hindi in Devanagari script"},
            "Mixed":      {"text": "tokens_mixed", "gold": "gold_tags_orig", "hint": "code-mixed Hindi (Devanagari and Roman)"}
        }
    },
    "NLI": {
        "dataset_path": "/content/drive/MyDrive/Figures/indic_mixed_dataset.jsonl",
        "scripts_config": {
            "Roman":      {"p": "premise_roman", "h": "hypothesis_roman", "gold": "label"},
            "Devanagari": {"p": "premise_orig",  "h": "hypothesis_orig",  "gold": "label"},
            "Mixed":      {"p": "premise_mixed", "h": "hypothesis_mixed", "gold": "label"}
        }
    }
}

task_cfg = TASKS[SELECTED_TASK]
DATA_PATH = task_cfg["dataset_path"]
scripts_config = task_cfg["scripts_config"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
print(f"Loading {SELECTED_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(SELECTED_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    SELECTED_MODEL,
    quantization_config=bnb_config,
    output_hidden_states=True,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

head_weight = model.lm_head.weight

# Determine number of states
dummy_inputs = tokenizer(["dummy"], return_tensors="pt").to(device)
with torch.no_grad():
    dummy_outputs = model(**dummy_inputs)
NUM_LAYERS = len(dummy_outputs.hidden_states)
del dummy_outputs, dummy_inputs
torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════════════════════
# 4. UNIFIED LOGIT LENS LOGIC
# ══════════════════════════════════════════════════════════════════════════════
def prepare_sample(row, conf, task):
    """Extracts prompt components and gold target based on the task type."""
    if task == "NER":
        text, gold = row.get(conf["text"], ""), row.get(conf["gold"], "")
        if not text or not gold or gold == 'ERROR': return None, None, None
        system_prompt = conf["hint"]
        user_prompt = f"Text: {text}"
        target_str = gold
        
    elif task == "POS":
        tokens, tags = row.get(conf["text"], []), row.get(conf["gold"], [])
        if not tokens or not tags or tags == 'ERROR': return None, None, None
        system_prompt = (
            f"You are a linguistic annotator specializing in {conf['hint']}. "
            "Assign a Universal Dependencies POS tag to each word in the provided list. "
            "Valid tags: NOUN, VERB, ADJ, ADV, PRON, DET, ADP, CCONJ, SCONJ, NUM, PART, PUNCT, INTJ, SYM, PROPN, AUX, X. "
            "Return ONLY a comma-separated list of tags in the exact same order. No explanations, no markdown."
        )
        user_prompt = f"Words: {json.dumps(tokens, ensure_ascii=False)}"
        target_str = tags[0] # Target is the first tag
        
    elif task == "NLI":
        p, h, g = row.get(conf["p"], ""), row.get(conf["h"], ""), row.get(conf["gold"], "")
        if not p or not h or g == "" or g == 'ERROR': return None, None, None
        if isinstance(g, int):
            g = {0: "entailment", 1: "neutral", 2: "contradiction"}.get(g, str(g))
        system_prompt = (
            "Classify the relationship between the premise and hypothesis.\n"
            "Reply with exactly one word — either: entailment, neutral, or contradiction.\n"
            "No explanations. No punctuation. Just the one word.\n\n"
        )
        user_prompt = f"Premise: {p}\nHypothesis: {h}"
        target_str = str(g).strip()
        
    return system_prompt, user_prompt, target_str

def format_messages(system_prompt, user_prompt, model_id):
    """Handles model-specific chat template quirks (like Gemma lacking a system role)."""
    if "gemma" in model_id.lower():
        return [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def run_logit_lens(valid_rows, conf):
    """Returns a 2D numpy array of shape (num_layers, num_samples) with 1s and 0s."""
    outcomes = np.zeros((NUM_LAYERS, len(valid_rows)), dtype=np.float32)

    for sample_idx, row in enumerate(tqdm(valid_rows, desc="Probing Layers")):
        system_prompt, user_prompt, target_str = prepare_sample(row, conf, SELECTED_TASK)
        
        gold_tokens = tokenizer(target_str, add_special_tokens=False).input_ids
        target_token_id = gold_tokens[0]

        messages = format_messages(system_prompt, user_prompt, SELECTED_MODEL)
        
        # Handle Qwen's specific chat template requirements
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if "qwen" in SELECTED_MODEL.lower():
            kwargs["enable_thinking"] = False

        prompt = tokenizer.apply_chat_template(messages, **kwargs)
        inputs = tokenizer([prompt], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        for layer_idx, layer_hs in enumerate(outputs.hidden_states):
            last_token_vec = layer_hs[0, -1, :].to(device=head_weight.device, dtype=head_weight.dtype)

            # Apply final layer norm to hidden representation
            normed_vec = model.model.norm(last_token_vec)
            logits = model.lm_head(normed_vec)
            pred_token_id = logits.argmax().item()

            if pred_token_id == target_token_id:
                outcomes[layer_idx, sample_idx] = 1.0

        del outputs, inputs
        torch.cuda.empty_cache()

    return outcomes

def calculate_pearson_correlation(A_matrix, B_matrix):
    correlations = []
    for l in range(NUM_LAYERS):
        std_a, std_b = np.std(A_matrix[l]), np.std(B_matrix[l])
        if std_a == 0 or std_b == 0:
            correlations.append(0.0)
        else:
            r = np.corrcoef(A_matrix[l], B_matrix[l])[0, 1]
            correlations.append(r)
    return correlations

# ══════════════════════════════════════════════════════════════════════════════
# 5. PRE-FILTER DATA AND EXECUTE
# ══════════════════════════════════════════════════════════════════════════════
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw_rows = [json.loads(line.strip()) for line in f]

# Strictly Pre-filter rows to guarantee alignment across all 3 scripts
valid_rows = []
for row in raw_rows:
    is_valid = True
    for script_name, conf in scripts_config.items():
        _, _, target_str = prepare_sample(row, conf, SELECTED_TASK)
        if target_str is None:
            is_valid = False
            break
        g_tokens = tokenizer(target_str, add_special_tokens=False).input_ids
        if not g_tokens:
            is_valid = False
            break
            
    if is_valid:
        valid_rows.append(row)

valid_rows = valid_rows[:MAX_ROWS] if MAX_ROWS else valid_rows
print(f"\nLocked in {len(valid_rows)} fully valid, aligned samples across all 3 scripts.")

# Extract Outcomes
sample_outcomes = {}
accuracy_results = {}

for script_name, conf in scripts_config.items():
    print(f"\nRunning Logit Lens for {script_name}...")
    outcomes = run_logit_lens(valid_rows, conf)
    sample_outcomes[script_name] = outcomes
    accuracy_results[script_name] = (outcomes.mean(axis=1) * 100).tolist()

print("\nCalculating Pearson correlations between scripts...")
correlation_results = {
    "Roman–Devanagari": calculate_pearson_correlation(sample_outcomes["Roman"], sample_outcomes["Devanagari"]),
    "Roman–Mixed":      calculate_pearson_correlation(sample_outcomes["Roman"], sample_outcomes["Mixed"]),
    "Devanagari–Mixed": calculate_pearson_correlation(sample_outcomes["Devanagari"], sample_outcomes["Mixed"]),
}

# ══════════════════════════════════════════════════════════════════════════════
# 6. ARR PLOTTING (BOTH GRAPHS)
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating paper-quality plots...")
display_name = SELECTED_MODEL.split("/")[-1]

plot_line(
    data=accuracy_results,
    plot_type="logit_lens",
    model_name=display_name,
    task_name=SELECTED_TASK,
    save_dir=FIGURES_DIR
)

plot_line(
    data=correlation_results,
    plot_type="correlation",
    model_name=display_name,
    task_name=f"{SELECTED_TASK}_Correlation",
    save_dir=FIGURES_DIR
)