import gradio as gr
import torch
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification
from peft import PeftModel

# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────
MODEL_NAME  = 'distilroberta-base'
MAX_SEQ_LEN = 64
NUM_CLASSES = 3
SAVE_DIR    = r'C:\Users\DSU\OneDrive - Dakota State University\Desktop\pytorch\saved_models'

LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

EMOJI_MAP = {
    'Negative': '🔴 Negative',
    'Neutral' : '🟡 Neutral',
    'Positive': '🟢 Positive'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ──────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

print("Loading base model...")
base_model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_CLASSES
)

print("Loading LoRA adapter weights...")
model = PeftModel.from_pretrained(base_model, SAVE_DIR)
model = model.to(device)
model.eval()
print("Model ready on:", device)

# ──────────────────────────────────────────
# Predict Function
# ──────────────────────────────────────────
def predict_sentiment(text):
    if not text.strip():
        return "⚠️ Please enter a sentence.", {}, ""

    encoding = tokenizer(
        str(text),
        max_length     = MAX_SEQ_LEN,
        padding        = 'max_length',
        truncation     = True,
        return_tensors = 'pt'
    )

    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1)[0]
        pred    = probs.argmax().item()

    label      = LABEL_MAP[pred]
    confidence = round(probs[pred].item() * 100, 2)

    probabilities = {
        '🔴 Negative': round(probs[0].item(), 4),
        '🟡 Neutral' : round(probs[1].item(), 4),
        '🟢 Positive': round(probs[2].item(), 4)
    }

    result_text = "{} — Confidence: {}%".format(EMOJI_MAP[label], confidence)

    return result_text, probabilities, label


# ──────────────────────────────────────────
# Batch Predict Function
# ──────────────────────────────────────────
def predict_batch(texts):
    if not texts.strip():
        return "⚠️ Please enter at least one sentence."

    lines   = [line.strip() for line in texts.strip().split('\n') if line.strip()]
    output  = []

    for i, text in enumerate(lines, 1):
        encoding = tokenizer(
            str(text),
            max_length     = MAX_SEQ_LEN,
            padding        = 'max_length',
            truncation     = True,
            return_tensors = 'pt'
        )
        input_ids      = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=1)[0]
            pred    = probs.argmax().item()

        label      = LABEL_MAP[pred]
        confidence = round(probs[pred].item() * 100, 2)
        output.append("{}. {} → {} ({}%)".format(i, text[:60], EMOJI_MAP[label], confidence))

    return '\n'.join(output)


# ──────────────────────────────────────────
# Example Sentences
# ──────────────────────────────────────────
examples = [
    ["The company reported record profits and strong revenue growth."],
    ["The stock declined sharply after missing earnings estimates."],
    ["The board approved the quarterly dividend as expected."],
    ["Operating losses widened due to higher raw material costs."],
    ["The merger is expected to close in the fourth quarter."]
]

# ──────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────
with gr.Blocks(title="Financial Sentiment Analysis") as demo:

    gr.Markdown("""
    # 📊 Financial Sentiment Analysis
    ### DistilRoBERTa + LoRA | Fine-tuned on Financial News
    Classify financial text as **Negative**, **Neutral**, or **Positive**
    """)

    # ── Tab 1 — Single Prediction ──
    with gr.Tab("Single Prediction"):

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label       = "Enter Financial Text",
                    placeholder = "e.g. The company reported record profits this quarter.",
                    lines       = 3
                )
                with gr.Row():
                    predict_btn = gr.Button("Predict", variant="primary")
                    clear_btn   = gr.Button("Clear")

                gr.Examples(
                    examples  = examples,
                    inputs    = text_input,
                    label     = "Example Sentences"
                )

            with gr.Column(scale=1):
                result_output = gr.Textbox(
                    label    = "Prediction",
                    lines    = 2
                )
                probs_output = gr.Label(
                    label    = "Class Probabilities",
                    num_top_classes = 3
                )
                label_output = gr.Textbox(
                    label    = "Raw Label",
                    visible  = False
                )

        predict_btn.click(
            fn      = predict_sentiment,
            inputs  = text_input,
            outputs = [result_output, probs_output, label_output]
        )
        clear_btn.click(
            fn      = lambda: ("", {}, ""),
            outputs = [result_output, probs_output, label_output]
        )

    # ── Tab 2 — Batch Prediction ──
    with gr.Tab("Batch Prediction"):

        gr.Markdown("Enter **one sentence per line** — up to 50 sentences at once.")

        batch_input = gr.Textbox(
            label       = "Enter Multiple Sentences (one per line)",
            placeholder = "The company reported record profits.\nStock declined after earnings miss.\nDividend approved as expected.",
            lines       = 8
        )

        with gr.Row():
            batch_btn   = gr.Button("Predict All", variant="primary")
            batch_clear = gr.Button("Clear")

        batch_output = gr.Textbox(
            label = "Batch Results",
            lines = 10
        )

        batch_btn.click(
            fn      = predict_batch,
            inputs  = batch_input,
            outputs = batch_output
        )
        batch_clear.click(
            fn      = lambda: ("", ""),
            outputs = [batch_input, batch_output]
        )

    # ── Tab 3 — Model Info ──
    with gr.Tab("Model Info"):
        gr.Markdown("""
        ## Model Details

        | Component | Details |
        |---|---|
        | **Base Model** | distilroberta-base |
        | **Fine-tuning** | LoRA (r=16, alpha=32) |
        | **Task** | 3-class Sentiment Classification |
        | **Labels** | Negative (0) / Neutral (1) / Positive (2) |
        | **Max Seq Length** | 64 tokens |
        | **Device** | {} |

        ## Dataset
        | Property | Value |
        |---|---|
        | **Domain** | Financial News |
        | **Total Samples** | 5,307 |
        | **Negative** | 590 |
        | **Neutral** | 2,874 |
        | **Positive** | 1,843 |

        ## Training
        | Setting | Value |
        |---|---|
        | **Optimizer** | AdamW |
        | **Scheduler** | Linear Warmup |
        | **Epochs** | 3 |
        | **Batch Size** | 64 |
        """.format(str(device)))


# ──────────────────────────────────────────
# Launch
# ──────────────────────────────────────────
if __name__ == "__main__":
   demo.launch(
    server_name = "127.0.0.1",
    server_port = 7860,
    share       = False
)