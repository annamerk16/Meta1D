import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load fine-tuned model and tokenizer

MODEL_NAME = "MoEid25/RoBERTa_Bias_Detection_Model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()


# Prediction function

def detect_bias(text):
    if not text.strip():
        return {"Prediction": "No input provided."}

    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()

    confidence = probs[0][prediction].item()

    label_map = {0: "Non-Biased", 1: "Biased"}

    return {
        "Prediction": label_map[prediction],
        "Confidence": round(float(confidence), 4)
    }


# Gradio Interface

interface = gr.Interface(
    fn=detect_bias,
    inputs=gr.Textbox(lines=8, placeholder="Enter text…"),
    outputs=gr.JSON(),
    title="Bias Detection (RoBERTa Model)",
    description="Type or paste text to detect whether it contains biased content.",
    examples=[
        ["I think women are naturally worse at math."],
        ["Everyone deserves equal opportunities in society."],
        ["Immigrants take all of our jobs!!"]
    ]
)

# Launch (works locally and on Hugging Face Spaces)

if __name__ == "__main__":
    interface.launch()
