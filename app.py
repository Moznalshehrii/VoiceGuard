"""Step F: demo. Web UI -- user uploads a speech clip, model returns a
verdict (real/spoof) with a confidence score.

Run with:
    python3 app.py

Person 6 also owns compiling the FINAL RESULTS TABLE once everyone else's
numbers are in -- EER for: baseline CNN, main model on ASVspoof eval,
ASVspoof2021 DF (Person 2), In-the-Wild (Person 3), held-out attacks
(Person 4), with/without augmentation (Person 5). Put that table in
README.md under the "## Results" section, not in this file.
"""

import gradio as gr
import torch

from src.data.dataset import load_and_fix_length
from src.models.wav2vec_classifier import Wav2VecSpoofClassifier

CHECKPOINT_PATH = "checkpoints/wav2vec_spoof_real.pt"
SAMPLE_RATE = 16000
DURATION_S = 4.0

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Loading model on {device} ...")
model = Wav2VecSpoofClassifier().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()
print("Model loaded.")


def predict(audio_filepath: str):
    """Gradio passes the uploaded file's path here. Returns class -> confidence."""
    if audio_filepath is None:
        return {}

    waveform = load_and_fix_length(audio_filepath, SAMPLE_RATE, int(SAMPLE_RATE * DURATION_S), train=False)
    waveform = waveform.squeeze(0)
    # same zero-mean/unit-variance normalization as RawWaveformSpoofDataset,
    # so the demo sees exactly what the model was trained/evaluated on
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)
    waveform = waveform.unsqueeze(0).to(device)  # [1, num_samples]

    with torch.no_grad():
        logits = model(waveform)
        probs = torch.softmax(logits, dim=1)[0]

    return {
        "Real (Bonafide)": float(probs[1]),
        "AI-generated (Spoof)": float(probs[0]),
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload a speech clip / ارفع مقطع صوتي"),
    outputs=gr.Label(label="Verdict / النتيجة", num_top_classes=2),
    title="Sawt (صوت) — AI Voice Deepfake Detector",
    description=(
        "Upload a short (3-5 second) speech clip to check whether it's real human "
        "speech or AI-generated.\n"
        "ارفعي مقطع صوتي قصير (3-5 ثواني) لمعرفة إذا كان صوت بشري حقيقي أو مصطنع "
        "بالذكاء الاصطناعي."
    ),
)

if __name__ == "__main__":
    demo.launch(share=True)
