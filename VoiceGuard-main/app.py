"""Person 6 -- Step F: demo.

A simple web UI: user uploads a speech clip, model returns a verdict
(real/spoof) with a confidence score. Gradio is the quickest path to this.

Run with:
    python3 app.py
(installs gradio first: pip3 install --user gradio)

Person 6 also owns compiling the FINAL RESULTS TABLE once everyone else's
numbers are in -- EER for: baseline CNN, main model on ASVspoof eval,
ASVspoof2021 DF (Person 2), In-the-Wild (Person 3), held-out attacks
(Person 4), with/without augmentation (Person 5). Put that table in
README.md under a new "## Results" section, not in this file.
"""

import gradio as gr
import torch

from src.data.dataset import load_and_fix_length
from src.models.wav2vec_classifier import Wav2VecSpoofClassifier

CHECKPOINT_PATH = "checkpoints/wav2vec_spoof.pt"  # TODO (Person 6): point this at the real trained checkpoint
SAMPLE_RATE = 16000
DURATION_S = 4.0

# TODO (Person 6): load once at startup, not per-request
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = None  # Wav2VecSpoofClassifier().to(device); load state_dict; .eval()


def predict(audio_filepath: str) -> str:
    """Gradio passes the uploaded file's path here. Return a human-readable verdict string."""
    if model is None:
        return "Model not loaded yet -- see TODOs in app.py"

    # TODO (Person 6):
    # 1. waveform = load_and_fix_length(audio_filepath, SAMPLE_RATE, int(SAMPLE_RATE*DURATION_S), train=False)
    # 2. normalize the same way RawWaveformSpoofDataset does (zero-mean, unit-variance)
    # 3. run through the model with torch.no_grad(), softmax the logits
    # 4. return something like: "Real (92% confidence)" or "AI-generated (87% confidence)"
    raise NotImplementedError


demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload a speech clip"),
    outputs=gr.Textbox(label="Verdict"),
    title="Voice Guard",
    description="Upload a short speech clip to check whether it's real or AI-generated.",
)

if __name__ == "__main__":
    demo.launch()
