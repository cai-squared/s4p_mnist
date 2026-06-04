from __future__ import annotations

import io
import os

import gradio as gr
import requests
from PIL import Image

MODEL_API_URL = (
    os.environ.get("MODEL_API_URL", "").strip() or os.environ.get("API_URL", "").strip()
)
DEFAULT_MODEL_API_URL = "https://your-cloud-run-endpoint.a.run.app"
if not MODEL_API_URL:
    MODEL_API_URL = DEFAULT_MODEL_API_URL


def _format_api_url(url: str) -> str:
    return url.rstrip("/")


API_URL = _format_api_url(MODEL_API_URL)


def _make_prediction(image: Image.Image) -> tuple[str, str, str]:
    if image is None:
        return "", "", "Please draw or upload a digit image."

    image = image.convert("L").resize((28, 28))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    try:
        response = requests.post(
            f"{API_URL}/predict/image",
            files={"file": ("digit.png", buffer, "image/png")},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        return (
            str(payload.get("digit", "?")),
            f"{payload.get('confidence', 0.0):.3f}",
            "Prediction successful.",
        )
    except requests.Timeout:
        return "", "", "Prediction request timed out."
    except requests.HTTPError as exc:
        detail = None
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = response.text
        return "", "", f"Model API error: {detail or exc}."
    except requests.RequestException as exc:
        return "", "", f"Failed to call model API: {exc}."
    except Exception as exc:
        return "", "", f"Unexpected error: {exc}."


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="S4P MNIST Cloud Run Demo") as demo:
        gr.Markdown(
            "# S4P MNIST Cloud Run Demo\n"
            "Draw or upload a handwritten digit and the app "
            "will call the deployed Cloud Run model endpoint."
        )
        gr.Markdown(
            f"**Model endpoint:** `{API_URL}`\n\n"
            "If this is still the default placeholder, set `MODEL_API_URL` "
            "in your environment or in Hugging Face Space secrets."
        )

        with gr.Row():
            image_input = gr.Image(
                source="upload",
                tool="editor",
                type="pil",
                label="Draw or upload a digit",
            )
            with gr.Column(scale=0.5):
                digit_output = gr.Textbox(label="Predicted digit", interactive=False)
                confidence_output = gr.Textbox(label="Confidence", interactive=False)
                status_output = gr.Textbox(label="Status", interactive=False)
                predict_button = gr.Button("Predict")

        predict_button.click(
            fn=_make_prediction,
            inputs=image_input,
            outputs=[digit_output, confidence_output, status_output],
        )

        gr.Markdown(
            "---\n"
            "### Usage\n"
            "1. Draw a digit or upload a handwritten digit image.\n"
            "2. Click **Predict**.\n"
            "3. The app sends the image to the Cloud Run model endpoint "
            "and returns the predicted digit with confidence."
        )
    return demo


demo = build_interface()

if __name__ == "__main__":
    demo.launch()
