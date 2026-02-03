# Agent 0: Hugging Face OCR (image -> text)
from PIL import Image
# Hugging Face Transformers OCR model (TrOCR)
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
MODEL_NAME = "microsoft/trocr-small-printed"
_processor = None
_model = None

def _load_model():
    global _processor, _model

    if _processor is None or _model is None:
        _processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
        _model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    return _processor, _model

# https://huggingface.co/microsoft/trocr-small-printed
def extract_label_text(image_path: str) -> str:
    processor, model = _load_model()

    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    text = " ".join(text.split())

    return text
