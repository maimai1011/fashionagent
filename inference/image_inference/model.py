import logging

import mlflow.pyfunc
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel


logger = logging.getLogger(__name__)

class ClothingModel(mlflow.pyfunc.PythonModel):
    GARMENT_LABELS = [
        "t-shirt",
        "shirt",
        "blouse",
        "hoodie",
        "sweater",
        "jacket",
        "coat",
        "dress",
        "skirt",
        "jeans",
        "pants",
        "shorts",
        "activewear",
    ]

    def load_context(self, context):
        logger.info("function_start function=ClothingModel.load_context")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def predict(self, context, model_input):
        logger.info("function_start function=ClothingModel.predict")
        if isinstance(model_input, str):
            image = Image.open(model_input).convert("RGB")
        else:
            image = model_input.convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)

        emb = features[0].cpu().numpy()
        emb = emb / np.linalg.norm(emb)

        return emb.tolist()

    def classify_garment_type(self, image, labels=None):
        logger.info("function_start function=ClothingModel.classify_garment_type")
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        candidate_labels = labels or self.GARMENT_LABELS
        prompts = [f"a photo of a {label}" for label in candidate_labels]

        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1)[0]
        best_idx = int(torch.argmax(probs).item())
        return candidate_labels[best_idx], float(probs[best_idx].item())

    def get_embedding_dim(self):
        logger.info("function_start function=ClothingModel.get_embedding_dim")
        sample = Image.new("RGB", (224, 224), color="black")
        arr = np.asarray(self.predict(None, sample), dtype="float32")
        arr = np.squeeze(arr)

        if arr.ndim == 2:
            arr = arr.mean(axis=0)

        if arr.ndim != 1:
            raise ValueError(f"Unexpected embedding shape while inferring dim: {arr.shape}")

        return len(arr)
