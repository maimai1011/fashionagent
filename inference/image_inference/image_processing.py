"""Wardrobe image preprocessing and inference pipeline."""

from __future__ import annotations

import os
from functools import lru_cache
from io import BytesIO
from typing import TYPE_CHECKING

from PIL import Image, ImageOps

if TYPE_CHECKING:
    from inference.image_inference.model import ClothingModel


@lru_cache(maxsize=1)
def _background_session():
    """Reuse the background removal model across images in one worker process."""
    from rembg import new_session

    return new_session(os.getenv("REMBG_MODEL", "u2net"))


def normalize_wardrobe_image(
    source: bytes,
    max_dimension: int = 2_048,
    quality: int = 90,
) -> bytes:
    """Auto-orient, resize, and encode a wardrobe image as JPEG."""
    if not source:
        raise ValueError("source image cannot be empty")

    with Image.open(BytesIO(source)) as opened:
        opened.load()
        image = ImageOps.exif_transpose(opened)
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")

        output = BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
        return output.getvalue()


class ImageProcessing:
    """Background removal, object extraction, and garment metadata inference."""

    @staticmethod
    def remove_background(image: Image.Image) -> Image.Image:
        """Remove the image background and return an RGBA image."""
        from rembg import remove

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        output = remove(buffer.getvalue(), session=_background_session())
        return Image.open(BytesIO(output)).convert("RGBA")

    @staticmethod
    def extract_object(image: Image.Image, **_: object) -> Image.Image:
        """Keep only the largest visible object in an RGBA image."""
        import cv2
        import numpy as np

        array = np.array(image.convert("RGBA"))
        alpha = array[:, :, 3]
        _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            raise ValueError("no garment object was detected")

        largest = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest], -1, 255, -1)
        array[:, :, 3] = clean_mask
        return Image.fromarray(array)

    @staticmethod
    def crop_to_object(image: Image.Image) -> Image.Image:
        """Crop an RGBA image to its nontransparent object bounds."""
        import cv2
        import numpy as np

        array = np.array(image.convert("RGBA"))
        coordinates = cv2.findNonZero(array[:, :, 3])
        if coordinates is None:
            raise ValueError("cannot crop an image without a visible object")

        x, y, width, height = cv2.boundingRect(coordinates)
        return Image.fromarray(array[y : y + height, x : x + width])

    @staticmethod
    def detect_color(image: Image.Image) -> str:
        """Return a basic dominant-color label for visible garment pixels."""
        import cv2
        import numpy as np

        rgba = np.array(image.convert("RGBA"))
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]
        visible_mask = alpha > 32
        if not np.any(visible_mask):
            return "unknown"

        visible_rgb = rgb[visible_mask]
        hsv = cv2.cvtColor(visible_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
        hsv = hsv.reshape(-1, 3)
        hue = hsv[:, 0]
        saturation = hsv[:, 1]
        value = hsv[:, 2]

        colorful_mask = (saturation >= 40) & (value >= 40)
        if np.any(colorful_mask):
            hue = hue[colorful_mask]
            saturation = saturation[colorful_mask]
            value = value[colorful_mask]

        mean_saturation = float(np.mean(saturation))
        mean_value = float(np.mean(value))
        if mean_value < 40:
            return "black"
        if mean_saturation < 25 and mean_value > 200:
            return "white"
        if mean_saturation < 25:
            return "gray"

        histogram, _ = np.histogram(hue, bins=180, range=(0, 180))
        dominant_hue = int(np.argmax(histogram))
        if dominant_hue <= 10 or dominant_hue >= 170:
            return "red"
        if dominant_hue <= 25:
            return "orange"
        if dominant_hue <= 38:
            return "yellow"
        if dominant_hue <= 85:
            return "green"
        if dominant_hue <= 130:
            return "blue"
        if dominant_hue <= 160:
            return "purple"
        return "red"

    def run_image_pipeline(
        self,
        image: Image.Image,
        model: ClothingModel,
    ) -> tuple[Image.Image, str, str, list[float]]:
        """Run segmentation, metadata classification, and embedding inference."""
        processed = self.remove_background(image)
        processed = self.extract_object(processed)
        processed = self.crop_to_object(processed)
        color = self.detect_color(processed)
        clothing_type, _ = model.classify_garment_type(processed)
        embedding = model.predict(None, processed)
        return processed, color, clothing_type, embedding
