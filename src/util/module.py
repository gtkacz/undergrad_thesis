import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoImageProcessor, AutoModelForImageClassification,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, pipeline)


class SkinConditionsClassifier:
    def __init__(self, model: str, batch_size: int = 8):
        """
        The SkinConditionsClassifier class is a wrapper around the Hugging Face Transformers pipeline for image classification.

        Args:
            model (str, optional): The model to use for image classification.
            batch_size (int, optional): The batch size to use for image classification. Defaults to 8.
        """
        self._model_name = model
        self.pipe = pipeline('image-classification', model=model, tokenizer=model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), batch_size=batch_size)
        self.processor = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModelForImageClassification.from_pretrained(model)

    def predict(self, img: str | np.ndarray) -> list[dict[str, str | float]]:
        """
        Predict the skin condition of an image.

        Args:
            img (str | np.ndarray): The image to predict the skin condition of.

        Returns:
            list[dict[str, str | float]]: The predicted skin condition(s) of the image.
        """
        return self.pipe.predict(img)

    def refresh_pipeline(self) -> None:
        """
        Refresh the pipeline with the current model.
        """
        self.pipe = pipeline('image-classification', model=self._model_name, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def set_model_name(self, model: str) -> None:
        """
        Set the model to use for image classification.

        Args:
            model (str): The model to use for image classification.
        """
        self._model_name = model
        self.refresh_pipeline()

    def get_model_name(self) -> str:
        """
        Get the model used for image classification.

        Returns:
            str: The model used for image classification.
        """
        return self._model_name
    
    def get_labels(self) -> dict[int, str]:
        """
        Get the labels for the model.

        Returns:
            dict[int, str]: The labels for the model.
        """
        return self.pipe.model.config.id2label
