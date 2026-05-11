"""Классификатор команд на основе SetFit. Реализует IClassifier (domain.ports)."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    if not hasattr(torch.distributed, "is_initialized"):
        def _is_initialized():
            return False
        torch.distributed.is_initialized = _is_initialized
except ImportError:
    pass

from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer

from app.adapters.ml.hf_retry import retry_hf


def _get_hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN")


class CommandsClassifier:
    """Классификатор команд на основе SetFit для few-shot learning. Реализует порт IClassifier."""

    def __init__(
        self, model_name: str, confidence_threshold: float = 0.5, cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.model: Optional[SetFitModel] = None
        self.is_trained = False
        self.confidence_threshold = float(confidence_threshold)
        self.cache_dir = cache_dir

    def train(
        self,
        texts: List[str],
        labels: List[str],
        num_iterations: int = 20,
        num_epochs: int = 1,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        device: Optional[str] = None,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError(
                f"Количество текстов ({len(texts)}) не совпадает с количеством меток ({len(labels)})"
            )
        import torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir_path = None
        if self.cache_dir:
            cache_dir_path = Path(self.cache_dir)
            cache_dir_path.mkdir(parents=True, exist_ok=True)
            cache_dir_path = str(cache_dir_path)
        hf_token = _get_hf_token()

        def _load_base_model():
            try:
                return SetFitModel.from_pretrained(
                    self.model_name,
                    cache_dir=cache_dir_path,
                    use_safetensors=True,
                    token=hf_token,
                )
            except Exception:
                return SetFitModel.from_pretrained(
                    self.model_name,
                    cache_dir=cache_dir_path,
                    token=hf_token,
                )

        self.model = retry_hf(_load_base_model)
        self.model = self.model.to(device)
        train_dataset = Dataset.from_dict({"text": texts, "label": labels})
        learning_rate_float = float(learning_rate)
        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=train_dataset,
            num_iterations=num_iterations,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate_float,
            column_mapping={"text": "text", "label": "label"},
        )
        trainer.train()
        self.is_trained = True

    def predict(self, text: str, return_confidence: bool = False) -> str | Tuple[str, float]:
        if not self.is_trained or self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        predictions, confidences = self._predict_with_confidence([text])
        command = predictions[0]
        confidence = confidences[0]
        if confidence < self.confidence_threshold:
            command = "unknown"
        if return_confidence:
            return command, confidence
        return command

    def _predict_with_confidence(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        probs = self.model.predict_proba(texts)
        preds = self.model.predict(texts)
        predictions = []
        confidences = []
        if hasattr(probs, "tolist"):
            probs = probs.tolist()
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        else:
            preds = list(preds)
        for i, prob in enumerate(probs):
            if isinstance(prob, (list, np.ndarray)):
                max_idx = np.argmax(prob)
                max_prob = float(prob[max_idx])
            else:
                max_prob = float(prob)
            predictions.append(str(preds[i]))
            confidences.append(float(max_prob))
        return predictions, confidences

    def predict_batch(
        self, texts: List[str], return_confidence: bool = False
    ) -> List[str] | Tuple[List[str], List[float]]:
        if not self.is_trained or self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        predictions, confidences = self._predict_with_confidence(texts)
        commands = []
        for pred, conf in zip(predictions, confidences):
            if conf < self.confidence_threshold:
                commands.append("unknown")
            else:
                commands.append(pred)
        if return_confidence:
            return commands, confidences
        return commands

    def save(self, model_path: str):
        if not self.is_trained or self.model is None:
            raise ValueError("Модель не обучена. Нечего сохранять.")
        import shutil
        import tempfile
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / path.name
            self.model.save_pretrained(str(temp_path))
            if path.exists():
                shutil.rmtree(path)
            shutil.move(str(temp_path), str(path))

    def load(self, model_path: str, confidence_threshold: Optional[float] = None):
        import warnings
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        hf_token = _get_hf_token()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*mistral.*regex.*", category=UserWarning)
            try:
                self.model = SetFitModel.from_pretrained(
                    str(path), use_safetensors=True, token=hf_token
                )
            except Exception:
                self.model = SetFitModel.from_pretrained(str(path), token=hf_token)
        self.is_trained = True
        if confidence_threshold is not None:
            self.confidence_threshold = float(confidence_threshold)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        hf_token = _get_hf_token()
        if self.model is None:
            def _load_base():
                try:
                    return SetFitModel.from_pretrained(
                        self.model_name,
                        use_safetensors=True,
                        token=hf_token,
                    )
                except Exception:
                    return SetFitModel.from_pretrained(self.model_name, token=hf_token)
            self.model = retry_hf(_load_base)
        if hasattr(self.model, "model_body"):
            embedding_model = self.model.model_body
        elif hasattr(self.model, "model"):
            embedding_model = self.model.model
        else:
            embedding_model = self.model
        if hasattr(embedding_model, "encode"):
            embeddings = embedding_model.encode(texts, convert_to_numpy=True)
        else:
            from sentence_transformers import SentenceTransformer
            base_model = SentenceTransformer(self.model_name, token=hf_token)
            embeddings = base_model.encode(texts, convert_to_numpy=True)
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        result = []
        for emb in embeddings:
            if isinstance(emb, (list, np.ndarray)):
                result.append([float(x) for x in emb])
            else:
                result.append([float(emb)])
        return result
