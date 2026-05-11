"""Unit-тесты для Pydantic-схем запросов API."""

import pytest
from pydantic import ValidationError

from app.api.routes.predict import (
    EmbedRequest,
    PredictBatchRequest,
    PredictRequest,
)
from app.api.routes.training import TrainRequest

# --- EmbedRequest ---


def test_embed_request_valid():
    """Корректный EmbedRequest принимается."""
    r = EmbedRequest(inputs=["hello", "world"])
    assert r.inputs == ["hello", "world"]


def test_embed_request_empty_list_rejected():
    """Пустой список inputs отклоняется (min_length=1)."""
    with pytest.raises(ValidationError):
        EmbedRequest(inputs=[])


def test_embed_request_empty_string_rejected():
    """Пустая строка в inputs отклоняется."""
    with pytest.raises(ValidationError, match="пустым"):
        EmbedRequest(inputs=["ok", ""])


def test_embed_request_too_long_text_rejected():
    """Текст > 5000 символов отклоняется."""
    with pytest.raises(ValidationError, match="5000"):
        EmbedRequest(inputs=["x" * 5001])


# --- PredictRequest ---


def test_predict_request_valid():
    """Корректный PredictRequest принимается."""
    r = PredictRequest(text="лягись", return_confidence=False)
    assert r.text == "лягись"
    assert r.return_confidence is False


def test_predict_request_empty_text_rejected():
    """Пустой текст отклоняется (min_length=1)."""
    with pytest.raises(ValidationError):
        PredictRequest(text="")


def test_predict_request_too_long_rejected():
    """Текст > 5000 символов отклоняется."""
    with pytest.raises(ValidationError):
        PredictRequest(text="x" * 5001)


# --- PredictBatchRequest ---


def test_predict_batch_request_valid():
    """Корректный PredictBatchRequest принимается."""
    r = PredictBatchRequest(texts=["a", "b"], return_confidence=True)
    assert r.texts == ["a", "b"]
    assert r.return_confidence is True


def test_predict_batch_request_empty_string_rejected():
    """Пустая строка в texts отклоняется."""
    with pytest.raises(ValidationError, match="пустым"):
        PredictBatchRequest(texts=["ok", ""])


def test_predict_batch_request_too_long_text_rejected():
    """Текст > 5000 символов в batch отклоняется."""
    with pytest.raises(ValidationError, match="5000"):
        PredictBatchRequest(texts=["x" * 5001])


# --- TrainRequest ---


def test_train_request_all_optional():
    """TrainRequest допускает все поля пустыми (optional)."""
    r = TrainRequest()
    assert r.num_iterations is None
    assert r.num_epochs is None
    assert r.batch_size is None
    assert r.learning_rate is None


def test_train_request_valid_bounds():
    """TrainRequest принимает значения в допустимых границах."""
    r = TrainRequest(
        num_iterations=1,
        num_epochs=1,
        batch_size=1,
        learning_rate=0.01,
    )
    assert r.num_iterations == 1
    assert r.learning_rate == 0.01


def test_train_request_num_iterations_ge_1():
    """num_iterations < 1 отклоняется."""
    with pytest.raises(ValidationError):
        TrainRequest(num_iterations=0)


def test_train_request_learning_rate_gt_0():
    """learning_rate <= 0 отклоняется."""
    with pytest.raises(ValidationError):
        TrainRequest(learning_rate=0.0)


def test_train_request_learning_rate_le_1():
    """learning_rate > 1 отклоняется."""
    with pytest.raises(ValidationError):
        TrainRequest(learning_rate=1.1)
