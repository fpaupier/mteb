from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from datasets import Dataset
from PIL import Image
from sentence_transformers import SentenceTransformer
from sentence_transformers.base.modules import Transformer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from mteb.mocks.mock_tasks import MockRetrievalTask
from mteb.models.model_implementations.google_embeddinggemma import (
    EmbeddingGemma2Wrapper,
    embedding_gemma_2,
)
from mteb.types import PromptType


class CapturingModel(torch.nn.Module):
    def __init__(self, *args: Any, truncate_dim=None, **kwargs: Any):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.projection = torch.nn.Linear(4, 4, bias=False)
        self.truncate_dim = truncate_dim or 768
        self.init_kwargs = kwargs
        self.calls = []
        self.prompts = {
            "query": "task: search result | query: ",
            "document": "title: none | text: ",
            "Retrieval": "task: search result | query: ",
            "Retrieval-document": "title: none | text: ",
            "Reranking": "task: search result | query: ",
            "Classification": "task: classification | query: ",
            "Clustering": "task: clustering | query: ",
            "STS": "task: sentence similarity | query: ",
        }

    def encode(self, inputs, **kwargs: Any):
        self.calls.append((inputs, kwargs))
        vectors = np.ones((len(inputs), self.truncate_dim), dtype=np.float32)
        if kwargs.get("normalize_embeddings"):
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


@pytest.fixture
def wrapper(monkeypatch):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", CapturingModel)
    return EmbeddingGemma2Wrapper("google/embeddinggemma-2")


def encode(
    wrapper,
    data,
    *,
    task_type="Retrieval",
    prompt_type=None,
    domains=None,
    **kwargs: Any,
):
    metadata = MockRetrievalTask.metadata.model_copy(
        update={"type": task_type, "domains": domains}
    )
    return wrapper.encode(
        DataLoader(
            Dataset.from_dict(data),
            batch_size=2,
            collate_fn=lambda rows: {
                key: [row[key] for row in rows] for key in rows[0]
            },
        ),
        task_metadata=metadata,
        hf_split="test",
        hf_subset="default",
        prompt_type=prompt_type,
        show_progress_bar=False,
        **kwargs,
    )


def test_context_window_limits_actual_tokenizer(monkeypatch):
    unknown_word = "[UNK]"
    backend = Tokenizer(
        WordLevel({unknown_word: 0, "token": 1}, unk_token=unknown_word)
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token=unknown_word
    )
    # Exercise the real SentenceTransformer -> Transformer -> tokenizer setter
    # without constructing or downloading any model weights.
    transformer = Transformer.__new__(Transformer)
    torch.nn.Module.__init__(transformer)  # noqa: PLC2801 -- skip weight-loading constructor
    transformer.processor = SimpleNamespace(tokenizer=tokenizer)
    model = SentenceTransformer.__new__(SentenceTransformer)
    torch.nn.Module.__init__(model)  # noqa: PLC2801 -- skip weight-loading constructor
    model.add_module("0", transformer)
    model.prompts = {}
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda *args, **kwargs: model
    )

    wrapper = EmbeddingGemma2Wrapper("google/embeddinggemma-2")

    assert wrapper.model.max_seq_length == 8192
    assert tokenizer.model_max_length == 8192
    assert len(tokenizer("token " * 9000, truncation=True)["input_ids"]) == 8192


def test_encode_preserves_caller_processing_kwargs(wrapper):
    processing_kwargs = {
        "text": {"max_length": 1024, "truncation": True, "pad_to_multiple_of": 128},
        "video": {"do_sample_frames": False},
    }
    encode(wrapper, {"text": ["example"]}, processing_kwargs=processing_kwargs)
    assert wrapper.model.calls[-1][1]["processing_kwargs"] == processing_kwargs


@pytest.mark.parametrize(
    "task_type", ["Retrieval", "Reranking", "InstructionRetrieval"]
)
def test_document_uses_title_and_body_once(wrapper, task_type):
    encode(
        wrapper,
        {
            "text": ["A title The body", "No title"],
            "body": ["The body", "No title"],
            "title": ["A title", ""],
        },
        task_type=task_type,
        prompt_type=PromptType.document,
    )
    inputs, kwargs = wrapper.model.calls[-1]
    assert inputs == ["title: A title | text: The body", "title: none | text: No title"]
    assert kwargs["prompt"] == ""  # noqa: PLC1901 -- None would enable a default prompt


@pytest.mark.parametrize(
    ("task_type", "prompt_type", "expected"),
    [
        ("Retrieval", PromptType.query, "task: search result | query: example"),
        ("Retrieval", PromptType.document, "title: none | text: example"),
        ("Classification", None, "task: classification | query: example"),
        (
            "Classification",
            PromptType.document,
            "task: classification | query: example",
        ),
        ("Clustering", None, "task: clustering | query: example"),
        ("STS", None, "task: sentence similarity | query: example"),
    ],
)
def test_task_prefixes(wrapper, task_type, prompt_type, expected):
    encode(wrapper, {"text": ["example"]}, task_type=task_type, prompt_type=prompt_type)
    assert wrapper.model.calls[-1][0] == [expected]


def test_programming_retrieval_uses_code_prefix(wrapper):
    encode(
        wrapper,
        {"text": ["example"]},
        prompt_type=PromptType.query,
        domains=["Programming", "Written"],
    )
    assert wrapper.model.calls[-1][0] == ["task: code retrieval | query: example"]


@pytest.mark.parametrize(
    ("task_type", "expected"),
    [
        ("ZeroShotClassification", "task: classification | query: example"),
        ("AudioZeroshotClassification", "task: classification | query: example"),
        ("VideoZeroshotClassification", "task: classification | query: example"),
        ("ImageClustering", "task: clustering | query: example"),
        ("AudioPairClassification", "task: sentence similarity | query: example"),
    ],
)
def test_unmapped_task_text_uses_simplified_prefix(wrapper, task_type, expected):
    encode(wrapper, {"text": ["example"]}, task_type=task_type)
    assert wrapper.model.calls[-1][0] == [expected]


@pytest.mark.parametrize("prompt", ["custom: ", ""])
def test_custom_code_query_prefix_takes_precedence(monkeypatch, prompt):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", CapturingModel)
    wrapper = EmbeddingGemma2Wrapper(
        "google/embeddinggemma-2", model_prompts={"query": prompt}
    )
    encode(
        wrapper,
        {"text": ["example"]},
        prompt_type=PromptType.query,
        domains=["Programming"],
    )
    assert wrapper.model.calls[-1][0] == [prompt + "example"]


@pytest.mark.parametrize("prompt_key", ["document", "Retrieval", "Retrieval-document"])
def test_custom_document_prefix_takes_precedence(monkeypatch, prompt_key):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", CapturingModel)
    wrapper = EmbeddingGemma2Wrapper(
        "google/embeddinggemma-2", model_prompts={prompt_key: "custom: "}
    )
    encode(
        wrapper,
        {"text": ["A title The body"], "body": ["The body"], "title": ["A title"]},
        prompt_type=PromptType.document,
    )
    assert wrapper.model.calls[-1][0] == ["custom: The body"]


def test_unmatched_custom_prefix_retains_document_format(monkeypatch):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", CapturingModel)
    wrapper = EmbeddingGemma2Wrapper(
        "google/embeddinggemma-2", model_prompts={"query": "custom: "}
    )
    encode(
        wrapper,
        {"text": ["A title The body"], "body": ["The body"], "title": ["A title"]},
        prompt_type=PromptType.document,
    )
    assert wrapper.model.calls[-1][0] == ["title: A title | text: The body"]


def test_rejects_preloaded_model():
    with pytest.raises(TypeError, match="requires a model name or path"):
        EmbeddingGemma2Wrapper(CapturingModel(), embed_dim=128)


def test_image_has_no_text_prefix(wrapper):
    encode(
        wrapper,
        {"image": [Image.new("RGB", (8, 8))]},
        task_type="Any2AnyRetrieval",
        prompt_type=PromptType.document,
    )
    inputs, kwargs = wrapper.model.calls[-1]
    assert list(inputs[0]) == ["image"]
    assert kwargs["prompt"] == ""  # noqa: PLC1901 -- None would enable a default prompt
    assert kwargs["normalize_embeddings"] is True


def test_mixed_input_preserves_text_before_media(wrapper):
    encode(
        wrapper,
        {"text": ["What is this? <|image|>"], "image": [Image.new("RGB", (8, 8))]},
        task_type="Any2AnyRetrieval",
        prompt_type=PromptType.query,
    )
    inputs, kwargs = wrapper.model.calls[-1]
    assert list(inputs[0]) == ["text", "image"]
    assert inputs[0]["text"] == "task: search result | query: What is this? <|image|>"
    assert kwargs["prompt"] == ""  # noqa: PLC1901 -- None would enable a default prompt


@pytest.mark.parametrize("dimension", [128, 256, 512, 768])
def test_matryoshka_embeddings_are_normalized(monkeypatch, dimension):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", CapturingModel)
    wrapper = EmbeddingGemma2Wrapper("google/embeddinggemma-2", embed_dim=dimension)
    embeddings = encode(wrapper, {"text": ["example"]}, task_type="STS")
    assert embeddings.shape == (1, dimension)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-6)


@pytest.mark.parametrize("dtype", ["float16", torch.float16])
def test_rejects_float16(dtype):
    with pytest.raises(ValueError, match="float32 or bfloat16"):
        EmbeddingGemma2Wrapper("unused", model_kwargs={"dtype": dtype})


def test_rejects_unsupported_dimension():
    with pytest.raises(ValueError, match="128, 256, 512 or 768"):
        EmbeddingGemma2Wrapper("unused", embed_dim=100)


def test_selective_loading_metadata_keeps_experiment(monkeypatch):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", CapturingModel)
    config_kwargs: dict[str, Any] = {"vision_config": None, "audio_config": None}
    wrapper = EmbeddingGemma2Wrapper(
        "google/embeddinggemma-2", config_kwargs=config_kwargs
    )
    experiment = {"config_kwargs": config_kwargs, "embed_dim": 256}
    wrapper.mteb_model_meta = embedding_gemma_2.model_copy(
        update={"experiment_kwargs": experiment}
    )
    meta = wrapper.mteb_model_meta
    assert meta.modalities == ["text"]
    assert meta.n_parameters == 48
    assert meta.n_embedding_parameters == 32
    assert meta.experiment_kwargs == experiment
    assert meta.name == "google/embeddinggemma-2"
    assert wrapper.model.init_kwargs["config_kwargs"] == config_kwargs
    assert wrapper.model.init_kwargs["model_kwargs"]["dtype"] == torch.float32
