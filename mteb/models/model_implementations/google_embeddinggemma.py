from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from tqdm.auto import tqdm

from mteb.models import SentenceTransformerEncoderWrapper
from mteb.models.model_implementations.google_gemini import GECKO_TRAINING_DATA
from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import (
    _postprocess_dense_embeddings,
    _resolve_prompt,
    _setup_modality_collator,
)
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs

logger = logging.getLogger(__name__)


MULTILINGUAL_EVALUATED_LANGUAGES = [
    "arb-Arab",
    "ben-Beng",
    "eng-Latn",
    "spa-Latn",
    "deu-Latn",
    "pes-Arab",
    "fin-Latn",
    "fra-Latn",
    "hin-Deva",
    "ind-Latn",
    "jpn-Jpan",
    "kor-Hang",
    "rus-Cyrl",
    "swh-Latn",
    "tel-Telu",
    "tha-Thai",
    "yor-Latn",
    "zho-Hant",
    "zho-Hans",
]


EMBEDDING_GEMMA_CITATION = """
@misc{vera2025embeddinggemmapowerfullightweighttext,
      title={EmbeddingGemma: Powerful and Lightweight Text Representations},
      author={Henrique Schechter Vera and Sahil Dua and Biao Zhang and Daniel Salz and Ryan Mullins and Sindhu Raghuram Panyam and Sara Smoot and Iftekhar Naim and Joe Zou and Feiyang Chen and Daniel Cer and Alice Lisak and Min Choi and Lucas Gonzalez and Omar Sanseviero and Glenn Cameron and Ian Ballantyne and Kat Black and Kaifeng Chen and Weiyi Wang and Zhe Li and Gus Martins and Jinhyuk Lee and Mark Sherwood and Juyeong Ji and Renjie Wu and Jingxiao Zheng and Jyotinder Singh and Abheesht Sharma and Divyashree Sreepathihalli and Aashi Jain and Adham Elarabawy and AJ Co and Andreas Doumanoglou and Babak Samari and Ben Hora and Brian Potetz and Dahun Kim and Enrique Alfonseca and Fedor Moiseev and Feng Han and Frank Palma Gomez and Gustavo Hernández Ábrego and Hesen Zhang and Hui Hui and Jay Han and Karan Gill and Ke Chen and Koert Chen and Madhuri Shanbhogue and Michael Boratko and Paul Suganthan and Sai Meher Karthik Duddu and Sandeep Mariserla and Setareh Ariafar and Shanfeng Zhang and Shijie Zhang and Simon Baumgartner and Sonam Goenka and Steve Qiu and Tanmaya Dabral and Trevor Walker and Vikram Rao and Waleed Khawaja and Wenlei Zhou and Xiaoqi Ren and Ye Xia and Yichang Chen and Yi-Ting Chen and Zhe Dong and Zhongli Ding and Francesco Visin and Gaël Liu and Jiageng Zhang and Kathleen Kenealy and Michelle Casbon and Ravin Kumar and Thomas Mesnard and Zach Gleicher and Cormac Brick and Olivier Lacombe and Adam Roberts and Qin Yin and Yunhsuan Sung and Raphael Hoffmann and Tris Warkentin and Armand Joulin and Tom Duerig and Mojtaba Seyedhosseini},
      year={2025},
      eprint={2509.20354},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.20354},
}"""

embedding_gemma_300m = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,  # type: ignore[call-arg]
    name="google/embeddinggemma-300m",
    model_type=["dense"],
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=True,
    revision="64614b0b8b64f0c6c1e52b07e4e9a4e8fe4d2da2",
    release_date="2025-09-04",
    n_parameters=307_581_696,
    n_embedding_parameters=201_326_592,
    embed_dim=768,
    max_tokens=2048,
    license="gemma",
    reference="https://ai.google.dev/gemma/docs/embeddinggemma/model_card",
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
    similarity_fn_name="cosine",
    memory_usage_mb=1155,
    citation=EMBEDDING_GEMMA_CITATION,
    extra_requirements_groups=["embeddinggemma"],
)


class EmbeddingGemma2Wrapper(SentenceTransformerEncoderWrapper):
    """EmbeddingGemma 2 with task prefixes, document titles and native media inputs."""

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        *,
        device: str | None = None,
        embed_dim: int | None = None,
        config_kwargs: dict[str, Any] | None = None,
        model_prompts: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        import torch

        if not isinstance(model, str):
            raise TypeError(
                "EmbeddingGemma2Wrapper requires a model name or path, "
                "so loading options can be applied to the model"
            )
        if embed_dim is not None and embed_dim not in {128, 256, 512, 768}:
            raise ValueError(
                "EmbeddingGemma 2 supports 128, 256, 512 or 768 dimensions"
            )
        config_kwargs = dict(config_kwargs or {})
        self._modalities = ["text"]
        if config_kwargs.get("vision_config", True) is not None:
            self._modalities.extend(["image", "video"])
        if config_kwargs.get("audio_config", True) is not None:
            self._modalities.append("audio")
        model_kwargs = dict(kwargs.pop("model_kwargs", {}) or {})
        dtype = model_kwargs.get("dtype", model_kwargs.get("torch_dtype"))
        if dtype in {torch.float16, "float16"}:
            raise ValueError("EmbeddingGemma 2 requires float32 or bfloat16 inference")
        if dtype is None:
            model_kwargs["dtype"] = torch.float32
        # The model card specifies 1 FPS video and mono 16 kHz audio.
        kwargs.setdefault("fps", None if kwargs.get("num_frames") else 1.0)
        kwargs.setdefault("target_sampling_rate", 16000)
        super().__init__(
            model,
            revision=revision,
            device=device,
            embed_dim=embed_dim,
            config_kwargs=config_kwargs,
            model_kwargs=model_kwargs,
            **kwargs,
        )
        # The checkpoint does not configure a tokenizer limit. Enforce the
        # advertised context window for default SentenceTransformers preprocessing.
        self.model.max_seq_length = 8192
        self._custom_model_prompts = self.validate_task_to_prompt_name(model_prompts)

    @property
    def mteb_model_meta(self) -> ModelMeta:
        return self._mteb_model_meta

    @mteb_model_meta.setter
    def mteb_model_meta(self, meta: ModelMeta) -> None:
        import torch

        # ModelMeta.load_model assigns the registered metadata after construction.
        # Keep its experiment kwargs while reflecting the encoders actually loaded.
        parameters = list(self.model.parameters())
        n_parameters = sum(parameter.numel() for parameter in parameters)
        n_embedding_parameters = sum(
            module.weight.numel()
            for module in self.model.modules()
            if isinstance(module, torch.nn.Embedding)
        )
        self._mteb_model_meta = meta.model_copy(
            update={
                "modalities": self._modalities,
                "n_parameters": n_parameters,
                "n_embedding_parameters": n_embedding_parameters,
                "memory_usage_mb": round(
                    sum(p.numel() * p.element_size() for p in parameters) / 1024**2
                ),
            },
            deep=True,
        )

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        is_multimodal = _setup_modality_collator(
            inputs,
            fps=self.fps,
            max_frames=self.max_frames,
            num_frames=self.num_frames,
            target_sampling_rate=self.target_sampling_rate,
            max_samples=self.max_samples,
        )
        custom_prompt = _resolve_prompt(
            self._custom_model_prompts, task_metadata, prompt_type
        )
        prompt = custom_prompt
        if prompt is None:
            prompt = _resolve_prompt(self.model_prompts, task_metadata, prompt_type)
        if (
            custom_prompt is None
            and prompt_type == PromptType.query
            and task_metadata.simplified_task_type == "retrieval"
            and "Programming" in (task_metadata.domains or [])
        ):
            prompt = "task: code retrieval | query: "
        if prompt is None:
            # Multimodal task types such as ZeroShotClassification have no entry
            # in the checkpoint's prompt map, but their text still needs a prefix.
            prompt = {
                "classification": "task: classification | query: ",
                "clustering": "task: clustering | query: ",
                "semantic-similarity": "task: sentence similarity | query: ",
                "pair-classification": "task: sentence similarity | query: ",
                "retrieval": "task: search result | query: ",
            }[task_metadata.simplified_task_type]
        is_document = (
            prompt_type == PromptType.document
            and task_metadata.simplified_task_type == "retrieval"
        )

        def prepare_batch(batch: BatchedInput) -> list[Any]:
            text_key = "body" if is_document and "body" in batch else "text"
            keys = [key for key in ("image", "audio", "video") if key in batch]
            if text_key in batch:
                keys.insert(0, text_key)
            prepared = []
            for index in range(len(batch[keys[0]])):
                item = {key: batch[key][index] for key in keys}
                if text_key in item:
                    text = item.pop(text_key)
                    if is_document and custom_prompt is None:
                        title = batch["title"][index] if "title" in batch else None
                        text = f"title: {title or 'none'} | text: {text}"
                    else:
                        text = prompt + text
                    # Text comes first when inputs contain multiple modalities.
                    item = {"text": text, **item}
                prepared.append(item if is_multimodal else item["text"])
            return prepared

        # Prefixes are already attached to text only. An explicit empty prompt
        # prevents SentenceTransformers from adding any default prompt to media.
        encode_kwargs = {**kwargs, "prompt": "", "normalize_embeddings": True}
        if not is_multimodal:
            texts = [text for batch in inputs for text in prepare_batch(batch)]
            return _postprocess_dense_embeddings(
                self.model.encode(texts, **encode_kwargs)
            )
        batches = []
        for batch in tqdm(
            inputs,
            desc="Encoding multimodal inputs",
            disable=not kwargs.get("show_progress_bar", True),
        ):
            embeddings = self.model.encode(
                prepare_batch(batch),
                **{**encode_kwargs, "show_progress_bar": False},
            )
            batches.append(_postprocess_dense_embeddings(embeddings))
        return cast("Array", np.concatenate(batches, axis=0))


embedding_gemma_2 = ModelMeta(
    loader=EmbeddingGemma2Wrapper,
    name="google/embeddinggemma-2",
    model_type=["dense"],
    modalities=["text", "image", "audio", "video"],
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=True,
    revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
    release_date="2026-10-06",
    n_parameters=744_371_488,
    n_embedding_parameters=134_217_728,
    embed_dim=[128, 256, 512, 768],
    max_tokens=8192,
    license="apache-2.0",
    reference="https://huggingface.co/google/embeddinggemma-2",
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    similarity_fn_name="cosine",
    memory_usage_mb=2840,
    citation=None,
    extra_requirements_groups=["embeddinggemma2"],
)
