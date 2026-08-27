from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

from workflow_control.types import ModelMetadata, ProviderUsage


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ModelBackend(Protocol):
    model_id: str

    def count_prompt(self, messages: list[dict[str, str]]) -> tuple[int, str, str]: ...

    def generate(
        self, messages: list[dict[str, str]], max_output_tokens: int
    ) -> tuple[str, ProviderUsage, ModelMetadata]: ...


@dataclass(frozen=True)
class LocalModelConfig:
    model_id: str
    revision: str | None = None
    tokenizer_revision: str | None = None
    local_files_only: bool = True
    trust_remote_code: bool = False
    device: str = "cuda"
    dtype: str = "bfloat16"
    cache_dir: str | None = ".model_cache"


class NativeChatCounter:
    """Exact chat-template token counting over a tokenizer or processor."""

    def __init__(self, chat_component: Any, *, thinking_disabled: bool = False) -> None:
        self.chat_component = chat_component
        self.tokenizer = getattr(chat_component, "tokenizer", chat_component)
        self.thinking_disabled = thinking_disabled

    @property
    def template_hash(self) -> str:
        template = str(getattr(self.tokenizer, "chat_template", ""))
        return sha256_text(template)

    def _template_options(self) -> dict[str, Any]:
        if self.thinking_disabled:
            return {"enable_thinking": False}
        return {}

    def render(self, messages: list[dict[str, str]]) -> str:
        return str(
            self.chat_component.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self._template_options(),
            )
        )

    def count(self, messages: list[dict[str, str]]) -> tuple[int, str, str]:
        rendered = self.render(messages)
        token_ids = self.chat_component.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **self._template_options(),
        )
        if isinstance(token_ids, Mapping):
            token_ids = token_ids["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise ValueError("only batch size 1 is supported")
            token_ids = token_ids[0]
        return len(token_ids), rendered, sha256_text(rendered)


class _LoadedModelGuard:
    owner: int | None = None

    @classmethod
    def claim(cls, backend: object) -> None:
        identity = id(backend)
        if cls.owner is not None and cls.owner != identity:
            raise RuntimeError("another local model is already loaded; unload it first")
        cls.owner = identity

    @classmethod
    def release(cls, backend: object) -> None:
        if cls.owner == id(backend):
            cls.owner = None


def _safetensors_loader_details(
    family: str, *, platform_name: str | None = None
) -> tuple[str, str | None]:
    is_windows_gemma4 = family == "gemma4" and (platform_name or sys.platform) == "win32"
    if is_windows_gemma4:
        return "pread", "windows_gemma4_pread"
    return "mmap", None


@contextmanager
def _scoped_safetensors_loader(
    modeling_utils: Any,
    *,
    family: str,
    platform_name: str | None = None,
) -> Iterator[None]:
    backend, _ = _safetensors_loader_details(family, platform_name=platform_name)
    if backend != "pread":
        yield
        return

    import safetensors

    original_safe_open = modeling_utils.safe_open

    def pread_safe_open(*args: Any, **kwargs: Any) -> Any:
        kwargs["backend"] = "pread"
        return safetensors.safe_open(*args, **kwargs)

    modeling_utils.safe_open = pread_safe_open
    try:
        yield
    finally:
        modeling_utils.safe_open = original_safe_open


def _load_model_from_pretrained(
    model_class: Any,
    source: str,
    *,
    family: str,
    modeling_utils: Any,
    platform_name: str | None = None,
    **kwargs: Any,
) -> Any:
    with _scoped_safetensors_loader(
        modeling_utils,
        family=family,
        platform_name=platform_name,
    ):
        return model_class.from_pretrained(source, **kwargs)


class HuggingFaceBackend:
    def __init__(self, config: LocalModelConfig, *, family: str) -> None:
        if family not in {"gemma4", "qwen3", "llama", "mistral"}:
            raise ValueError(f"unsupported local model family: {family}")
        self.config = config
        self.family = family
        self.model_id = config.model_id
        self.tokenizer: Any | None = None
        self.chat_component: Any | None = None
        self.model: Any | None = None
        self.counter: NativeChatCounter | None = None
        self._torch: Any | None = None

    @property
    def safetensors_backend(self) -> str:
        return _safetensors_loader_details(self.family)[0]

    @property
    def loader_workaround(self) -> str | None:
        return _safetensors_loader_details(self.family)[1]

    def load(self, *, tokenizer_only: bool = False) -> None:
        _LoadedModelGuard.claim(self)
        try:
            import torch
            import transformers

            if self.config.device != "cuda":
                raise RuntimeError("frozen pilot configuration requires CUDA")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            if self.config.dtype != "bfloat16" or not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16 is unavailable; precision fallback is prohibited")
            self._torch = torch
            source = self._local_source()
            tokenizer_arguments = {
                "revision": self.config.tokenizer_revision or self.config.revision,
                "local_files_only": self.config.local_files_only,
                "trust_remote_code": self.config.trust_remote_code,
            }
            if self.family == "gemma4":
                self.chat_component = transformers.AutoProcessor.from_pretrained(
                    source, **tokenizer_arguments
                )
                self.tokenizer = self.chat_component.tokenizer
            else:
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    source, **tokenizer_arguments
                )
                self.chat_component = self.tokenizer
            self.counter = NativeChatCounter(
                self.chat_component,
                thinking_disabled=self.family in {"gemma4", "qwen3"},
            )
            if tokenizer_only:
                return
            model_class = (
                transformers.AutoModelForMultimodalLM
                if self.family == "gemma4"
                else transformers.AutoModelForCausalLM
            )
            loaded_model: Any = _load_model_from_pretrained(
                model_class,
                source,
                family=self.family,
                modeling_utils=transformers.modeling_utils,
                revision=self.config.revision,
                local_files_only=self.config.local_files_only,
                trust_remote_code=self.config.trust_remote_code,
                dtype=torch.bfloat16,
                device_map={"": "cuda:0"},
                low_cpu_mem_usage=True,
            )
            if getattr(loaded_model, "is_quantized", False):
                raise RuntimeError("quantized model loading is prohibited")
            parameter_devices = {parameter.device.type for parameter in loaded_model.parameters()}
            if parameter_devices != {"cuda"}:
                raise RuntimeError(
                    f"all parameters must load on CUDA without fallback; got {parameter_devices}"
                )
            floating_dtypes = {
                parameter.dtype for parameter in loaded_model.parameters() if parameter.is_floating_point()
            }
            if floating_dtypes != {torch.bfloat16}:
                raise RuntimeError(
                    f"all floating parameters must be BF16; got {floating_dtypes}"
                )
            self.model = loaded_model
            self.model.eval()
        except Exception:
            self.unload()
            raise

    def _local_source(self) -> str:
        if self.config.cache_dir and self.config.revision:
            escaped = self.model_id.replace("/", "--")
            candidate = (
                Path(self.config.cache_dir)
                / f"models--{escaped}"
                / "snapshots"
                / self.config.revision
            )
            if candidate.exists():
                return str(candidate)
        return self.model_id

    def unload(self) -> None:
        model = self.model
        self.model = None
        self.counter = None
        self.chat_component = None
        self.tokenizer = None
        if model is not None:
            del model
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
            self._torch.cuda.empty_cache()
        _LoadedModelGuard.release(self)

    def count_prompt(self, messages: list[dict[str, str]]) -> tuple[int, str, str]:
        if self.counter is None:
            raise RuntimeError("tokenizer is not loaded")
        return self.counter.count(messages)

    def generate(
        self, messages: list[dict[str, str]], max_output_tokens: int
    ) -> tuple[str, ProviderUsage, ModelMetadata]:
        if (
            self.model is None
            or self.tokenizer is None
            or self.chat_component is None
            or self.counter is None
        ):
            raise RuntimeError("model is not loaded")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        torch = self._torch
        if torch is None:
            raise RuntimeError("torch runtime is not loaded")
        options = (
            {"enable_thinking": False} if self.family in {"gemma4", "qwen3"} else {}
        )
        inputs = self.chat_component.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **options,
        ).to("cuda")
        input_length = int(inputs["input_ids"].shape[-1])
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_output_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        output_ids = generated[0, input_length:]
        output_tokens = int(output_ids.shape[-1])
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        rendered = self.counter.render(messages)
        finish_reason = "length" if output_tokens == max_output_tokens else "stop"
        usage = ProviderUsage(
            input_tokens=input_length,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            mapped_finish_class=finish_reason,
            latency_seconds=time.perf_counter() - started,
        )
        metadata = self.metadata(rendered_prompt=rendered)
        return text, usage, metadata

    def metadata(self, *, rendered_prompt: str | None = None) -> ModelMetadata:
        if self.counter is None or self.tokenizer is None:
            raise RuntimeError("tokenizer is not loaded")
        torch_version = getattr(self._torch, "__version__", None)
        cuda_version = getattr(getattr(self._torch, "version", None), "cuda", None)
        return ModelMetadata(
            model_id=self.model_id,
            model_revision=self.config.revision,
            tokenizer_revision=self.config.tokenizer_revision or self.config.revision,
            transformers_version=_package_version("transformers"),
            torch_version=torch_version,
            cuda_version=cuda_version,
            dtype=self.config.dtype,
            device=self.config.device,
            decoding_parameters={
                "do_sample": False,
                "batch_size": 1,
                "num_beams": 1,
                "thinking": (
                    "disabled"
                    if self.family in {"gemma4", "qwen3"}
                    else "not_applicable"
                ),
            },
            chat_template_sha256=self.counter.template_hash,
            rendered_prompt_sha256=(
                sha256_text(rendered_prompt) if rendered_prompt is not None else None
            ),
            tokenizer_vocabulary_size=len(self.tokenizer),
            safetensors_backend=self.safetensors_backend,
            loader_workaround=self.loader_workaround,
        )


class GeminiAccountingAdapter:
    """Strict accounting adapter for the existing Gemini scientific client."""

    model_id = "gemini-2.5-flash"

    @staticmethod
    def validate_call_record(record: Any) -> ProviderUsage:
        if not getattr(record, "admitted", False):
            raise ValueError("cannot debit a non-admitted provider call")
        usage = record.usage
        if getattr(usage, "token_count_estimated", True):
            raise RuntimeError("provider-reported token accounting is required")
        if usage.total_tokens != usage.input_tokens + usage.output_tokens:
            raise ValueError("inconsistent provider token totals")
        return ProviderUsage(
            input_tokens=int(usage.input_tokens),
            output_tokens=int(usage.output_tokens),
            finish_reason=getattr(record, "finish_reason", None),
            mapped_finish_class=_map_finish_reason(getattr(record, "finish_reason", None)),
            latency_seconds=float(getattr(record, "provider_wall_time_seconds", 0.0)),
        )


class GeminiScientificBackend:
    """Common-interface wrapper around the frozen Gemini client implementation."""

    model_id = "gemini-2.5-flash"

    def __init__(self) -> None:
        from agent.models import GeminiClient, ModelConfig

        self.client = GeminiClient(
            ModelConfig(name=self.model_id, temperature=0.0, thinking_budget=0)
        )

    @staticmethod
    def _render(messages: list[dict[str, str]]) -> str:
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError("Gemini scientific path requires one direct user prompt")
        return messages[0]["content"]

    def count_prompt(self, messages: list[dict[str, str]]) -> tuple[int, str, str]:
        rendered = self._render(messages)
        token_count, estimated = self.client._count_prompt_tokens(rendered)
        if estimated:
            raise RuntimeError("exact provider prompt count is required")
        return token_count, rendered, sha256_text(rendered)

    def generate(
        self, messages: list[dict[str, str]], max_output_tokens: int
    ) -> tuple[str, ProviderUsage, ModelMetadata]:
        from agent.budget import BudgetManager

        rendered = self._render(messages)
        input_tokens, _, prompt_hash = self.count_prompt(messages)
        budget = BudgetManager(input_tokens + max_output_tokens)
        record = self.client.generate("stage", rendered, budget, max_output_tokens)
        usage = GeminiAccountingAdapter.validate_call_record(record)
        metadata = ModelMetadata(
            model_id=self.model_id,
            model_revision=None,
            tokenizer_revision=None,
            transformers_version=None,
            torch_version=None,
            cuda_version=None,
            dtype="provider_managed",
            device="provider",
            decoding_parameters={
                "temperature": 0.0,
                "thinking_budget": 0,
                "batch_size": 1,
            },
            chat_template_sha256=sha256_text("gemini-direct-content-v1"),
            rendered_prompt_sha256=prompt_hash,
        )
        return record.raw_output, usage, metadata


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _map_finish_reason(reason: str | None) -> str:
    normalized = (reason or "").upper()
    if normalized in {"STOP", "EOS", "END_TURN"}:
        return "stop"
    if normalized in {"MAX_TOKENS", "LENGTH"}:
        return "length"
    if normalized in {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST"}:
        return "safety"
    return "unknown"
