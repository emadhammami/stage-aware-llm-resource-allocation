from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import safetensors

from workflow_control.backends import (
    GeminiAccountingAdapter,
    NativeChatCounter,
    _load_model_from_pretrained,
    _safetensors_loader_details,
)


class FakeTokenizer:
    chat_template = "fixture-template"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                **kwargs,
            }
        )
        return [1, 2, 3, 4] if tokenize else "rendered prompt"


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )


def test_windows_gemma4_selects_pread() -> None:
    assert _safetensors_loader_details("gemma4", platform_name="win32") == (
        "pread",
        "windows_gemma4_pread",
    )


@pytest.mark.parametrize(
    ("family", "platform_name"),
    [
        ("gemma4", "linux"),
        ("qwen3", "win32"),
        ("llama", "win32"),
    ],
)
def test_pread_is_not_selected_for_other_platform_or_family(
    family: str, platform_name: str
) -> None:
    assert _safetensors_loader_details(family, platform_name=platform_name) == (
        "mmap",
        None,
    )


def test_windows_gemma4_safe_open_is_restored_after_success(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def original_safe_open(*args: Any, **kwargs: Any) -> str:
        return "original"

    def fake_safetensors_safe_open(*args: Any, **kwargs: Any) -> str:
        calls.append(kwargs)
        return "pread-opened"

    modeling_utils = SimpleNamespace(safe_open=original_safe_open)
    monkeypatch.setattr(safetensors, "safe_open", fake_safetensors_safe_open)

    class SuccessfulModel:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs: Any) -> str:
            assert modeling_utils.safe_open is not original_safe_open
            assert modeling_utils.safe_open("weights", framework="pt") == "pread-opened"
            assert kwargs == {
                "revision": "fixture-revision",
                "local_files_only": True,
                "trust_remote_code": False,
                "dtype": "bfloat16",
                "device_map": {"": "cuda:0"},
                "low_cpu_mem_usage": True,
            }
            return source

    loaded = _load_model_from_pretrained(
        SuccessfulModel,
        "fixture",
        family="gemma4",
        modeling_utils=modeling_utils,
        platform_name="win32",
        revision="fixture-revision",
        local_files_only=True,
        trust_remote_code=False,
        dtype="bfloat16",
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    )

    assert loaded == "fixture"
    assert calls == [{"framework": "pt", "backend": "pread"}]
    assert modeling_utils.safe_open is original_safe_open


def test_windows_gemma4_safe_open_is_restored_after_exception(monkeypatch) -> None:
    def original_safe_open(*args: Any, **kwargs: Any) -> str:
        return "original"

    modeling_utils = SimpleNamespace(safe_open=original_safe_open)
    monkeypatch.setattr(safetensors, "safe_open", lambda *args, **kwargs: "pread-opened")

    class FailingModel:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs: Any) -> None:
            assert modeling_utils.safe_open is not original_safe_open
            raise RuntimeError("fixture load failure")

    with pytest.raises(RuntimeError, match="fixture load failure"):
        _load_model_from_pretrained(
            FailingModel,
            "fixture",
            family="gemma4",
            modeling_utils=modeling_utils,
            platform_name="win32",
        )

    assert modeling_utils.safe_open is original_safe_open


@pytest.mark.parametrize("family", ["gemma4", "qwen3", "llama"])
def test_non_pread_paths_do_not_override_safe_open(family: str) -> None:
    def original_safe_open(*args: Any, **kwargs: Any) -> str:
        return "original"

    modeling_utils = SimpleNamespace(safe_open=original_safe_open)

    class SuccessfulModel:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs: Any) -> str:
            assert modeling_utils.safe_open is original_safe_open
            return source

    platform_name = "linux" if family == "gemma4" else "win32"
    _load_model_from_pretrained(
        SuccessfulModel,
        "fixture",
        family=family,
        modeling_utils=modeling_utils,
        platform_name=platform_name,
    )

    assert modeling_utils.safe_open is original_safe_open


def test_qwen_chat_template_count_disables_thinking() -> None:
    tokenizer = FakeTokenizer()
    count, rendered, digest = NativeChatCounter(
        tokenizer, thinking_disabled=True
    ).count([{"role": "user", "content": "x"}])
    assert count == 4
    assert rendered == "rendered prompt"
    assert len(digest) == 64
    assert all(call["enable_thinking"] is False for call in tokenizer.calls)


@pytest.mark.parametrize("component", [FakeTokenizer(), FakeProcessor()])
def test_gemma_and_llama_native_chat_count_without_qwen_option(component) -> None:
    count, _, _ = NativeChatCounter(component).count([{"role": "user", "content": "x"}])
    assert count == 4
    tokenizer = getattr(component, "tokenizer", component)
    assert all("enable_thinking" not in call for call in tokenizer.calls)


def test_gemini_provider_accounting_compatibility() -> None:
    record = SimpleNamespace(
        admitted=True,
        usage=SimpleNamespace(
            input_tokens=11,
            output_tokens=7,
            total_tokens=18,
            token_count_estimated=False,
        ),
        provider_wall_time_seconds=0.2,
        finish_reason="STOP",
    )
    usage = GeminiAccountingAdapter.validate_call_record(record)
    assert usage.input_tokens == 11
    assert usage.output_tokens == 7
    assert usage.total_tokens == 18
    assert usage.mapped_finish_class == "stop"


def test_gemini_estimated_accounting_is_rejected() -> None:
    record = SimpleNamespace(
        admitted=True,
        usage=SimpleNamespace(
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            token_count_estimated=True,
        ),
    )
    with pytest.raises(RuntimeError, match="provider-reported"):
        GeminiAccountingAdapter.validate_call_record(record)


@pytest.mark.parametrize(
    ("model_id", "revision", "family", "expected"),
    [
        (
            "Qwen/Qwen3-8B",
            "b968826d9c46dd6066d109eabc6255188de91218",
            "qwen3",
            16,
        ),
        (
            "meta-llama/Llama-3.1-8B-Instruct",
            "0e9e39f249a16976918f6564b8830bc894c89659",
            "llama",
            None,
        ),
        (
            "google/gemma-4-E4B-it",
            "ee0ef6023621cff504d758262d4e04895a5af4a2",
            "gemma4",
            None,
        ),
    ],
)
def test_cached_native_tokenizers_exact_chat_count(
    model_id: str, revision: str, family: str, expected: int | None
) -> None:
    transformers = pytest.importorskip("transformers")
    root = Path(__file__).parents[2]
    escaped = model_id.replace("/", "--")
    location = root / ".model_cache" / f"models--{escaped}" / "snapshots" / revision
    if not location.exists():
        pytest.skip("pinned tokenizer cache is not installed")
    if family == "gemma4":
        component = transformers.AutoProcessor.from_pretrained(
            str(location.relative_to(root)), local_files_only=True
        )
    else:
        component = transformers.AutoTokenizer.from_pretrained(
            str(location.relative_to(root)), local_files_only=True
        )
    counter = NativeChatCounter(
        component, thinking_disabled=family in {"gemma4", "qwen3"}
    )
    messages = [{"role": "user", "content": "offline tokenizer preflight"}]
    first = counter.count(messages)
    second = counter.count(messages)
    assert first == second
    count, rendered, digest = first
    assert count > 0
    assert len(digest) == 64
    if expected is not None:
        assert count == expected
    if family == "qwen3":
        assert "<think>\n\n</think>" in rendered
