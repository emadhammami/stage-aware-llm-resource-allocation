from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import yaml

from workflow_control.backends import HuggingFaceBackend, LocalModelConfig

MODEL_ORDER = ("qwen3", "llama", "gemma4")
PREFLIGHT_PROMPT = [{"role": "user", "content": "offline tokenizer preflight"}]


def _memory(torch: Any) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "free_bytes": int(free),
        "total_bytes": int(total),
    }


def _snapshot(model: dict[str, Any], root: Path) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=str(model["model_id"]),
        revision=str(model["revision"]),
        cache_dir=root / ".model_cache",
        max_workers=1,
        allow_patterns=[
            "*.json",
            "*.jinja",
            "*.model",
            "*.safetensors",
            "*.safetensors.index.json",
            "tokenizer*",
            "special_tokens_map.json",
            "processor*",
        ],
    )


def _error_metadata(error: Exception) -> dict[str, Any]:
    response = getattr(error, "response", None)
    return {
        "error_type": type(error).__name__,
        "status_code": getattr(response, "status_code", None),
    }


def check_model(model: dict[str, Any], root: Path) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "model_id": model["model_id"],
        "family": model["family"],
        "revision": model["revision"],
        "tokenizer_revision": model["tokenizer_revision"],
        "dtype": model["dtype"],
        "device": model["device"],
        "generation_performed": False,
    }
    backend = HuggingFaceBackend(
        LocalModelConfig(
            model_id=model["model_id"],
            revision=model["revision"],
            tokenizer_revision=model["tokenizer_revision"],
            local_files_only=True,
            device=model["device"],
            dtype=model["dtype"],
        ),
        family=model["family"],
    )
    result["safetensors_backend"] = backend.safetensors_backend
    result["loader_workaround"] = backend.loader_workaround
    try:
        result["snapshot_path"] = _snapshot(model, root)
    except Exception as error:
        result.update(
            {
                "status": "download_blocked",
                **_error_metadata(error),
            }
        )
        return result

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result["memory_before"] = _memory(torch)
    try:
        backend.load()
        torch.cuda.synchronize()
        token_count, rendered, rendered_hash = backend.count_prompt(PREFLIGHT_PROMPT)
        if backend.model is None:
            raise RuntimeError("model disappeared during load-only preflight")
        result.update(
            {
                "status": "loaded",
                "parameter_count": sum(parameter.numel() for parameter in backend.model.parameters()),
                "token_count": token_count,
                "rendered_prompt": rendered,
                "rendered_prompt_sha256": rendered_hash,
                "metadata": backend.metadata(rendered_prompt=rendered).to_dict(),
                "memory_loaded": _memory(torch),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        )
    except Exception as error:
        result.update(
            {
                "status": "load_failed",
                **_error_metadata(error),
            }
        )
    finally:
        backend.unload()
        gc.collect()
        torch.cuda.empty_cache()
        after = _memory(torch)
        result["memory_after_unload"] = after
        before = result["memory_before"]
        result["memory_released"] = (
            after["allocated_bytes"] <= before["allocated_bytes"]
            and after["reserved_bytes"] <= before["reserved_bytes"]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and load one model without generation")
    parser.add_argument("--family", choices=MODEL_ORDER, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path(".model_cache/open_models_preflight_results.json")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    configuration = yaml.safe_load(
        (root / "research/stage_aware/pilot60_config.yaml").read_text(encoding="utf-8")
    )
    model = next(row for row in configuration["models"] if row["family"] == args.family)
    result = check_model(model, root)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    existing[args.family] = result
    output.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
