from __future__ import annotations

import ast
from pathlib import Path

from agent.state import PatchRecord


def _function_defs(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _candidate_function(source: str, function_name: str | None) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    tree = ast.parse(source)
    functions = _function_defs(tree)
    if function_name:
        return next((fn for fn in functions if fn.name == function_name), None)
    if len(functions) == 1:
        return functions[0]
    return None


def _normalize_patch_code(patch_code: str, target_name: str) -> str:
    patch_code = patch_code.strip()
    try:
        patch_tree = ast.parse(patch_code)
    except SyntaxError:
        return patch_code
    patch_functions = _function_defs(patch_tree)
    if patch_functions:
        chosen = next((fn for fn in patch_functions if fn.name == target_name), patch_functions[0])
        return ast.get_source_segment(patch_code, chosen) or patch_code
    return patch_code


def apply_function_patch(
    source_path: str | Path,
    patch_code: str,
    function_name: str | None = None,
) -> PatchRecord:
    path = Path(source_path)
    source = path.read_text(encoding="utf-8")
    try:
        target = _candidate_function(source, function_name)
    except SyntaxError as exc:
        return PatchRecord(
            applied=False,
            syntax_valid=False,
            affected_function=function_name,
            error=f"Original source syntax error: {exc}",
        )
    if target is None:
        return PatchRecord(
            applied=False,
            syntax_valid=False,
            affected_function=function_name,
            proposed_snippet=patch_code,
            error="Could not identify a unique target function.",
        )
    if target.end_lineno is None:
        return PatchRecord(
            applied=False,
            syntax_valid=False,
            affected_function=target.name,
            error="Target function has no end line metadata.",
        )

    normalized_patch = _normalize_patch_code(patch_code, target.name)
    source_lines = source.splitlines()
    original_snippet = "\n".join(source_lines[target.lineno - 1 : target.end_lineno])
    new_lines = (
        source_lines[: target.lineno - 1]
        + normalized_patch.splitlines()
        + source_lines[target.end_lineno :]
    )
    new_source = "\n".join(new_lines) + ("\n" if source.endswith("\n") else "")
    try:
        ast.parse(new_source)
    except SyntaxError as exc:
        return PatchRecord(
            applied=True,
            syntax_valid=False,
            affected_function=target.name,
            original_snippet=original_snippet,
            proposed_snippet=normalized_patch,
            error=str(exc),
        )
    path.write_text(new_source, encoding="utf-8")
    return PatchRecord(
        applied=True,
        syntax_valid=True,
        affected_function=target.name,
        original_snippet=original_snippet,
        proposed_snippet=normalized_patch,
    )

