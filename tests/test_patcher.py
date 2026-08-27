import ast

from benchmark.patcher import apply_function_patch


def test_ast_function_replacement(sample_source):
    result = apply_function_patch(
        sample_source,
        "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
        function_name="gcd",
    )
    assert result.applied
    assert result.syntax_valid
    assert result.affected_function == "gcd"
    patched = sample_source.read_text(encoding="utf-8")
    ast.parse(patched)
    assert "while b" in patched


def test_syntax_error_is_classified(sample_source):
    result = apply_function_patch(sample_source, "def gcd(:\n    pass\n", function_name="gcd")
    assert result.applied
    assert not result.syntax_valid

