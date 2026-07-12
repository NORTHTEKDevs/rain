"""Parsing tests for the Ollama judge primitive.

The judge call itself hits the local Ollama daemon and is slow; these tests
cover only the parse + structure of the response, which is what tends to
break when the underlying model changes its output formatting.
"""

from rain.feedback.ollama_judge import (
    build_user_prompt,
    parse_judge_response,
)


def test_parse_bare_json_object():
    txt = '{"correct": true, "confidence": 0.9, "reasoning": "yep"}'
    j = parse_judge_response(txt)
    assert j is not None
    assert j.correct is True
    assert j.confidence == 0.9
    assert j.reasoning == "yep"


def test_parse_json_inside_prose():
    txt = (
        "Sure, here's my judgment:\n"
        '{"correct": false, "confidence": 0.7, "reasoning": "off by one"}\n'
        "Hope that helps!"
    )
    j = parse_judge_response(txt)
    assert j is not None
    assert j.correct is False
    assert j.confidence == 0.7


def test_parse_json_inside_markdown_fence():
    txt = '```json\n{"correct": true, "confidence": 0.5, "reasoning": ""}\n```'
    j = parse_judge_response(txt)
    assert j is not None and j.correct is True


def test_confidence_is_clamped_to_unit_interval():
    j = parse_judge_response('{"correct": true, "confidence": 1.5, "reasoning": ""}')
    assert j is not None and j.confidence == 1.0
    j2 = parse_judge_response('{"correct": true, "confidence": -0.5, "reasoning": ""}')
    assert j2 is not None and j2.confidence == 0.0


def test_parse_returns_none_on_garbage():
    assert parse_judge_response("not json at all") is None
    assert parse_judge_response("{ malformed") is None


def test_parse_returns_none_when_required_key_missing():
    # No 'confidence' key
    assert parse_judge_response('{"correct": true, "reasoning": ""}') is None
    # No 'correct' key
    assert parse_judge_response('{"confidence": 0.9, "reasoning": ""}') is None


def test_parse_handles_non_dict_top_level():
    # Top-level array, not an object
    assert parse_judge_response("[1,2,3]") is None


def test_build_user_prompt_carries_question_and_answer():
    p = build_user_prompt("What is the capital of France?", "Paris.")
    assert "What is the capital of France?" in p
    assert "Paris." in p
    assert "JSON" in p


def test_build_triple_prompt_de_underscores_for_judge():
    """The clean-form helper renders snake_case triples as spaced prose so the
    judge doesn't reject based on form (e.g. 'sandro_botticelli' vs 'Sandro Botticelli')."""
    from rain.feedback.ollama_judge import build_triple_prompt

    q, a = build_triple_prompt("albert_einstein", "born_in", "ulm_germany")
    assert "_" not in q and "_" not in a
    assert q == "What is the born in of albert einstein?"
    assert a == "The born in of albert einstein is ulm germany."
