from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentlightning.types import PromptTemplate


def test_prompt_template_format_f_string() -> None:
    template = PromptTemplate(template="Hello {name}!", engine="f-string")

    assert template.format(name="World") == "Hello World!"


def test_prompt_template_format_jinja(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyTemplate:
        def __init__(self, source: str) -> None:
            self.source = source

        def render(self, **context: str) -> str:
            result = self.source
            for key, value in context.items():
                result = result.replace(f"{{{{ {key} }}}}", value)
            return result

    monkeypatch.setitem(sys.modules, "jinja2", SimpleNamespace(Template=DummyTemplate))

    template = PromptTemplate(template="Hello {{ name }}!", engine="jinja")

    assert template.format(name="World") == "Hello World!"


def test_prompt_template_format_poml_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, dict[str, object], str]] = []

    def dummy_poml(template: object, context: dict[str, object], format: str) -> dict[str, object]:
        calls.append((template, context, format))
        return {"template": template, "context": context, "format": format}

    monkeypatch.setitem(sys.modules, "poml", SimpleNamespace(poml=dummy_poml))

    template = PromptTemplate(template="<poml>{{ name }}</poml>", engine="poml")

    result = template.format(name="World")

    assert calls == [("<poml>{{ name }}</poml>", {"name": "World"}, "openai_chat")]
    assert result == {"template": "<poml>{{ name }}</poml>", "context": {"name": "World"}, "format": "openai_chat"}


def test_prompt_template_format_poml_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, dict[str, object], str]] = []

    def dummy_poml(template: object, context: dict[str, object], format: str) -> dict[str, object]:
        calls.append((template, context, format))
        return {"template": template, "context": context, "format": format}

    monkeypatch.setitem(sys.modules, "poml", SimpleNamespace(poml=dummy_poml))

    poml_file = tmp_path / "sample.poml"
    poml_file.write_text("<poml>{{ name }}</poml>")

    template = PromptTemplate(template=str(poml_file), engine="poml")

    result = template.format(name="World", _poml_format="raw")

    assert len(calls) == 1
    call_template, context, output_format = calls[0]
    assert isinstance(call_template, Path)
    assert call_template == poml_file
    assert context == {"name": "World"}
    assert output_format == "raw"
    assert result == {"template": poml_file, "context": {"name": "World"}, "format": "raw"}
