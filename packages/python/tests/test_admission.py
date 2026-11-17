"""Task 7: frequency admission, context in cache keys, no cache for conversations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import caelum
from caelum import OpenAI
from caelum.cache import EmbeddingResult, admission
from caelum.cache.admission import AdmissionFilter
from caelum.context import context_fingerprint, depends_on_conversation, detect_language
from tests.conftest import FakeServer

BOBO = "Livrez-vous à Bobo ?"
OUAGA = "et à Ouaga ?"


def client(server: FakeServer, tmp_path: Path, **kwargs: Any) -> OpenAI:
    return OpenAI(
        api_key="sk-test",
        http_client=server.http_client(),
        cache=True,
        cache_path=tmp_path / "admission.sqlite",
        **kwargs,
    )


def ask(c: OpenAI, question: str, **kwargs: Any) -> Any:
    options: dict[str, Any] = {"model": "gpt-4o", **kwargs}
    messages: Any = options.pop("messages", [{"role": "user", "content": question}])
    create: Any = c.chat.completions.create
    return create(messages=messages, **options)


def paid_calls(server: FakeServer) -> int:
    return len([r for r in server.requests if r["path"].endswith("/chat/completions")])


# -- 7.1 frequency admission -------------------------------------------------------------


def test_n1_caches_at_the_first_answer(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, cache_admission=1)
    ask(c, "Vos horaires ?")
    assert caelum.cache_hit(ask(c, "Vos horaires ?"))
    assert paid_calls(server) == 1


def test_n2_savings_start_at_the_third_request(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, cache_admission=2)
    first, second, third = (ask(c, "Vos horaires ?") for _ in range(3))
    assert not caelum.cache_hit(first)
    assert not caelum.cache_hit(second)
    assert caelum.cache_hit(third)
    assert paid_calls(server) == 2


def test_n3_needs_three_answers(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, cache_admission=3)
    hits = [caelum.cache_hit(ask(c, "Vos horaires ?")) for _ in range(5)]
    assert hits == [False, False, False, True, True]


def test_default_admission_is_two(server: FakeServer, tmp_path: Path, monkeypatch: Any) -> None:
    assert AdmissionFilter().n == 2  # the shipped default
    monkeypatch.setattr(admission, "DEFAULT_ADMISSION", 2)  # undo the test-wide N=1
    c = client(server, tmp_path)
    hits = [caelum.cache_hit(ask(c, "Vos horaires ?")) for _ in range(3)]
    assert hits == [False, False, True]


def test_a_question_asked_once_is_never_stored(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, cache_admission=2)
    ask(c, "Avez-vous la robe bleue de la vitrine ?")
    ask(c, "Vos horaires ?")
    state = c.__dict__["_caelum_state"]
    assert len(state.exact_cache.store) == 0
    summary = c.metrics.summary()
    assert summary["admission_pending"] == 2
    assert summary["admission_refused"] == 2
    assert summary["admission_admitted"] == 0


def test_statistics_count_admissions(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, cache_admission=2)
    for _ in range(3):
        ask(c, "Vos horaires ?")
    summary = c.metrics.summary()
    assert summary["admission_admitted"] == 1
    assert summary["admission_refused"] == 1
    assert summary["admission_pending"] == 0


def test_admission_can_be_set_per_customer(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, cache_admission=2, cache_customer_admission={"vip": 1})
    with caelum.customer("vip"):
        ask(c, "Vos horaires ?")
        assert caelum.cache_hit(ask(c, "Vos horaires ?"))  # N=1 for this customer
    with caelum.customer("other"):
        ask(c, "Vos horaires ?")
        assert not caelum.cache_hit(ask(c, "Vos horaires ?"))  # N=2


class _Embedder:
    vectors = {
        "How do I return an item?": [1.0, 0.0],
        "How can I send back a product?": [0.99, 0.141],
    }

    @property
    def model(self) -> str:
        return "admission-embedder"

    def embed(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(vector=self.vectors.get(text, [0.0, 1.0]), tokens=1)


def test_questions_with_the_same_meaning_count_together(server: FakeServer, tmp_path: Path) -> None:
    c = OpenAI(
        api_key="sk-test",
        http_client=server.http_client(),
        semantic_cache=True,
        semantic_embedder=_Embedder(),
        semantic_verifier=False,
        cache_path=tmp_path / "sem.sqlite",
        cache_admission=2,
    )
    assert not caelum.cache_hit(ask(c, "How do I return an item?"))  # 1st: pending
    assert not caelum.cache_hit(ask(c, "How can I send back a product?"))  # 2nd: admitted
    third = ask(c, "How do I return an item?")
    assert caelum.cache_type(third) == "semantic"
    assert paid_calls(server) == 2


def test_candidates_expire_and_are_bounded() -> None:
    now = [0.0]
    f = AdmissionFilter(n=2, window=60, max_candidates=3, clock=lambda: now[0])
    for key in ("a", "b", "c", "d"):
        assert f.observe(key, None) is False
    assert f.pending == 3  # "a", the oldest, was dropped
    assert f.observe("a", None) is False  # forgotten: counted again from 1
    now[0] = 61
    assert f.pending == 0  # all expired
    assert f.observe("b", None) is False


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
def test_invalid_admission_is_refused(server: FakeServer, tmp_path: Path, bad: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        client(server, tmp_path, cache_admission=bad)
    with pytest.raises((ValueError, TypeError)):
        client(server, tmp_path, cache_customer_admission={"acme": bad})


# -- 7.2 context in the cache key ---------------------------------------------------------


def test_another_customer_never_gets_the_answer(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, cache_admission=1)
    with caelum.customer("boutique-a"):
        ask(c, "Vos horaires ?")
    with caelum.customer("boutique-b"):
        assert not caelum.cache_hit(ask(c, "Vos horaires ?"))
    assert paid_calls(server) == 2


def test_another_language_never_gets_the_answer(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, cache_admission=1)
    with caelum.language("fr"):
        ask(c, "OK ?")
    with caelum.language("en"):
        assert not caelum.cache_hit(ask(c, "OK ?"))
    with caelum.language("fr"):
        assert caelum.cache_hit(ask(c, "OK ?"))


def test_another_model_never_gets_the_answer(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, cache_admission=1)
    ask(c, "Vos horaires ?", model="gpt-4o")
    assert not caelum.cache_hit(ask(c, "Vos horaires ?", model="gpt-4o-mini"))


def test_other_system_instructions_never_get_the_answer(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, cache_admission=1)

    def with_system(system: str) -> Any:
        return ask(
            c,
            "",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "Vos horaires ?"},
            ],
        )

    with_system("Tu es l'assistant de la boutique Kadi Mode.")
    assert not caelum.cache_hit(with_system("Tu es l'assistant de la boutique Faso Tech."))


REQUEST: dict[str, Any] = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "Boutique Kadi Mode"},
        {"role": "user", "content": "Vos horaires ?"},
    ],
}


def test_fingerprint_contains_the_customer() -> None:
    with caelum.customer("a"):
        first = context_fingerprint(REQUEST, "gpt-4o")
    with caelum.customer("b"):
        assert context_fingerprint(REQUEST, "gpt-4o") != first


def test_fingerprint_contains_the_language() -> None:
    with caelum.language("fr"):
        first = context_fingerprint(REQUEST, "gpt-4o")
    with caelum.language("en"):
        assert context_fingerprint(REQUEST, "gpt-4o") != first


def test_fingerprint_contains_the_model() -> None:
    assert context_fingerprint(REQUEST, "gpt-4o") != context_fingerprint(REQUEST, "gpt-4o-mini")


def test_fingerprint_contains_the_system_instructions() -> None:
    other = {
        **REQUEST,
        "messages": [{"role": "system", "content": "Boutique Faso Tech"}, REQUEST["messages"][1]],
    }
    assert context_fingerprint(REQUEST, "gpt-4o") != context_fingerprint(other, "gpt-4o")
    anthropic = {"model": "claude", "system": "A", "messages": [REQUEST["messages"][1]]}
    assert context_fingerprint(anthropic, "claude") != context_fingerprint(
        {**anthropic, "system": "B"}, "claude"
    )


def test_language_detection_separates_translations() -> None:
    assert detect_language("Quels sont vos horaires ?") == "fr"
    assert detect_language("What are your opening hours?") == "en"
    assert detect_language("¿Cuál es su horario?") == "es"
    with pytest.raises(ValueError):
        with caelum.language("français"):
            pass


# -- 7.3 conversations ---------------------------------------------------------------------


def conversation() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": BOBO},
        {"role": "assistant", "content": "Oui, en 48 heures."},
        {"role": "user", "content": OUAGA},
    ]


def test_a_follow_up_question_is_never_cached(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, cache_admission=1)
    ask(c, BOBO)
    assert caelum.cache_hit(ask(c, BOBO))  # a standalone question is cached
    for _ in range(3):
        assert not caelum.cache_hit(ask(c, "", messages=conversation()))
    assert paid_calls(server) == 4
    assert len(c.__dict__["_caelum_state"].exact_cache.store) == 1  # only "Bobo"


def test_conversations_are_detected_for_every_sdk_shape() -> None:
    assert depends_on_conversation({"messages": conversation()})
    assert depends_on_conversation(
        {"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]}
    )
    assert depends_on_conversation({"contents": [{"role": "user"}, {"role": "model"}]})  # Gemini
    assert not depends_on_conversation({"messages": [{"role": "user", "content": BOBO}]})
    assert not depends_on_conversation(
        {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": BOBO}]}
    )


def test_cache_bypass_argument(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, cache_admission=1)
    ask(c, BOBO)
    response = ask(c, BOBO, cache_bypass=True)
    assert not caelum.cache_hit(response)
    assert "cache_bypass" not in server.requests[-1]["body"]  # never sent to the provider
    with pytest.raises(TypeError):
        ask(c, BOBO, cache_bypass="yes")


def test_cache_bypass_block(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, cache_admission=1)
    with caelum.cache_bypass():
        ask(c, BOBO)
        assert not caelum.cache_hit(ask(c, BOBO))
    assert len(c.__dict__["_caelum_state"].exact_cache.store) == 0
