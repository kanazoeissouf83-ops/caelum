"""Task 2: questions with personal or time-bound data never touch the cache."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

import caelum
from caelum import AsyncOpenAI, OpenAI
from caelum.cache import EmbeddingResult
from caelum.exclusions import CATEGORIES, ExclusionPolicy
from tests.conftest import FakeServer

MUST_EXCLUDE: dict[str, list[str]] = {
    "order": [
        "Où est ma commande n° 4471 ?",
        "Statut de #4471 svp",
        "Mon colis AB123456 n'est pas arrivé",
        "Where is order 88213?",
        "Suivi du colis RR123456789FR",
        "Où en est ma commande 12 ?",  # short number, kept near the reference word
        "statut de 4471 stp",
        "ref: 88",
    ],
    "phone": [
        "Appelez-moi au +226 70 12 34 56",
        "Mon numéro : 70 12 34 56",
        "Call me on +233 24 412 3456",
        "joignable au 0244123456",
        "Contact : 00225 07 08 09 10 11",
        "tel 024-412-3456",
    ],
    "email": [
        "Écrivez à awa.kone@gmail.com",
        "my email is john.doe+shop@yahoo.fr",
        "contact: a@b.co",
    ],
    "amount": [
        "J'ai payé 15 000 FCFA",
        "Remboursez-moi 5000 F",
        "It costs $25",
        "200 GHS de frais ?",
        "15 000 francs pour la livraison ?",
        "j'ai versé 12 500 XOF",
    ],
    "datetime": [
        "Livrez-vous demain ?",
        "Vous êtes ouverts ce soir ?",
        "Livraison avant 17h possible ?",
        "Je passe lundi prochain",
        "Livrez le 12 ?",
        "Can you deliver tomorrow?",
        "Livraison le 3 décembre ?",
        "rendez-vous à 14:30",
    ],
    "payment_identity": [
        "Ma carte 4970 1012 3456 7890 a été débitée",
        "IBAN BF12 3456 7890 1234 5678 9012",
        "mon mot de passe est Kadi2026!",
        "CNIB numéro B12345678",
        "password: hunter22",
        "voici ma clé sk-proj-abcd1234EFGH5678ijkl",  # OpenAI-style API key
        "mon token ghp_abcdefghij1234567890ABCDEF",  # GitHub token
        "Authorization: Bearer abcdef1234567890xyzLMNOP",  # bearer token
        "le jeton est eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF",  # JWT
        "api_key = 9f8e7d6c5b4a3210",  # key assignment
    ],
}

NORMAL = [
    "Quels sont vos horaires ?",
    "Livrez-vous à Bobo-Dioulasso ?",
    "Puis-je payer par Orange Money ?",
    "Quel est le délai de livraison ?",
    "Comment suivre ma commande ?",
    "Comment changer mon mot de passe ?",
    "Avez-vous des pagnes wax ?",
    "Acceptez-vous les cartes Visa ?",
    "Vous ouvrez le dimanche ?",
    "Quelle est votre politique de retour ?",
    "Où se trouve votre boutique à Ouagadougou ?",
    "Faites-vous la livraison à Accra ?",
    "Combien coûte la livraison ?",
    "Livrez-vous 24h/24 ?",
    "Vous livrez en 48h ?",
    "Do you ship to Lagos?",
    "Vendez-vous des iPhone 15 ?",
    "Avez-vous la robe en taille 42 ?",
    "Comment obtenir une clé API ?",  # no value pasted → still cacheable
    "Où est ma commande ?",  # no number → still cacheable
    "Je voudrais commander 2 articles",  # small quantity, not an order number
]


def flat(examples: dict[str, list[str]]) -> list[tuple[str, str]]:
    return [(category, text) for category, texts in examples.items() for text in texts]


# -- detection ---------------------------------------------------------------------------


def test_every_category_has_at_least_three_examples() -> None:
    assert set(MUST_EXCLUDE) == set(CATEGORIES)
    assert all(len(texts) >= 3 for texts in MUST_EXCLUDE.values())
    assert len(NORMAL) >= 10


@pytest.mark.parametrize(("category", "text"), flat(MUST_EXCLUDE))
def test_personal_or_time_bound_questions_are_excluded(category: str, text: str) -> None:
    assert ExclusionPolicy().check(text, None) == category


@pytest.mark.parametrize("text", NORMAL)
def test_ordinary_questions_stay_cacheable(text: str) -> None:
    assert ExclusionPolicy().check(text, None) is None


# -- end to end ---------------------------------------------------------------------------


class Embedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def model(self) -> str:
        return "exclusion-embedder"

    def embed(self, text: str) -> EmbeddingResult:
        self.calls.append(text)
        return EmbeddingResult(vector=[1.0, 0.0], tokens=1)


def client(server: FakeServer, tmp_path: Path, **kwargs: Any) -> OpenAI:
    options: dict[str, Any] = {
        "cache": True,
        "cache_path": tmp_path / "excl.sqlite",
        "cache_admission": 1,
        **kwargs,
    }
    return OpenAI(api_key="sk-test", http_client=server.http_client(), **options)


def ask(c: Any, question: str) -> Any:
    return c.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": question}]
    )


def test_an_excluded_question_always_goes_to_the_ai(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path)
    for _ in range(3):
        assert not caelum.cache_hit(ask(c, "Où est ma commande n° 4471 ?"))
    assert server.calls == 3
    ask(c, "Quels sont vos horaires ?")
    assert caelum.cache_hit(ask(c, "Quels sont vos horaires ?"))  # not over-excluded
    summary = c.metrics.summary()
    assert summary["exclusions"] == {"order": 3}
    assert summary["excluded_requests"] == 3


def test_the_text_of_an_excluded_question_is_kept_nowhere(
    server: FakeServer, tmp_path: Path
) -> None:
    embedder = Embedder()
    secret = "Mon numéro est le +226 70 12 34 56, rappelez-moi"
    c = client(
        server,
        tmp_path,
        cache_admission=2,
        semantic_cache=True,
        semantic_embedder=embedder,
        semantic_verifier=False,
    )
    ask(c, secret)
    ask(c, secret)
    state = c.__dict__["_caelum_state"]
    state.exact_cache.store.flush()
    # Not in the cache files (exact and semantic share the file).
    for file in tmp_path.glob("excl.sqlite*"):
        data = file.read_bytes()
        assert b"70 12 34 56" not in data
        assert secret.encode("utf-8") not in data
    # Not in the admission candidates, not sent to the embedding API.
    assert state.admission.pending == 0
    assert embedder.calls == []
    # Not in the statistics: only the category is counted.
    summary = json.dumps(c.metrics.summary(), ensure_ascii=False)
    assert "70 12 34 56" not in summary
    assert c.metrics.summary()["exclusions"] == {"phone": 2}


def test_the_async_client_applies_the_same_rule(server: FakeServer, tmp_path: Path) -> None:
    c = AsyncOpenAI(
        api_key="sk-test",
        http_client=server.async_http_client(),
        cache=True,
        cache_path=tmp_path / "excl-async.sqlite",
        cache_admission=1,
    )

    async def scenario() -> list[bool]:
        hits: list[bool] = []
        for question in ["Livrez-vous demain ?"] * 2 + ["Quels sont vos horaires ?"] * 2:
            response = await c.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": question}]
            )
            hits.append(caelum.cache_hit(response))
        return hits

    assert asyncio.run(scenario()) == [False, False, False, True]
    assert c.metrics.summary()["exclusions"] == {"datetime": 2}


# -- settings ------------------------------------------------------------------------------


def test_a_category_can_be_switched_off(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, cache_exclusions={"amount": False})
    ask(c, "La livraison coûte 1 000 FCFA ?")
    assert caelum.cache_hit(ask(c, "La livraison coûte 1 000 FCFA ?"))


def test_categories_can_be_set_per_customer(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, cache_customer_exclusions={"acme": {"datetime": False}})
    with caelum.customer("acme"):
        ask(c, "Livrez-vous demain ?")
        assert caelum.cache_hit(ask(c, "Livrez-vous demain ?"))
    with caelum.customer("other"):
        ask(c, "Livrez-vous demain ?")
        assert not caelum.cache_hit(ask(c, "Livrez-vous demain ?"))


def test_custom_patterns_exclude_questions(server: FakeServer, tmp_path: Path) -> None:
    c = client(
        server,
        tmp_path,
        cache_exclude_patterns=[r"\bclient\s+\d+"],
        cache_customer_exclusions={"acme": {"patterns": [r"\bmatricule\b"]}},
    )
    for _ in range(2):
        assert not caelum.cache_hit(ask(c, "Fiche du client 42 ?"))
    with caelum.customer("acme"):
        for _ in range(2):
            assert not caelum.cache_hit(ask(c, "Quel est mon matricule ?"))
    assert c.metrics.summary()["exclusions"] == {"custom": 4}


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"cache_exclude_patterns": ["(unclosed"]}, "invalid regular expression"),
        ({"cache_exclude_patterns": "not a list"}, "list of regular expressions"),
        (
            {"cache_customer_exclusions": {"acme": {"patterns": ["[a-"]}}},
            "invalid regular expression",
        ),
        ({"cache_exclusions": {"addresses": False}}, "unknown category"),
        ({"cache_exclusions": {"amount": "no"}}, "True or False"),
        ({"cache_exclude_patterns": ["a" * 600]}, "longer than"),
    ],
)
def test_invalid_settings_are_refused_clearly(
    server: FakeServer, tmp_path: Path, options: dict[str, Any], message: str
) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        client(server, tmp_path, **options)
    assert list(tmp_path.glob("excl.sqlite*")) == []  # refused before any cache file is opened


def test_exclusions_do_not_apply_without_a_cache(server: FakeServer) -> None:
    c = OpenAI(api_key="sk-test", http_client=server.http_client())
    ask(c, "Livrez-vous demain ?")
    assert c.metrics.summary()["exclusions"] == {}
