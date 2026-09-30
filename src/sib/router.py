"""Distilled router (FR-07): small students trained on a teacher's soft labels.

Two students: intent (ask / draft / summarize / action) and difficulty (can a cheap model answer
this?). `choose_models` then implements margin-aware routing: per request class, the cheapest
model whose measured quality meets the target.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from sib.embeddings import HashingEmbedder, tokenize_words

INTENTS = ["ask", "draft", "summarize", "action"]

_TEACHER_CUES: dict[str, list[str]] = {
    "ask": ["what", "how", "when", "where", "which", "who", "whom", "many", "much"],
    "draft": ["draft", "write", "prepare", "compose", "redraft"],
    "summarize": ["summarise", "summarize", "summary", "overview", "recap", "tldr", "brief", "digest"],
    "action": ["email", "send", "schedule", "calendar", "book", "forward", "upload", "invite", "remind"],
}

_TEMPLATES: dict[str, list[str]] = {
    "draft": ["Draft a {doc} to {who} about {topic}", "Write a {doc} for {who}", "Prepare a {doc} regarding {topic}",
              "Please compose a {doc} on {topic} for {who}", "Redraft the {doc} in our house style"],
    "summarize": ["Summarise the {topic} documents", "Give me a summary of the {topic}", "Quick recap of {topic} please",
                  "Overview of where we are on {topic}", "TLDR of the correspondence on {topic}"],
    "action": ["Email {who} the latest {doc}", "Schedule a call with {who} on Friday", "Send the {doc} to {who}",
               "Book a calendar slot to discuss {topic}", "Forward the {doc} to {who} and remind me Monday"],
}
_FILL = {
    "doc": ["demand letter", "termination notice", "client update", "reply", "engagement letter"],
    "who": ["opposing counsel", "the client", "the partner", "the court registry"],
    "topic": ["the lease dispute", "the supply contract", "unpaid invoices", "the employment claim", "arbitration"],
}


def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = z / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def teacher_intent_probs(text: str, temperature: float = 2.0) -> np.ndarray:
    """Stand-in for a frontier model labelling requests. Returns soft targets over INTENTS."""
    words = set(tokenize_words(text))
    scores = np.array([0.8 + 2.5 * len(words & set(_TEACHER_CUES["ask"]))]
                      + [2.5 * len(words & set(_TEACHER_CUES[i])) for i in INTENTS[1:]])
    return softmax(scores, temperature)


def synthetic_requests(questions: list[str], n_per_intent: int = 60, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out = list(questions)
    for intent in INTENTS[1:]:
        for _ in range(n_per_intent):
            template = rng.choice(_TEMPLATES[intent])
            out.append(template.format(**{k: rng.choice(v) for k, v in _FILL.items()}))
    rng.shuffle(out)
    return out


class SoftmaxStudent:
    """Linear softmax classifier trained with the knowledge-distillation loss.

    loss = alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * CE(hard label, student)
    """

    def __init__(self, n_features: int, n_classes: int) -> None:
        self.W = np.zeros((n_classes, n_features), dtype=np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)

    def logits(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W.T + self.b

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.logits(x))

    def fit(
        self, x: np.ndarray, teacher_probs: np.ndarray, epochs: int = 300, lr: float = 0.5,
        temperature: float = 2.0, alpha: float = 0.7, weight_decay: float = 1e-4,
    ) -> list[float]:
        hard = np.eye(teacher_probs.shape[1], dtype=np.float32)[teacher_probs.argmax(axis=1)]
        # Re-soften the teacher distribution at the distillation temperature.
        soft = softmax(np.log(teacher_probs + 1e-9), temperature)
        losses = []
        for _ in range(epochs):
            z = self.logits(x)
            p_t, p = softmax(z, temperature), softmax(z)
            kl = float((soft * (np.log(soft + 1e-9) - np.log(p_t + 1e-9))).sum(axis=1).mean())
            ce = float(-(hard * np.log(p + 1e-9)).sum(axis=1).mean())
            losses.append(alpha * temperature**2 * kl + (1 - alpha) * ce)
            grad_z = (alpha * temperature * (p_t - soft) + (1 - alpha) * (p - hard)) / len(x)
            self.W -= lr * (grad_z.T @ x + weight_decay * self.W)
            self.b -= lr * grad_z.sum(axis=0)
        return losses


@dataclass
class RouteDecision:
    intent: str
    p_hard: float
    request_class: str  # e.g. "ask:easy"
    model: str


class DistilledRouter:
    """Intent is read from the request text. Difficulty is read from retrieval features only
    (how much of the question the retrieved passages cover lexically): unlike word features,
    those transfer to phrasings the student never saw.
    """

    EXTRA = 3

    def __init__(self, embedder: HashingEmbedder | None = None, hard_threshold: float = 0.5) -> None:
        self.embedder = embedder or HashingEmbedder(128)
        self.intent = SoftmaxStudent(self.embedder.dim, len(INTENTS))
        self.difficulty = SoftmaxStudent(self.EXTRA, 2)
        self.hard_threshold = hard_threshold
        self.table: dict[str, str] = {}
        self.report: dict[str, object] = {}

    def features(self, text: str, retrieval: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        return np.asarray(retrieval, dtype=np.float32)

    def fit_intent(self, texts: list[str], teacher_probs: np.ndarray) -> list[float]:
        return self.intent.fit(self.embedder.embed(texts), teacher_probs)

    def fit_difficulty(self, features: np.ndarray, p_hard: np.ndarray) -> list[float]:
        return self.difficulty.fit(features, np.stack([1 - p_hard, p_hard], axis=1))

    def classify(self, text: str, retrieval: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[str, float]:
        intent = INTENTS[int(self.intent.predict_proba(self.embedder.embed_one(text)[None])[0].argmax())]
        p_hard = float(self.difficulty.predict_proba(self.features(text, retrieval)[None])[0, 1])
        return intent, p_hard

    def route(self, text: str, retrieval: tuple[float, float, float], default_model: str,
              intent: str | None = None) -> RouteDecision:
        """Pass `intent` when the endpoint already fixes it; only difficulty is predicted then."""
        predicted, p_hard = self.classify(text, retrieval)
        intent = intent or predicted
        cls = f"{intent}:{'hard' if p_hard >= self.hard_threshold else 'easy'}"
        return RouteDecision(intent, p_hard, cls, self.table.get(cls, default_model))


def choose_models(
    quality: dict[str, dict[str, float]], cost: dict[str, dict[str, Decimal]], target: dict[str, float], fallback: str,
) -> dict[str, str]:
    """Margin-aware routing: per class, the cheapest model with measured quality >= target.

    `cost[cls][model]` must include the cost of escalations that model triggers on that class.
    """
    table = {}
    for cls, scores in quality.items():
        ok = [m for m, s in scores.items() if s >= target[cls]]
        table[cls] = min(ok, key=lambda m: cost[cls][m]) if ok else fallback
    return table
