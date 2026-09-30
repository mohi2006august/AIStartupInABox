"""Drafting model (FR-06): SFT base + per-firm LoRA adapters with an overfitting guard.

The model is deliberately small but real: a linear softmax layer that picks a drafting variant for
each style slot (salutation, obligation verb, date format, ...) given the document type and
audience. The base is fine-tuned on public drafting; each firm gets a low-rank adapter
(dW = B @ A) trained on its own few hundred examples. Small data overfits, so adapters are only
accepted if they beat the base on a held-out split and do not memorise planted canaries.
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from datetime import date

import numpy as np

SLOTS: dict[str, list[str]] = {
    "reference": ["In the matter of {matter}", "Re: {matter}", "Subject: {matter}"],
    "date_format": ["formal", "day_first", "month_first"],
    "salutation": ["Dear Sirs,", "Dear Counsel,", "Hello,"],
    "opening": ["We write on behalf of our client, {client}.", "We act for {client} in this matter.",
                "I am writing on behalf of {client}."],
    "defined_term": [' (the "Agreement")', " (the Agreement)", ""],
    "demand_lead": ["We hereby call upon you to", "You are required to", "Please"],
    "deadline": ["within {n} days of the date hereof", "within {n} days of this letter", "in the next {n} days"],
    "obligation": ["shall", "must", "will"],
    "reservation": ["All rights are expressly reserved.", "Our client reserves all of its rights.",
                    "We hope to resolve this amicably."],
    "signoff": ["Yours faithfully,", "Yours sincerely,", "Kind regards,"],
}
SLOT_NAMES = list(SLOTS)
DOC_TYPES = ["demand_letter", "termination_notice", "client_update"]
AUDIENCES = ["court", "opposing_counsel", "client"]
N_VARIANTS = 3
VOCAB = len(SLOT_NAMES) * N_VARIANTS
DIM = 256

Example = tuple[str, str, str, int]  # (slot, doc_type, audience, variant index)


def featurize(slot: str, doc_type: str, audience: str) -> np.ndarray:
    x = np.zeros(DIM, dtype=np.float32)
    for feat in (f"s:{slot}", f"sd:{slot}|{doc_type}", f"sa:{slot}|{audience}", f"sda:{slot}|{doc_type}|{audience}"):
        x[zlib.crc32(feat.encode()) % DIM] += 1.0
    return x


def _matrix(examples: list[Example]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.stack([featurize(s, d, a) for s, d, a, _ in examples])
    slot_idx = np.array([SLOT_NAMES.index(s) for s, _, _, _ in examples])
    y = slot_idx * N_VARIANTS + np.array([v for _, _, _, v in examples])
    mask = np.full((len(examples), VOCAB), -1e9, dtype=np.float32)
    for row, si in enumerate(slot_idx):
        mask[row, si * N_VARIANTS : (si + 1) * N_VARIANTS] = 0.0
    return x, y, mask


def public_style(slot: str, doc_type: str, audience: str) -> int:
    """Generic legal drafting: most formal for courts, plainest for clients."""
    return AUDIENCES.index(audience)


@dataclass
class FirmStyle:
    """House style: some slots are fixed firm-wide, the rest follow general practice."""

    tenant_id: str
    house: dict[str, int] = field(default_factory=dict)
    overrides: dict[tuple[str, str], int] = field(default_factory=dict)

    @classmethod
    def generate(cls, tenant_id: str) -> "FirmStyle":
        rng = random.Random(zlib.crc32(tenant_id.encode()))
        house = {s: rng.randrange(N_VARIANTS) for s in SLOT_NAMES if rng.random() < 0.6}
        overrides = {(rng.choice(SLOT_NAMES), rng.choice(DOC_TYPES)): rng.randrange(N_VARIANTS) for _ in range(3)}
        return cls(tenant_id, house, overrides)

    def preferred(self, slot: str, doc_type: str, audience: str) -> int:
        if (slot, doc_type) in self.overrides:
            return self.overrides[(slot, doc_type)]
        return self.house.get(slot, public_style(slot, doc_type, audience))


def sample_examples(n: int, seed: int, style: FirmStyle | None = None, noise: float = 0.1) -> list[Example]:
    rng = random.Random(seed)
    out: list[Example] = []
    for _ in range(n):
        s, d, a = rng.choice(SLOT_NAMES), rng.choice(DOC_TYPES), rng.choice(AUDIENCES)
        v = style.preferred(s, d, a) if style else public_style(s, d, a)
        out.append((s, d, a, rng.randrange(N_VARIANTS) if rng.random() < noise else v))
    return out


def all_contexts() -> list[tuple[str, str, str]]:
    return [(s, d, a) for s in SLOT_NAMES for d in DOC_TYPES for a in AUDIENCES]


@dataclass
class LoRAAdapter:
    A: np.ndarray  # (rank, DIM)
    B: np.ndarray  # (VOCAB, rank)
    alpha: float

    @property
    def rank(self) -> int:
        return self.A.shape[0]

    @property
    def delta(self) -> np.ndarray:
        return (self.alpha / self.rank) * (self.B @ self.A)

    @property
    def n_params(self) -> int:
        return self.A.size + self.B.size


class StyleModel:
    def __init__(self) -> None:
        self.W = np.zeros((VOCAB, DIM), dtype=np.float32)

    def _probs(self, x: np.ndarray, mask: np.ndarray, adapter: LoRAAdapter | None) -> np.ndarray:
        w = self.W + adapter.delta if adapter else self.W
        z = x @ w.T + mask
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def loss(self, examples: list[Example], adapter: LoRAAdapter | None = None) -> float:
        x, y, mask = _matrix(examples)
        p = self._probs(x, mask, adapter)
        return float(-np.log(p[np.arange(len(y)), y] + 1e-9).mean())

    def accuracy(self, examples: list[Example], adapter: LoRAAdapter | None = None) -> float:
        x, y, mask = _matrix(examples)
        return float((self._probs(x, mask, adapter).argmax(axis=1) == y).mean())

    def predict(self, slot: str, doc_type: str, audience: str, adapter: LoRAAdapter | None = None) -> int:
        x, _, mask = _matrix([(slot, doc_type, audience, 0)])
        return int(self._probs(x, mask, adapter).argmax(axis=1)[0]) % N_VARIANTS

    def train_base(self, examples: list[Example], epochs: int = 200, lr: float = 0.5) -> list[float]:
        """Supervised fine-tuning of the full weight matrix on public drafting examples."""
        x, y, mask = _matrix(examples)
        onehot = np.eye(VOCAB, dtype=np.float32)[y]
        losses = []
        for _ in range(epochs):
            p = self._probs(x, mask, None)
            losses.append(float(-np.log(p[np.arange(len(y)), y] + 1e-9).mean()))
            self.W -= lr * ((p - onehot).T @ x) / len(y)
        return losses


def train_lora(
    model: StyleModel, train: list[Example], val: list[Example], rank: int, epochs: int = 400, lr: float = 0.3,
    patience: int | None = 20, seed: int = 0,
) -> tuple[LoRAAdapter, dict[str, list[float]]]:
    """Train only A and B; the base W stays frozen. Early-stops on val loss when patience is set."""
    rng = np.random.default_rng(seed)
    adapter = LoRAAdapter(
        A=rng.normal(0, 1 / np.sqrt(DIM), (rank, DIM)).astype(np.float32),
        B=np.zeros((VOCAB, rank), dtype=np.float32),
        alpha=2.0 * rank,
    )
    x, y, mask = _matrix(train)
    onehot = np.eye(VOCAB, dtype=np.float32)[y]
    scale = adapter.alpha / rank
    history: dict[str, list[float]] = {"train": [], "val": []}
    best, best_val, stale = (adapter.A.copy(), adapter.B.copy()), float("inf"), 0
    for _ in range(epochs):
        grad_w = ((model._probs(x, mask, adapter) - onehot).T @ x) / len(y)
        grad_b, grad_a = scale * grad_w @ adapter.A.T, scale * adapter.B.T @ grad_w
        adapter.B -= lr * grad_b
        adapter.A -= lr * grad_a
        history["train"].append(model.loss(train, adapter))
        history["val"].append(model.loss(val, adapter))
        if history["val"][-1] < best_val - 1e-4:
            best, best_val, stale = (adapter.A.copy(), adapter.B.copy()), history["val"][-1], 0
        else:
            stale += 1
            if patience is not None and stale >= patience:
                break
    if patience is not None:
        adapter.A, adapter.B = best
    return adapter, history


MAX_GUARDED_RANK = 16


@dataclass
class AdapterReport:
    tenant_id: str
    rank: int
    epochs_run: int
    train_acc: float
    holdout_acc: float
    base_holdout_acc: float
    gain_pts: float
    generalisation_gap_pts: float
    canary_memorised: float
    accepted: bool
    reason: str


def fit_firm_adapter(
    model: StyleModel, tenant_id: str, examples: list[Example], rank: int = 8, min_gain_pts: float = 2.0,
    guard: bool = True, n_canaries: int = 4, seed: int = 0,
) -> tuple[LoRAAdapter | None, AdapterReport]:
    """Overfitting guard: capped rank, early stopping, canary check, fall back to base on small gain."""
    if guard and rank > MAX_GUARDED_RANK:
        raise ValueError(f"rank {rank} exceeds the guarded maximum of {MAX_GUARDED_RANK} for small firm datasets")
    rng = random.Random(seed)
    data = list(examples)
    rng.shuffle(data)
    n_val = n_hold = max(len(data) * 15 // 100, 1)
    val, holdout, train = data[:n_val], data[n_val : n_val + n_hold], data[n_val + n_hold :]
    patience = 20 if guard else None
    adapter, history = train_lora(model, train, val, rank, patience=patience, seed=seed)
    train_acc, hold_acc = model.accuracy(train, adapter), model.accuracy(holdout, adapter)
    base_acc = model.accuracy(holdout)
    memorised = canary_exposure(model, adapter, train, val, rank, patience, n_canaries, seed)
    gain = (hold_acc - base_acc) * 100
    accepted, reason = True, "accepted"
    if guard and gain < min_gain_pts:
        accepted, reason = False, f"gain {gain:.1f} pts < {min_gain_pts} pts: fall back to base model"
    elif guard and memorised > 0.5:
        accepted, reason = False, f"memorised {memorised:.0%} of canaries: fall back to base model"
    report = AdapterReport(tenant_id, rank, len(history["train"]), train_acc, hold_acc, base_acc, gain,
                           (train_acc - hold_acc) * 100, memorised, accepted, reason)
    return (adapter if accepted else None), report


def canary_exposure(
    model: StyleModel, clean: LoRAAdapter, train: list[Example], val: list[Example], rank: int,
    patience: int | None, n_canaries: int, seed: int,
) -> float:
    """Differential memorisation probe.

    Plant examples in contexts the training set never covers, labelled with a variant the clean
    adapter does not predict there, then retrain. The share of canaries the probe adapter now
    reproduces is memorisation: nothing else in the data supports those labels.
    """
    rng = random.Random(seed + 1)
    seen = {(s, d, a) for s, d, a, _ in train}
    free = [c for c in all_contexts() if c not in seen] or all_contexts()
    rng.shuffle(free)
    canaries = [(s, d, a, (model.predict(s, d, a, clean) + 1 + rng.randrange(N_VARIANTS - 1)) % N_VARIANTS)
                for s, d, a in free[:n_canaries]]
    probe, _ = train_lora(model, train + canaries, val, rank, patience=patience, seed=seed)
    return float(np.mean([model.predict(s, d, a, probe) == v for s, d, a, v in canaries]))


# --- Rendering -----------------------------------------------------------------------------


@dataclass
class DraftRequest:
    tenant_id: str
    doc_type: str
    audience: str
    client: str
    counterparty: str
    matter: str
    action: str  # e.g. "pay the outstanding rent of INR 240,000"
    days: int = 14
    on: date = date(2026, 9, 30)


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def format_date(d: date, variant: int) -> str:
    month = d.strftime("%B")
    return [f"Dated this {_ordinal(d.day)} day of {month}, {d.year}", f"{d.day} {month} {d.year}",
            f"{month} {d.day}, {d.year}"][variant]


def render_draft(choices: dict[str, int], req: DraftRequest, firm_name: str) -> str:
    t = {slot: SLOTS[slot][choices[slot]] for slot in SLOT_NAMES}
    lines = [
        t["reference"].format(matter=req.matter),
        format_date(req.on, choices["date_format"]),
        "",
        t["salutation"],
        "",
        f"{t['opening'].format(client=req.client)} We refer to the agreement between {req.client} and "
        f"{req.counterparty}{t['defined_term']}.",
        f"{t['demand_lead']} {req.action} {t['deadline'].format(n=req.days)}. Failing this, our client "
        f"{t['obligation']} pursue all available remedies.",
        t["reservation"],
        "",
        t["signoff"],
        firm_name,
    ]
    return "\n".join(lines)


def few_shot_choices(shots: list[Example], doc_type: str, audience: str) -> dict[str, int]:
    """In-context style transfer (what an API model does with examples in the prompt): copy the most
    specific matching example, else fall back to general practice."""
    choices = {}
    for slot in SLOT_NAMES:
        for match in (lambda e: e[:3] == (slot, doc_type, audience), lambda e: e[0] == slot and e[2] == audience,
                      lambda e: e[0] == slot):
            votes = [e[3] for e in shots if match(e)]
            if votes:
                choices[slot] = max(set(votes), key=votes.count)
                break
        else:
            choices[slot] = public_style(slot, doc_type, audience)
    return choices


def model_choices(model: StyleModel, doc_type: str, audience: str, adapter: LoRAAdapter | None) -> dict[str, int]:
    return {slot: model.predict(slot, doc_type, audience, adapter) for slot in SLOT_NAMES}


def style_match(choices: dict[str, int], style: FirmStyle, doc_type: str, audience: str) -> float:
    return float(np.mean([choices[s] == style.preferred(s, doc_type, audience) for s in SLOT_NAMES]))
