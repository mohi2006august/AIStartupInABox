from decimal import Decimal

import numpy as np

from sib.corpus import build_corpus
from sib.router import INTENTS, DistilledRouter, choose_models, synthetic_requests, teacher_intent_probs


def test_teacher_gives_soft_targets():
    probs = teacher_intent_probs("Draft a termination notice and email it to the client")
    assert abs(probs.sum() - 1) < 1e-6
    assert INTENTS[int(probs.argmax())] in ("draft", "action")
    assert sorted(probs)[-2] > 0.15  # the runner-up keeps real mass: that is the dark knowledge


def test_student_matches_teacher_on_unseen_requests():
    corpus = build_corpus()
    train = synthetic_requests([p[0] for p in corpus.training_pairs], seed=0)
    router = DistilledRouter()
    losses = router.fit_intent(train, np.stack([teacher_intent_probs(t) for t in train]))
    assert losses[-1] < losses[0]
    test = synthetic_requests([q.question for q in corpus.questions], n_per_intent=40, seed=99)
    agree = np.mean([router.classify(t)[0] == INTENTS[int(teacher_intent_probs(t).argmax())] for t in test])
    assert agree >= 0.9


def test_difficulty_student_learns_from_retrieval_features():
    rng = np.random.default_rng(0)
    router = DistilledRouter()
    texts = [f"question {i}" for i in range(200)]
    coverage = rng.uniform(0, 1, 200)
    feats = np.stack([router.features(t, (c, 0.0, 0.0)) for t, c in zip(texts, coverage)])
    router.fit_difficulty(feats, (coverage < 0.5).astype(np.float32))
    assert router.classify("question 1", (0.9, 0.0, 0.0))[1] < 0.5 < router.classify("question 1", (0.1, 0.0, 0.0))[1]


def test_choose_models_picks_cheapest_that_meets_target():
    quality = {"ask:easy": {"big": 99.0, "small": 98.0}, "ask:hard": {"big": 97.0, "small": 70.0}}
    cost = {"big": Decimal("0.02"), "small": Decimal("0.001")}
    table = choose_models(quality, cost, {"ask:easy": 97.0, "ask:hard": 95.0}, fallback="big")
    assert table == {"ask:easy": "small", "ask:hard": "big"}
    assert choose_models(quality, cost, {"ask:easy": 99.5, "ask:hard": 99.5}, fallback="big")["ask:easy"] == "big"
