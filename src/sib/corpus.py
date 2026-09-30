"""Synthetic law-firm corpus: tenants, matters, documents and an eval set with known answers.

Everything is generated deterministically from a seed so evals and `sib bench` are reproducible.
Documents use formal drafting vocabulary; "hard" questions use colloquial synonyms, which is what
separates strong models (and tuned embeddings) from weak ones.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# canonical drafting term -> colloquial variants (all stemmed, see `stem`)
THESAURUS: dict[str, list[str]] = {
    "lessee": ["tenant", "renter"],
    "lessor": ["landlord", "owner"],
    "lease": ["tenancy"],
    "remit": ["pay", "payment", "settle", "owed"],
    "terminate": ["end", "cancel"],
    "notice": ["warning", "advance"],
    "deposit": ["bond"],
    "maintenance": ["upkeep", "fixing"],
    "annum": ["annual", "year", "yearly"],
    "dispute": ["conflict", "disagreement"],
    "arbitration": ["tribunal", "settled"],
    "seated": ["place", "location"],
    "employee": ["staff", "worker"],
    "remuneration": ["wage", "salary", "earn"],
    "competing": ["rival", "competitor"],
    "cessation": ["leaving", "departure"],
    "disclose": ["reveal", "keep"],
    "confidential": ["secret"],
    "probationary": ["trial"],
    "supplier": ["vendor", "seller"],
    "purchaser": ["buyer", "customer"],
    "deliver": ["ship", "dispatch"],
    "good": ["product", "item"],
    "liability": ["exposure", "responsible"],
    "exceed": ["maximum", "most"],
    "invoice": ["bill"],
    "warrant": ["guarantee"],
    "defect": ["fault"],
}
SYNONYM_TO_CANONICAL: dict[str, str] = {v: k for k, vs in THESAURUS.items() for v in vs}

_WORD = re.compile(r"[a-z0-9]+")


def stem(word: str) -> str:
    w = word.lower()
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def stems(text: str) -> list[str]:
    return [stem(w) for w in _WORD.findall(text.lower().replace("'s", ""))]


_NUM = {
    1: "one", 2: "two", 3: "three", 5: "five", 6: "six", 7: "seven", 12: "twelve", 14: "fourteen",
    18: "eighteen", 21: "twenty-one", 24: "twenty-four", 30: "thirty", 36: "thirty-six", 45: "forty-five",
    60: "sixty", 90: "ninety", 120: "one hundred twenty",
}


def num(n: int) -> str:
    return f"{_NUM[n]} ({n})"


def money(n: int) -> str:
    return f"INR {n:,}"


@dataclass(frozen=True)
class ClauseSpec:
    key: str
    template: str
    easy: str
    hard: str
    hard_train: str  # a different paraphrase, used only to train embeddings / router
    answer_field: str


CLAUSES: dict[str, list[ClauseSpec]] = {
    "lease": [
        ClauseSpec("rent_day", "The Lessee shall remit the monthly rent of {rent} to the Lessor no later than the {day} day of each calendar month.",
                   "By which day of each calendar month shall the Lessee remit the monthly rent?",
                   "When is the tenant's payment owed to the landlord?",
                   "What date must the renter pay the owner by?", "day"),
        ClauseSpec("notice", "Either party may terminate this Lease upon {notice} days written notice to the other party.",
                   "Upon how many days written notice may either party terminate this Lease?",
                   "How much warning is needed to end the tenancy?",
                   "How long in advance must someone cancel the tenancy?", "notice"),
        ClauseSpec("deposit", "The Lessee shall lodge a security deposit of {deposit} with the Lessor upon execution of this Lease.",
                   "What security deposit shall the Lessee lodge with the Lessor?",
                   "How much bond does the renter put down with the owner?",
                   "What upfront bond is the tenant required to hand the landlord?", "deposit"),
        ClauseSpec("repairs", "The Lessor shall carry out structural repairs and the Lessee shall bear interior maintenance costs up to {cap} per annum.",
                   "Up to what amount per annum shall the Lessee bear interior maintenance costs?",
                   "What is the yearly ceiling on upkeep the tenant covers?",
                   "How much fixing expense does the renter absorb each year?", "cap"),
        ClauseSpec("seat", "Any dispute arising under this Lease shall be referred to arbitration seated in {city}.",
                   "Where is arbitration seated for any dispute arising under this Lease?",
                   "In which place are conflicts under the tenancy settled?",
                   "Which location hosts the tribunal for tenancy disagreements?", "city"),
    ],
    "employment": [
        ClauseSpec("salary", "The Employer shall pay the Employee an annual remuneration of {salary} in twelve equal instalments.",
                   "What annual remuneration shall the Employer pay the Employee?",
                   "How much are the staff member's yearly wages?",
                   "What salary does the worker earn each year?", "salary"),
        ClauseSpec("noncompete", "The Employee shall not engage in any competing business for {months} months following cessation of employment.",
                   "For how many months following cessation of employment shall the Employee not engage in any competing business?",
                   "How long is the worker barred from joining a rival after leaving?",
                   "After departure, for what period can staff not work for a competitor?", "months"),
        ClauseSpec("confidentiality", "The Employee shall not disclose Confidential Information for {years} years following cessation of employment.",
                   "For how many years shall the Employee not disclose Confidential Information?",
                   "How long must staff keep secrets after leaving?",
                   "For what duration can a worker not reveal secret material after departure?", "years"),
        ClauseSpec("probation", "The Employee shall serve a probationary period of {prob} months from the commencement date.",
                   "What probationary period shall the Employee serve?",
                   "How long is the worker's trial phase?",
                   "What is the length of the trial period for new staff?", "prob"),
    ],
    "supply": [
        ClauseSpec("delivery", "The Supplier shall deliver the Goods within {deliv} days of receipt of each purchase order.",
                   "Within how many days of each purchase order shall the Supplier deliver the Goods?",
                   "How quickly must the vendor ship products after an order?",
                   "What is the seller's deadline to dispatch items once ordered?", "deliv"),
        ClauseSpec("liability", "The aggregate liability of the Supplier under this Agreement shall not exceed {liab}.",
                   "What amount shall the aggregate liability of the Supplier not exceed?",
                   "What is the maximum exposure of the vendor?",
                   "What is the most the seller can be held responsible for?", "liab"),
        ClauseSpec("payment", "The Purchaser shall settle each invoice within {pay} days of the invoice date.",
                   "Within how many days shall the Purchaser settle each invoice?",
                   "When must the buyer pay bills?",
                   "How long does the customer have to pay a bill?", "pay"),
        ClauseSpec("warranty", "The Supplier warrants the Goods against defects for {warr} months from delivery.",
                   "For how many months does the Supplier warrant the Goods against defects?",
                   "How long is the vendor's guarantee against faults in the products?",
                   "What guarantee period does the seller give on items?", "warr"),
    ],
}

BOILERPLATE = [
    "This Agreement constitutes the entire understanding between the parties and supersedes all prior negotiations.",
    "If any provision is held unenforceable, the remaining provisions shall continue in full force.",
    "This Agreement may be executed in counterparts, each of which shall be deemed an original.",
    "Neither party shall be in breach for delay caused by events beyond its reasonable control.",
    "No waiver of any breach shall be deemed a waiver of any subsequent breach.",
    "Neither party may assign its rights without the prior written consent of the other party.",
    "Headings are for convenience only and shall not affect interpretation.",
]

TITLES = {"lease": "Commercial Lease Agreement", "employment": "Employment Agreement", "supply": "Master Supply Agreement"}
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Pune"]
PARTIES = ["Acme Textiles", "Borealis Foods", "Cobalt Logistics", "Dune Pharma", "Ember Retail", "Fjord Software",
           "Garnet Steel", "Harbor Freight Lines", "Indus Agro", "Juniper Media", "Kestrel Motors", "Lumen Energy"]
FIRMS = [("firm_aurora", "Aurora Legal LLP"), ("firm_birch", "Birch & Co Advocates"), ("firm_cedar", "Cedar Chambers")]


@dataclass
class Document:
    id: str
    tenant_id: str
    matter_id: str
    title: str
    kind: str
    text: str


@dataclass
class EvalQuestion:
    id: str
    tenant_id: str
    matter_id: str
    question: str
    expected: str
    difficulty: str  # "easy" | "hard"
    clause: str


@dataclass
class Matter:
    id: str
    tenant_id: str
    name: str
    kind: str


@dataclass
class Corpus:
    tenants: dict[str, str] = field(default_factory=dict)
    matters: list[Matter] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    questions: list[EvalQuestion] = field(default_factory=list)
    training_pairs: list[tuple[str, str, str]] = field(default_factory=list)  # (query, clause, clause key)

    def matters_for(self, tenant_id: str) -> list[Matter]:
        return [m for m in self.matters if m.tenant_id == tenant_id]


def _values(rng: random.Random) -> dict[str, str]:
    return {
        "rent": money(rng.choice([85_000, 120_000, 240_000, 310_000])),
        "day": rng.choice(["third (3rd)", "fifth (5th)", "seventh (7th)", "tenth (10th)"]),
        "notice": num(rng.choice([30, 60, 90, 120])),
        "deposit": money(rng.choice([450_000, 600_000, 900_000])),
        "cap": money(rng.choice([25_000, 40_000, 75_000])),
        "city": rng.choice(CITIES),
        "salary": money(rng.choice([1_800_000, 2_400_000, 3_600_000])),
        "months": num(rng.choice([12, 18, 24])),
        "years": num(rng.choice([1, 2, 5])),
        "prob": num(rng.choice([3, 6])),
        "deliv": num(rng.choice([7, 14, 21])),
        "liab": money(rng.choice([5_000_000, 12_000_000, 20_000_000])),
        "pay": num(rng.choice([30, 45, 60])),
        "warr": num(rng.choice([12, 24, 36])),
    }


def _agreement_text(kind: str, values: dict[str, str], rng: random.Random) -> tuple[str, list[str]]:
    clause_sentences = [c.template.format(**values) for c in CLAUSES[kind]]
    body = clause_sentences + rng.sample(BOILERPLATE, 4)
    rng.shuffle(body)
    return " ".join(body), clause_sentences


def _side_documents(tenant_id: str, matter_id: str, a: str, b: str, rng: random.Random) -> list[Document]:
    email = (
        f"Dear Counsel, we act for {a} in the matter against {b}. Our client remains concerned about the delay in "
        f"resolving the outstanding issues. Please confirm your availability for a call on {rng.randint(2, 27)} "
        f"October. We reserve all rights in the meantime."
    )
    memo = (
        f"Internal memo. Limitation analysis for {a} is pending partner review. The client prefers settlement over "
        f"litigation. Next step is to collect the signed originals from {b} and update the chronology."
    )
    return [
        Document(f"{matter_id}-email", tenant_id, matter_id, "Correspondence with opposing counsel", "email", email),
        Document(f"{matter_id}-memo", tenant_id, matter_id, "Internal strategy memo", "memo", memo),
    ]


def build_corpus(seed: int = 7, matters_per_firm: int = 4, split: str = "eval") -> Corpus:
    """split="train" phrases hard questions with the training paraphrases (use a different seed),
    so routers and embedders can be fitted without ever touching an eval question."""
    rng = random.Random(seed)
    hard_attr = "hard" if split == "eval" else "hard_train"
    corpus = Corpus(tenants=dict(FIRMS))
    kinds = list(CLAUSES)
    for tenant_id, _ in FIRMS:
        for i in range(matters_per_firm):
            kind = kinds[i % len(kinds)]
            a, b = rng.sample(PARTIES, 2)
            matter = Matter(f"{tenant_id[5:]}-m{i + 1}", tenant_id, f"{a} v. {b}", kind)
            corpus.matters.append(matter)
            values = _values(rng)
            text, _ = _agreement_text(kind, values, rng)
            corpus.documents.append(Document(f"{matter.id}-agreement", tenant_id, matter.id, TITLES[kind], kind, text))
            corpus.documents.extend(_side_documents(tenant_id, matter.id, a, b, rng))
            for clause in CLAUSES[kind]:
                for difficulty in ("easy", "hard"):
                    corpus.questions.append(EvalQuestion(
                        f"{matter.id}-{clause.key}-{difficulty}", tenant_id, matter.id,
                        getattr(clause, hard_attr if difficulty == "hard" else "easy"), values[clause.answer_field],
                        difficulty, clause.key,
                    ))
    # Public training set: different values, different paraphrases; never overlaps the eval questions.
    for _ in range(6):
        values = _values(rng)
        for kind in kinds:
            for clause in CLAUSES[kind]:
                sentence = clause.template.format(**values)
                corpus.training_pairs.append((clause.hard_train, sentence, clause.key))
                for _ in range(2):
                    corpus.training_pairs.append((synthetic_query(sentence, rng), sentence, clause.key))
    return corpus


def synthetic_query(sentence: str, rng: random.Random, keep: float = 0.25) -> str:
    """Teacher-style query generation: restate a clause in colloquial vocabulary.

    Drafting terms are swapped for synonyms and most other words are dropped, which forces the
    embedder to learn that "tenant" and "Lessee" mean the same thing instead of leaning on the
    words a query and its clause happen to share. It never sees an eval question.
    """
    words = []
    for word in sentence.split():
        canonical = stem(re.sub(r"[^A-Za-z]", "", word))
        if canonical in THESAURUS:
            words.append(rng.choice(THESAURUS[canonical]) if rng.random() < 0.8 else word.lower())
        elif not any(ch.isdigit() for ch in word) and rng.random() < keep:
            words.append(word.lower().strip(".,"))
    return " ".join(words)
