"""The design-partner firms' subscriptions (the corpus tenants)."""
from __future__ import annotations

from datetime import date

from sib.billing import Subscription

SUBSCRIPTIONS = [
    Subscription("firm_aurora", "team", 5, "referral", date(2026, 7, 1)),
    Subscription("firm_birch", "team", 6, "linkedin_ads", date(2026, 8, 1)),
    Subscription("firm_cedar", "team", 3, "content_seo", date(2026, 9, 15), trial=True),
]
