# intent_agent.py
from typing import Tuple
DOMAINS = {
    "finance": ["revenue", "profit", "income", "loss", "expense", "invoice"],
    "sales": ["sales", "order", "customer", "units", "revenue"],
    "hr": ["employee", "salary", "hiring", "attrition", "hire", "payroll"],
    "marketing": ["campaign", "ctr", "impression", "conversion", "leads"],
    "inventory": ["stock", "inventory", "sku", "warehouse", "qty", "quantity"]
}

def guess_intent(query: str) -> Tuple[str, float]:
    """Return (domain, score) for a query using simple keyword matching."""
    q = (query or "").lower()
    scores = {}
    for d, keywords in DOMAINS.items():
        s = sum(1 for k in keywords if k in q)
        scores[d] = s
    best = max(scores, key=lambda k: scores[k])
    score = scores[best]
    if score == 0:
        return "general", 0.0
    # normalize score
    return best, float(score) / max(1, len(DOMAINS[best]))
