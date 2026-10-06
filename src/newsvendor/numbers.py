"""Grounded lexical candidates from prose. No task, slot, applicability or gold is read."""

import re
from functools import lru_cache

WORDS = {
    word: i
    for i, word in enumerate(
        (
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        )
    )
}
NUMBER = r"-?\d+(?:,\d{3})*(?:\.\d+)?"
COUNT = re.compile(
    r"(?<![\w-])("
    + NUMBER
    + "|"
    + "|".join(WORDS)
    + r")\s+(?:individual\s+)?(?:items?|units?|pieces?)\b",
    re.IGNORECASE,
)
MONEY = re.compile(r"(USD|GBP|EUR|[$£€])\s*(" + NUMBER + r")", re.IGNORECASE)
TERMS = {
    "refund": r"refund|credit|reimburse",
    "handling fee": r"handling|inspection|fee|deduct|collection charge",
    "selling price": r"selling|retail|checkout|charge customers",
    "shortage cost": r"shortage|inconvenience|unmet|missed",
}


def clauses(text, start, end):
    separator = r"[;\n|]|(?<!\d)[.,](?!\d)"
    before = re.split(separator, text[:start])[-1].lower()
    after = re.split(separator, text[end:])[0].lower()
    # Do not borrow the unit of a subsequent monetary amount.
    following = MONEY.search(after)
    if following:
        after = after[: following.start()]
    return before, after


@lru_cache(maxsize=16384)
def lexical(text):
    result = []
    for match in COUNT.finditer(text):
        token = match.group(1).lower()
        value = WORDS[token] if token in WORDS else float(token.replace(",", ""))
        result.append(
            {
                "label": "pack size",
                "value": float(value),
                "span": list(match.span(1)),
                "unit": "count",
            }
        )
    for match in MONEY.finditer(text):
        before, after = clauses(text, match.start(), match.end())
        hits = []
        for label, pattern in TERMS.items():
            left = list(re.finditer(pattern, before))
            right = re.search(pattern, after)
            if left:
                hits.append((len(before) - left[-1].end(), label))
            if right:
                hits.append((right.start(), label))
        label = min(hits)[1] if hits else "amount"
        if re.search(r"(?:per|a|each|for a)\s+(?:carton|box|pack|bundle)\b", after) or (
            re.match(r"\s*for\s+", after) and COUNT.search(after)
        ):
            unit = "currency/pack"
        elif re.search(
            r"\b(?:per|each|a|for each)\s+(?:[\w-]+\s+){0,2}(?:item|unit|piece)\b|\beach\b", after
        ) or re.search(
            r"\b(?:per|each|for each)\s+(?:[\w-]+\s+){0,2}(?:item|unit|piece)\b", before[-80:]
        ):
            unit = "currency/unit"
        elif label in TERMS and label != "amount":
            unit = (
                "currency/unit"
                if re.search(
                    r"\b(?:per|each)\b.*\b(?:unit|item)|\bper unit\b", before + " " + after
                )
                else "currency/unknown"
            )
        elif re.search(r"\b(?:carton|box|pack|bundle)\b", before) and COUNT.search(text):
            unit = "currency/pack"
        elif "between" in before and re.match(r"\s*and\b", after):
            following = re.split(r"[;\n|]|(?<!\d)[.,](?!\d)", text[match.end() :])[0]
            unit = (
                "currency/unit"
                if re.search(r"\bper\s+(?:[\w-]+\s+){0,2}(?:item|unit|piece)\b", following)
                else "currency/unknown"
            )
        else:
            unit = "currency/unknown"
        if unit == "currency/pack":
            label = "pack price"
        result.append(
            {
                "label": label,
                "value": float(match.group(2).replace(",", "")),
                "span": list(match.span(2)),
                "unit": unit,
                "currency": {"$": "USD", "£": "GBP", "€": "EUR"}.get(match[1], match[1].upper()),
            }
        )
    for match in re.finditer(
        r"\bnot returnable\b[^.\n]*?(\bno (?:disposal|salvage|residual) value\b)", text, re.I
    ):
        result.append(
            {
                "label": "explicit zero recovery",
                "value": 0.0,
                "span": list(match.span(1)),
                "unit": "currency/unit",
            }
        )
    return tuple(sorted(result, key=lambda a: a["span"]))


def atoms(doc):
    return [
        {"id": f"{doc['id']}:text:{i}", **a, "span": list(a["span"])}
        for i, a in enumerate(lexical(doc["text"]))
    ]
