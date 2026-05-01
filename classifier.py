"""
HTS Classifier Engine
---------------------
Suggests likely HTSUS codes for a plain-English product description.

The matching logic is intentionally transparent (rules-based, not a black box)
because trade compliance work requires explainability — you need to be able to
defend a classification to CBP, an auditor, or your own legal team.

Scoring approach:
  1. Extract attributes (material, gender, product type, construction) from input
  2. Score each HTS code's hierarchical description for keyword overlap
  3. Boost scores when extracted attributes are explicitly matched
  4. Return top N candidates with a confidence band (HIGH / MEDIUM / LOW)
"""

import pandas as pd
import re
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Domain dictionaries — the "rules" in our rules engine.
# These reflect how HTSUS chapters actually carve up the world.
# ---------------------------------------------------------------------------

MATERIALS = {
    "cotton", "wool", "woolen", "silk", "linen", "leather", "polyester",
    "nylon", "rayon", "acrylic", "synthetic", "plastic", "rubber", "wood",
    "wooden", "bamboo", "paper", "cardboard", "glass", "ceramic", "porcelain",
    "steel", "iron", "aluminum", "copper", "brass", "gold", "silver",
    "denim", "canvas", "felt", "fur", "down", "feathers", "ceramic",
    "metal", "metallic", "stone"
}

# Map normalized -> canonical material as it appears in HTSUS text
MATERIAL_CANONICAL = {
    "wooden": "wood",
    "woolen": "wool",
    "metallic": "metal",
}

GENDERS = {
    "men": "men's", "mens": "men's", "men's": "men's",
    "women": "women's", "womens": "women's", "women's": "women's",
    "boys": "boys'", "boys'": "boys'",
    "girls": "girls'", "girls'": "girls'",
    "infant": "babies'", "infants": "babies'", "baby": "babies'",
    "babies": "babies'", "babies'": "babies'",
    "children": "children", "kids": "children",
    "unisex": "unisex"
}

CONSTRUCTION = {
    "knitted", "knit", "crocheted", "woven", "non-woven", "nonwoven",
    "embroidered", "printed", "dyed"
}

# Common product type keywords (not exhaustive — the hierarchy match handles the rest)
PRODUCT_HINTS = {
    "shirt", "shirts", "blouse", "trousers", "pants", "jeans", "shorts",
    "jacket", "coat", "dress", "skirt", "suit", "sweater", "pullover",
    "cardigan", "hoodie", "t-shirt", "tshirt", "tee", "underwear", "bra",
    "socks", "stockings", "tights", "hat", "cap", "scarf", "gloves",
    "shoe", "shoes", "boot", "boots", "sneaker", "sandal",
    "bag", "backpack", "wallet", "belt", "handbag", "purse",
    "phone", "laptop", "computer", "battery", "charger", "cable",
    "toy", "toys", "game", "games", "doll", "puzzle", "card", "cards",
    "book", "magazine", "notebook",
    "chair", "chairs", "table", "desk", "lamp", "sofa",
    "tire", "tires", "wheel",
}

# English stopwords we don't want to match on
STOPWORDS = {
    "a", "an", "the", "for", "of", "in", "on", "with", "and", "or",
    "is", "are", "to", "from", "as", "at", "by", "this", "that"
}


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    # Keep apostrophes for now (men's, boys'), strip everything else
    text = re.sub(r"[^a-z0-9'\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Split into tokens, drop stopwords."""
    return [t for t in normalize(text).split() if t and t not in STOPWORDS]


def extract_attributes(query: str) -> dict:
    """
    Pull structured attributes out of a free-text product description.
    This is what makes the classifier 'smart' — it understands that
    'men's cotton t-shirt' has gender=men's, material=cotton, product=t-shirt.
    """
    tokens = tokenize(query)
    attrs = {
        "materials": [],
        "gender": None,
        "construction": [],
        "products": [],
        "raw_tokens": tokens,
    }
    for tok in tokens:
        if tok in MATERIALS:
            attrs["materials"].append(tok)
        if tok in GENDERS and not attrs["gender"]:
            attrs["gender"] = GENDERS[tok]
        if tok in CONSTRUCTION:
            attrs["construction"].append(tok)
        if tok in PRODUCT_HINTS:
            attrs["products"].append(tok)
    return attrs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_row(query_tokens: list[str], attrs: dict, full_desc: str) -> tuple[float, list[str]]:
    """
    Score one HTS row against the user's query.
    Returns (score 0-100, list of reasons explaining the score).

    Scoring philosophy:
      - Reward when ALL query tokens appear (not just some)
      - Reward attribute matches heavily (material, product type, gender)
      - Penalize long, sprawling descriptions where the matched terms are
        incidental (e.g., a "suit" entry that mentions "shirt" deep in its
        hierarchy shouldn't outrank a dedicated "shirt" entry)
    """
    desc_lower = full_desc.lower()
    desc_tokens = set(tokenize(full_desc))
    reasons = []
    score = 0.0

    # 1. Token overlap — reward both coverage AND completeness
    matched_tokens = [t for t in query_tokens if t in desc_tokens]
    if query_tokens:
        coverage = len(matched_tokens) / len(query_tokens)
        # All terms matched -> 35 points; half matched -> ~12 points (squared curve)
        score += (coverage ** 1.5) * 35
        if matched_tokens:
            reasons.append(f"Matched terms: {', '.join(matched_tokens)}")
        # Heavy penalty if NO query tokens matched at all
        if coverage == 0:
            return 0.0, []

    # 2. Material match — apply canonical form for stems like 'wooden' -> 'wood'
    if attrs["materials"]:
        for mat in attrs["materials"]:
            canonical = MATERIAL_CANONICAL.get(mat, mat)
            if canonical in desc_lower or mat in desc_lower:
                score += 22
                reasons.append(f"Material match: '{canonical}'")

    # 3. Gender match
    if attrs["gender"]:
        gender_root = attrs["gender"].rstrip("'s").rstrip("'")
        if gender_root in desc_lower:
            score += 18
            reasons.append(f"Gender match: {attrs['gender']}")

    # 4. Construction match (knit, woven, etc.)
    if attrs["construction"]:
        for con in attrs["construction"]:
            if con in desc_lower:
                score += 10
                reasons.append(f"Construction: '{con}'")

    # 5. Product type — strongest single signal
    if attrs["products"]:
        for prod in attrs["products"]:
            singular = prod.rstrip("s")
            if prod in desc_lower or singular in desc_lower:
                score += 20
                reasons.append(f"Product type: '{prod}'")

    # 6. Fuzzy similarity bonus for spelling variations
    fuzzy = fuzz.token_set_ratio(" ".join(query_tokens), desc_lower)
    score += (fuzzy / 100) * 8

    # 7. Description length penalty — focused descriptions beat sprawling ones.
    # A short description like "Of cotton > Men's" matching is far more
    # confident than a 30-word ensemble description that happens to mention shirts.
    desc_word_count = len(desc_lower.split())
    if desc_word_count > 25:
        score -= min((desc_word_count - 25) * 0.4, 12)

    return min(max(score, 0), 100), reasons


def confidence_band(score: float) -> str:
    """Translate raw score to a band a compliance person can interpret."""
    if score >= 75:
        return "HIGH"
    elif score >= 55:
        return "MEDIUM"
    elif score >= 35:
        return "LOW"
    return "VERY LOW"


# ---------------------------------------------------------------------------
# Main classify function
# ---------------------------------------------------------------------------

def classify(query: str, df: pd.DataFrame, top_n: int = 10) -> list[dict]:
    """
    Return the top N HTS code suggestions for a product description.
    Each result includes the code, description, duty rate, score, band,
    and reasons explaining why it matched.
    """
    if not query.strip():
        return []

    query_tokens = tokenize(query)
    attrs = extract_attributes(query)

    # Pre-filter: rows must contain at least one query token
    # (massive speedup vs scoring all 29k rows)
    if query_tokens:
        mask = df['full_description'].str.lower().apply(
            lambda d: any(t in d for t in query_tokens) if isinstance(d, str) else False
        )
        candidates = df[mask].copy()
    else:
        candidates = df.copy()

    # Score every candidate
    scores = []
    reasons_list = []
    for desc in candidates['full_description']:
        s, r = score_row(query_tokens, attrs, str(desc))
        scores.append(s)
        reasons_list.append(r)
    candidates['score'] = scores
    candidates['reasons'] = reasons_list

    # Prefer rows with longer (deeper, more specific) HTS codes when scores tie,
    # because a 10-digit statistical code is more useful than a 4-digit chapter.
    candidates['code_specificity'] = candidates['HTS Number'].astype(str).str.replace('.', '').str.len()
    candidates = candidates.sort_values(
        ['score', 'code_specificity'], ascending=[False, False]
    )

    top = candidates.head(top_n)
    results = []
    for _, row in top.iterrows():
        results.append({
            "hts_code": str(row['HTS Number']),
            "description": str(row['full_description']),
            "duty_general": str(row.get('General Rate of Duty', '') or ''),
            "duty_special": str(row.get('Special Rate of Duty', '') or ''),
            "duty_column2": str(row.get('Column 2 Rate of Duty', '') or ''),
            "unit": str(row.get('Unit of Quantity', '') or ''),
            "score": round(float(row['score']), 1),
            "confidence": confidence_band(float(row['score'])),
            "reasons": row['reasons'],
        })
    return results


def get_chapter(hts_code: str) -> str:
    """Extract the 2-digit chapter number from an HTS code."""
    digits = hts_code.replace('.', '')
    return digits[:2] if len(digits) >= 2 else ""


def chapter_notes_url(hts_code: str) -> str:
    """Build a deep link to the official USITC chapter page."""
    chapter = get_chapter(hts_code)
    if not chapter:
        return "https://hts.usitc.gov/"
    # USITC search supports filtering by chapter
    return f"https://hts.usitc.gov/search?query={chapter}"
