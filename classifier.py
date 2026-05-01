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

# ---------------------------------------------------------------------------
# SYNONYM LAYER — bridge everyday English to formal HTSUS vocabulary
# ---------------------------------------------------------------------------
# HTSUS uses customs-formal terms ("footwear", "automatic data processing
# machine") while users search in everyday English ("shoe", "laptop"). This
# dictionary maps common search terms to the actual words used in HTSUS
# descriptions so the matcher can find the right codes.
#
# Each entry is verified against the HTSUS dataset — every target term on the
# right side actually appears in real HTSUS descriptions. Add new mappings
# here whenever you find a misclassification rooted in vocabulary mismatch.
# ---------------------------------------------------------------------------
SYNONYMS = {
    # Footwear (chapter 64 uses "footwear", never "shoe")
    "shoe":      ["footwear"],
    "shoes":     ["footwear"],
    "sneaker":   ["footwear", "sports"],
    "sneakers":  ["footwear", "sports"],
    "trainer":   ["footwear", "sports"],
    "trainers":  ["footwear", "sports"],

    # Wallets/purses (4202.31 = "articles normally carried in the pocket or in the handbag")
    "wallet":    ["pocket", "handbag"],
    "wallets":   ["pocket", "handbag"],
    "billfold":  ["pocket", "handbag"],
    "purse":     ["handbag", "pocket"],

    # Phones (8517 uses "telephones", "smartphones")
    "phone":     ["telephone", "smartphone"],
    "phones":    ["telephone", "smartphone"],
    "cellphone": ["telephone", "smartphone"],
    "mobile":    ["telephone", "smartphone"],
    "smartphone":["telephone"],   # bridge to chapter 8517

    # Computers (8471 uses "portable automatic data processing")
    # Use "portable" only — generic words like "data" or "processing" caused false positives
    "laptop":    ["portable"],
    "laptops":   ["portable"],
    "computer":  ["portable"],
    "computers": ["portable"],
    "pc":        ["portable"],

    # Television (8528 = reception apparatus FOR TELEVISION; we want both words to differentiate from radio reception)
    "tv":        ["television"],
    "tvs":       ["television"],

    # Apparel terms HTSUS uses formally
    "hoodie":    ["sweatshirt", "pullover"],
    "hoodies":   ["sweatshirt", "pullover"],
    "tee":       ["t-shirt", "singlet"],
    "tshirt":    ["t-shirt"],

    # Audio gear (8518 uses "headphones", "earphones")
    "earbud":    ["earphone", "headphone"],
    "earbuds":   ["earphone", "headphone"],
    "airpods":   ["earphone", "headphone"],
    "headset":   ["headphone", "earphone"],
    "headsets":  ["headphone", "earphone"],

    # Eyewear (9004 uses "spectacles")
    "glasses":   ["spectacles", "eyewear"],
    "eyeglasses":["spectacles", "eyewear"],

    # Watches (chapter 91 uses "watch", "wrist")
    "wristwatch":["watch", "wrist"],
    "wristwatches":["watch", "wrist"],

    # Lighting (9405 uses "lamps", "luminaires")
    "bulb":      ["lamp", "luminaire"],
    "bulbs":     ["lamp", "luminaire"],
    "light":     ["lamp", "luminaire"],
    "lights":    ["lamp", "luminaire"],

    # Vehicles (chapter 87)
    "car":       ["vehicle", "automobile"],
    "cars":      ["vehicle", "automobile"],
    "truck":     ["vehicle"],
    "trucks":    ["vehicle"],
    "bike":      ["bicycle"],
    "bikes":     ["bicycle"],

    # Tires
    "tire":      ["pneumatic"],
    "tires":     ["pneumatic"],
    "tyre":      ["pneumatic"],
    "tyres":     ["pneumatic"],

    # Furniture (9401 uses "seats" not "chairs")
    "chair":     ["seat"],
    "chairs":    ["seat"],

    # Jewelry
    "jewellery": ["jewelry"],

    # Floor coverings (chapter 57 uses "carpets")
    "rug":       ["carpet"],
    "rugs":      ["carpet"],
}


def expand_with_synonyms(tokens: list[str]) -> tuple[list[str], dict]:
    """
    Expand a list of query tokens to include synonym matches.
    Returns (expanded_tokens, synonym_map) where synonym_map records
    which originals produced which expansions (for transparent reasoning).
    """
    expanded = list(tokens)
    syn_map = {}
    for tok in tokens:
        if tok in SYNONYMS:
            for synonym in SYNONYMS[tok]:
                if synonym not in expanded:
                    expanded.append(synonym)
                    syn_map.setdefault(tok, []).append(synonym)
    return expanded, syn_map

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

def _matches_token(query_token: str, desc_tokens: set, desc_lower: str) -> bool:
    """
    Check if a query token matches a description.
    Handles common singular/plural variation: 'bicycle' matches 'bicycles',
    'shoe' matches 'shoes'. Avoids false positives by only checking obvious
    plural forms (+s, +es) — not arbitrary substrings.
    """
    if query_token in desc_tokens:
        return True
    # Try common plural forms
    if (query_token + "s") in desc_tokens:
        return True
    if (query_token + "es") in desc_tokens:
        return True
    # Try removing trailing 's' (e.g., 'shoes' → check 'shoe')
    if query_token.endswith("s") and query_token[:-1] in desc_tokens:
        return True
    return False


def score_row(query_tokens: list[str], attrs: dict, full_desc: str) -> tuple[float, list[str]]:
    """
    Score one HTS row against the user's query.
    Returns (score 0-100, list of reasons explaining the score).

    Scoring philosophy:
      - Reward when ALL query tokens (or their synonyms) appear
      - Reward attribute matches heavily (material, product type, gender)
      - Reward matches at the HEADING level much more than deep-subcategory matches
      - Use synonym expansion to bridge everyday words to formal HTSUS vocabulary
      - Handle singular/plural naturally for both originals and synonyms
      - Penalize long, sprawling descriptions where matches are incidental
    """
    desc_lower = full_desc.lower()
    desc_tokens = set(tokenize(full_desc))
    reasons = []
    score = 0.0

    # Expand query tokens with synonyms — 'shoe' also looks for 'footwear', etc.
    expanded_tokens, syn_map = expand_with_synonyms(query_tokens)

    # Split the hierarchical description into levels.
    levels = [lvl.strip().lower() for lvl in full_desc.split(">")]

    # 1. Token overlap — handles plurals for originals; substring for synonyms
    matched_originals = [t for t in query_tokens if _matches_token(t, desc_tokens, desc_lower)]
    matched_synonyms = []
    for t in expanded_tokens:
        if t in query_tokens:
            continue
        if t in desc_lower:
            matched_synonyms.append(t)

    if query_tokens:
        covered = set()
        for orig in query_tokens:
            if _matches_token(orig, desc_tokens, desc_lower):
                covered.add(orig)
            elif orig in SYNONYMS:
                if any(syn in desc_lower for syn in SYNONYMS[orig]):
                    covered.add(orig)
        coverage = len(covered) / len(query_tokens)
        score += (coverage ** 1.5) * 35

        if matched_originals:
            reasons.append(f"Matched terms: {', '.join(matched_originals)}")
        if matched_synonyms:
            bridge_notes = []
            for syn in matched_synonyms:
                originals = [orig for orig, syns in syn_map.items() if syn in syns]
                if originals:
                    bridge_notes.append(f"{originals[0]}→{syn}")
            if bridge_notes:
                reasons.append(f"Synonym match: {', '.join(bridge_notes)}")

        if coverage == 0:
            return 0.0, []

    # 2. HEADING-LEVEL MATCH BONUS — including synonyms (substring against heading)
    if levels and expanded_tokens:
        heading = levels[0]
        heading_matches = [t for t in expanded_tokens if t in heading]
        if heading_matches:
            score += 25
            reasons.append(f"Heading-level match: {', '.join(heading_matches)}")

    # 3. Material match — apply canonical form for stems like 'wooden' -> 'wood'
    if attrs["materials"]:
        for mat in attrs["materials"]:
            canonical = MATERIAL_CANONICAL.get(mat, mat)
            if canonical in desc_lower or mat in desc_lower:
                score += 22
                reasons.append(f"Material match: '{canonical}'")

    # 4. Gender match
    if attrs["gender"]:
        gender_root = attrs["gender"].rstrip("'s").rstrip("'")
        if gender_root in desc_lower:
            score += 18
            reasons.append(f"Gender match: {attrs['gender']}")

    # 5. Construction match (knit, woven, etc.)
    if attrs["construction"]:
        for con in attrs["construction"]:
            if con in desc_lower:
                score += 10
                reasons.append(f"Construction: '{con}'")

    # 6. Product type — strongest single signal (also expands via synonyms)
    if attrs["products"]:
        for prod in attrs["products"]:
            singular = prod.rstrip("s")
            # Direct product hit
            if prod in desc_lower or singular in desc_lower:
                score += 20
                reasons.append(f"Product type: '{prod}'")
            # Synonym product hit (e.g. 'shoe' product, but description says 'footwear')
            elif prod in SYNONYMS:
                for syn in SYNONYMS[prod]:
                    if syn in desc_lower:
                        score += 18  # slightly less than direct match
                        reasons.append(f"Product (synonym): '{prod}'→'{syn}'")
                        break

    # 7. Fuzzy similarity bonus for spelling variations
    fuzzy = fuzz.token_set_ratio(" ".join(expanded_tokens), desc_lower)
    score += (fuzzy / 100) * 8

    # 8. Description length penalty
    desc_word_count = len(desc_lower.split())
    if desc_word_count > 25:
        score -= min((desc_word_count - 25) * 0.4, 12)

    return min(max(score, 0), 100), reasons


def is_supplemental_chapter(hts_code: str) -> bool:
    """
    Chapters 98 and 99 are supplemental tariff schedules, not primary
    classifications. Specifically:
      - 9802: Articles returned to the U.S. after processing abroad
      - 9817: Special imports (like prototypes for testing)
      - 9902: Temporary Reductions in Duties (Miscellaneous Tariff Bill)
      - 9903: Additional Duties (Section 301 China, Section 232 steel/aluminum)
    A real product is primary-classified in chapters 1-97 and then
    additionally checked against these supplemental chapters. We penalize
    them so they don't outrank the primary classification.
    """
    digits = hts_code.replace('.', '')
    if not digits:
        return False
    chapter = digits[:2]
    return chapter in ('98', '99')


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
    expanded_tokens, _ = expand_with_synonyms(query_tokens)

    # Pre-filter: rows must contain at least one query token OR synonym
    # (massive speedup vs scoring all 29k rows)
    if expanded_tokens:
        mask = df['full_description'].str.lower().apply(
            lambda d: any(t in d for t in expanded_tokens) if isinstance(d, str) else False
        )
        candidates = df[mask].copy()
    else:
        candidates = df.copy()

    # Score every candidate
    scores = []
    reasons_list = []
    for _, candidate_row in candidates.iterrows():
        desc = candidate_row['full_description']
        code = str(candidate_row['HTS Number'])
        s, r = score_row(query_tokens, attrs, str(desc))

        # Penalty for supplemental tariff chapters (98, 99) — these are
        # not primary classifications and should only surface when nothing
        # else matches. Apply a heavy penalty unless the user explicitly
        # asked for these chapters.
        if is_supplemental_chapter(code) and not any(
            t in ("9802", "9817", "9902", "9903", "98", "99") for t in query_tokens
        ):
            s = s * 0.5
            if r:
                r = r + ["Supplemental tariff schedule (penalty applied)"]

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


def get_heading(hts_code: str) -> str:
    """Extract the 4-digit heading from an HTS code (e.g. '6403' from '6403.19.50.00')."""
    digits = hts_code.replace('.', '')
    return digits[:4] if len(digits) >= 4 else digits


def chapter_notes_url(hts_code: str) -> str:
    """
    Build a deep link to the official USITC HTSUS site landing on the exact
    subheading for the suggested code. Using the /search?query= path with the
    full HTS code (e.g. 6403.19.50) lands users directly on that entry's
    context — duty rate, statistical breakouts, sibling entries.
    """
    if not hts_code or not str(hts_code).strip():
        return "https://hts.usitc.gov/"
    return f"https://hts.usitc.gov/search?query={str(hts_code).strip()}"