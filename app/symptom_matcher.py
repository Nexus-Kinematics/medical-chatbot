"""
symptom_matcher.py

Maps free-text, natural-language symptom descriptions onto the exact
canonical symptom vocabulary the trained model was fit on
(model/symptom_names.pkl).

Design goals (per project spec):
- The user should NOT have to type exact dataset symptom names.
- We should NEVER silently invent/assume a symptom that wasn't
  confidently detected. When in doubt, we leave it out.

Matching strategy (three tiers, in order of confidence):
    1. Direct phrase match   - a canonical symptom name (e.g. "sore
       throat") appears, as a contiguous run of words, inside the
       user's message.
    2. Synonym phrase match  - a hand-curated everyday phrase (e.g.
       "high temperature", "my head is pounding") maps to exactly one
       canonical symptom, and that phrase appears in the message.
    3. Fuzzy single-word fallback - catches small typos ("feaver")
       against single-word canonical symptoms and synonym keys, using
       a conservative similarity cutoff.

A lightweight, dependency-free "stemmer" (strip common English plural
suffixes) is used so that "seizure" matches the canonical "seizures",
"chills" matches "chills", etc., without pulling in NLTK/spaCy.
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable


# ---------------------------------------------------------------------------
# Curated synonym map: everyday phrase -> canonical symptom name.
# Only unambiguous mappings are included. Anything that could plausibly
# mean several different canonical symptoms (e.g. generic "stomach pain")
# is deliberately left OUT rather than guessed.
# ---------------------------------------------------------------------------
SYMPTOM_SYNONYMS: dict[str, str] = {
    # Fever / temperature
    "high temperature": "fever",
    "running a temperature": "fever",
    "burning up": "fever",
    "temperature is high": "fever",

    # Headache
    "head hurts": "headache",
    "my head is pounding": "headache",
    "pounding head": "headache",
    "head is throbbing": "headache",
    "splitting headache": "headache",

    # Dizziness
    "feel dizzy": "dizziness",
    "feeling dizzy": "dizziness",
    "room is spinning": "dizziness",
    "light headed": "dizziness",
    "lightheaded": "dizziness",
    "head is spinning": "dizziness",

    # Vomiting
    "throwing up": "vomiting",
    "threw up": "vomiting",
    "been sick": "vomiting",

    # Nasal / sinus
    "runny nose": "nasal congestion",
    "stuffy nose": "nasal congestion",
    "blocked nose": "nasal congestion",
    "congested nose": "nasal congestion",

    # Sleep
    "cant sleep": "insomnia",
    "can not sleep": "insomnia",
    "trouble sleeping": "insomnia",
    "unable to sleep": "insomnia",

    # Fatigue / weakness
    "no energy": "fatigue",
    "low energy": "fatigue",
    "feeling exhausted": "fatigue",
    "feeling weak": "weakness",
    "feel weak": "weakness",

    # Throat
    "throat hurts": "sore throat",
    "throat pain": "sore throat",
    "scratchy throat": "sore throat",

    # Breathing
    "out of breath": "shortness of breath",
    "short of breath": "shortness of breath",
    "cant catch my breath": "shortness of breath",
    "cant breathe": "difficulty breathing",
    "trouble breathing": "difficulty breathing",

    # GI
    "loose stools": "diarrhea",
    "watery stools": "diarrhea",
    "cant poop": "constipation",
    "constipated": "constipation",

    # Skin
    "itchy skin": "itching of skin",
    "skin is itchy": "itching of skin",
    "yellow skin": "jaundice",
    "yellowing skin": "jaundice",
    "yellowing of the skin": "jaundice",

    # Cardio
    "racing heart": "palpitations",
    "heart racing": "palpitations",
    "heart pounding": "palpitations",
    "heart is pounding": "palpitations",

    # General aches
    "body aches": "ache all over",
    "achy all over": "ache all over",
    "everything hurts": "ache all over",

    # Sweating / chills
    "sweating a lot": "sweating",
    "shivering": "chills",

    # Neuro / sensation
    "pins and needles": "paresthesia",
    "memory problems": "disturbance of memory",
    "forgetful": "disturbance of memory",
    "cant focus": "disturbance of memory",

    # Vision
    "blurry vision": "diminished vision",
    "blurred vision": "diminished vision",
    "seeing double": "double vision",

    # Ears
    "ringing ears": "ringing in ear",
    "ringing in my ears": "ringing in ear",
    "ear ache": "ear pain",
    "earache": "ear pain",

    # Body region aches
    "back hurts": "back pain",
    "my back hurts": "back pain",
    "neck hurts": "neck pain",
    "stiff neck": "neck stiffness or tightness",
    "knee hurts": "knee pain",

    # Lymph
    "swollen glands": "swollen lymph nodes",

    # Appetite / weight
    "not hungry": "decreased appetite",
    "loss of appetite": "decreased appetite",
    "gaining weight": "weight gain",
    "losing weight": "recent weight loss",

    # Urinary
    "peeing a lot": "frequent urination",
    "frequent peeing": "frequent urination",
    "blood in my pee": "blood in urine",
    "burning when i pee": "painful urination",
    "painful peeing": "painful urination",

    # Mood
    "feeling anxious": "anxiety and nervousness",
    "feeling nervous": "anxiety and nervousness",
    "feeling depressed": "depression",
    "feeling down": "depression",

    # Nose
    "nose bleed": "nosebleed",

    # Misc
    "feel sick": "feeling ill",
    "feeling unwell": "feeling ill",
}

# Minimum similarity ratio (0-1) required for the fuzzy single-word
# fallback to accept a match. Deliberately strict to avoid false
# positives / invented symptoms.
_FUZZY_CUTOFF = 0.87
_MIN_FUZZY_WORD_LEN = 4


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation (keep letters/digits/spaces), collapse whitespace."""
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _stem(word: str) -> str:
    """Very small, conservative plural-stripping stemmer (no external deps)."""
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("es") and word[-3] not in "aeiou" and len(word) > 5:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _stem_phrase(phrase: str) -> tuple[str, ...]:
    return tuple(_stem(w) for w in phrase.split())


def _contains_subsequence(tokens: list[str], sub: tuple[str, ...]) -> bool:
    """True if `sub` appears as a contiguous run inside `tokens`."""
    n, m = len(tokens), len(sub)
    if m == 0 or m > n:
        return False
    for i in range(n - m + 1):
        if tuple(tokens[i : i + m]) == sub:
            return True
    return False


def detect_symptoms(message: str, known_symptoms: Iterable[str]) -> list[str]:
    """
    Detect which canonical symptoms (from `known_symptoms`) are present
    in the free-text `message`.

    Returns a de-duplicated list of canonical symptom names, in the
    order they were first found. Never returns a symptom that wasn't
    confidently matched by phrase, synonym, or close-typo fuzzy match.
    """
    known_symptoms = list(known_symptoms)
    normalized = _normalize(message)
    if not normalized:
        return []

    tokens = normalized.split()
    stemmed_tokens = [_stem(t) for t in tokens]

    detected: list[str] = []
    matched_word_spans: set[int] = set()  # indices in `tokens` already consumed

    def mark_span(sub_len: int, start: int) -> None:
        for i in range(start, start + sub_len):
            matched_word_spans.add(i)

    def find_and_mark(stemmed_sub: tuple[str, ...]) -> bool:
        n, m = len(stemmed_tokens), len(stemmed_sub)
        if m == 0 or m > n:
            return False
        for i in range(n - m + 1):
            if tuple(stemmed_tokens[i : i + m]) == stemmed_sub:
                mark_span(m, i)
                return True
        return False

    # Tier 1: direct canonical phrase match (longest phrases first so
    # multi-word symptoms are preferred over shorter overlapping ones).
    for symptom in sorted(known_symptoms, key=lambda s: -len(s.split())):
        stemmed_sub = _stem_phrase(symptom)
        if find_and_mark(stemmed_sub):
            if symptom not in detected:
                detected.append(symptom)

    # Tier 2: curated synonym phrases.
    for phrase, canonical in SYMPTOM_SYNONYMS.items():
        if canonical not in known_symptoms:
            continue
        stemmed_sub = _stem_phrase(_normalize(phrase))
        if find_and_mark(stemmed_sub):
            if canonical not in detected:
                detected.append(canonical)

    # Tier 3: conservative fuzzy fallback for single, unmatched words
    # (catches small typos like "feaver" -> "fever").
    single_word_symptoms = [s for s in known_symptoms if " " not in s]
    fuzzy_pool = list(dict.fromkeys(single_word_symptoms + list(SYMPTOM_SYNONYMS.keys())))

    for i, word in enumerate(tokens):
        if i in matched_word_spans:
            continue
        if len(word) < _MIN_FUZZY_WORD_LEN:
            continue
        candidates = difflib.get_close_matches(
            word, fuzzy_pool, n=1, cutoff=_FUZZY_CUTOFF
        )
        if not candidates:
            continue
        match = candidates[0]
        canonical = SYMPTOM_SYNONYMS.get(match, match)
        if canonical in known_symptoms and canonical not in detected:
            detected.append(canonical)
            matched_word_spans.add(i)

    return detected
