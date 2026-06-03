import re

# "Phrase (ACRONYM)" or "ACRONYM (Phrase)"
_DEF_PAT = re.compile(
    r'(?:([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,6})\s+\(([A-Z]{2,6})\)'
    r'|\b([A-Z]{2,6})\s*\(([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,6})\))')

_STOP = frozenset(
    'and or the of in on at to for with by from as is are was were be been '
    'have has had do does did will would could should may might must can shall'.split())


def synonyms_for_keys(keys, text):
    # build acronym <-> phrase map from explicit definitions in the doc
    acronym_to_phrase = {}
    for match in _DEF_PAT.finditer(text):
        if match.group(1):
            phrase, acronym = match.group(1), match.group(2)
        else:
            acronym, phrase = match.group(3), match.group(4)
        # need at least 2 non-stop words
        content_words = [w for w in phrase.split() if w.lower() not in _STOP]
        if len(content_words) >= 2:
            acronym_to_phrase[acronym] = phrase
            acronym_to_phrase[acronym.lower()] = phrase

    # collect uppercase tokens 2-6 chars that could be acronyms
    upper_tokens = {t for t in re.findall(r'\b[A-Z]{2,6}\b', text)
                    if t not in ('AM', 'PM')}

    result = {}
    for key in keys:
        key_lower = key.lower().strip()
        words = [w for w in key_lower.split() if w not in _STOP]
        synonyms = []

        # single-word key that might be an acronym -> look up spelled-out form
        if len(words) == 1:
            compact = re.sub(r'[^a-z0-9]', '', key_lower)
            if compact in acronym_to_phrase:
                synonyms.append(acronym_to_phrase[compact])

        # multi-word key -> check if its acronym appears in the doc
        elif len(words) >= 2:
            acronym = ''.join(w[0] for w in words).upper()
            if acronym in upper_tokens:
                synonyms.append(acronym)

        result[key] = [s for s in synonyms if s.lower() != key_lower]
    return result
