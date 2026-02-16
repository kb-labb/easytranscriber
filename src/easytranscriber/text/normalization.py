import unicodedata

from easyaligner.text.normalization import SpanMapNormalizer


def text_normalizer(text: str) -> str:
    """
    Default text normalization function.

    Applies
        - Unicode normalization (NFKC)
        - Lowercasing
        - Normalization of whitespace
        - Remove parentheses and special characters

    Parameters
    ----------
    text : str
        Input text to normalize.

    Returns
    -------
    tuple
        Tuple containing (normalized_tokens, mapping).
    """
    # Unicode normalization
    normalizer = SpanMapNormalizer(text)
    # # Remove parentheses, brackets, stars, and their content
    # normalizer.transform(r"\(.*?\)", "")
    # normalizer.transform(r"\[.*?\]", "")
    # normalizer.transform(r"\*.*?\*", "")

    # Unicode normalization on tokens and lowercasing
    normalizer.transform(r"\S+", lambda m: unicodedata.normalize("NFKC", m.group()))
    normalizer.transform(r"\S+", lambda m: m.group().lower())
    normalizer.transform(r"[^\w\s]", "")  # Remove punctuation and special characters
    normalizer.transform(r"\s+", " ")  # Normalize whitespace to a single space
    normalizer.transform(r"^\s+|\s+$", "")  # Strip leading and trailing whitespace

    mapping = normalizer.get_token_map()
    normalized_tokens = [item["normalized_token"] for item in mapping]
    return normalized_tokens, mapping
