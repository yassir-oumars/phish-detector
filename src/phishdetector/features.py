import math, re, tldextract
from urllib.parse import urlparse

SUSPICIOUS = ["@", "//@", "-", "%", "?", "&", "=", "login", "verify", "update"]

def shannon_entropy(s: str) -> float:
    if not s: return 0.0
    from collections import Counter
    counts = Counter(s)
    n = len(s)
    return -sum((c/n) * math.log2(c/n) for c in counts.values())

def url_lexical_features(url: str) -> dict:
    u = url.strip()
    parsed = urlparse(u)
    host = parsed.netloc or ""
    pathq = (parsed.path or "") + (("?" + parsed.query) if parsed.query else "")
    ext = tldextract.extract(u)
    domain = ".".join([p for p in [ext.domain, ext.suffix] if p])

    feats = {}
    feats["url_len"] = len(u)
    feats["host_len"] = len(host)
    feats["path_len"] = len(parsed.path or "")
    feats["num_dots"] = u.count(".")
    feats["num_digits"] = sum(ch.isdigit() for ch in u)
    feats["num_hyphens"] = u.count("-")
    feats["num_params"] = u.count("&")
    feats["has_ip"] = int(bool(re.fullmatch(r"https?://\d{1,3}(?:\.\d{1,3}){3}.*", u)))
    feats["entropy"] = shannon_entropy(u.lower())
    feats["tld_len"] = len(ext.suffix or "")
    feats["suspicious_tokens"] = sum(tok in u.lower() for tok in SUSPICIOUS)
    feats["at_in_host"] = int("@" in host)
    feats["path_is_long"] = int(len(pathq) > 40)
    return feats
