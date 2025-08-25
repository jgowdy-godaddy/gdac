from __future__ import annotations
import time, re, httpx, urllib.parse

"""
Tools:
- fetch_url: Fetch a URL (text focus), auto-upgrade http->https, 15min cache, and cross-host redirect reporting.
  Args: {"url": "...", "timeout_ms": 20000, "max_bytes": 200000}
  Returns:
    - On success: "status=<code>\n<body-truncated>"
    - On cross-host redirect: "REDIRECT host=<new-host> url=<final-url>"
- search_web: Simple HTML search (DuckDuckGo HTML by default). Override with env:
    AGENT_WEB_SEARCH_URL, AGENT_WEB_SEARCH_QS
"""

_CACHE: dict[str, tuple[float, str]] = {}   # url -> (timestamp, data)
_TTL_SEC = 900

def _cache_get(url: str) -> str | None:
    t = _CACHE.get(url)
    if not t: return None
    ts, data = t
    if time.time() - ts > _TTL_SEC:
        _CACHE.pop(url, None)
        return None
    return data

def _cache_put(url: str, data: str) -> None:
    _CACHE[url] = (time.time(), data)

def fetch_url(repo: str, url: str, timeout_ms: int = 20000, max_bytes: int = 200_000) -> str:
    # auto-upgrade to https for plain http
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme == "http":
            url = urllib.parse.urlunparse(parsed._replace(scheme="https"))
    except Exception:
        pass

    cached = _cache_get(url)
    if cached is not None:
        return cached

    try:
        with httpx.Client(follow_redirects=False, timeout=timeout_ms/1000) as client:
            r = client.get(url)
            # Cross-host redirect? Tell caller to re-invoke with final URL.
            if r.is_redirect:
                loc = r.headers.get("location", "")
                try:
                    final = urllib.parse.urljoin(url, loc)
                    new_host = urllib.parse.urlparse(final).netloc
                    old_host = urllib.parse.urlparse(url).netloc
                    if new_host and new_host != old_host:
                        msg = f"REDIRECT host={new_host} url={final}"
                        _cache_put(url, msg)
                        return msg
                except Exception:
                    pass
                # same-host redirect; follow once
                r = client.get(final, follow_redirects=True)

        ctype = r.headers.get("content-type","").lower()
        if not ctype.startswith("text/") and "html" not in ctype:
            return f"UNSUPPORTED_CONTENT: {ctype}"
        body = r.text
        if len(body) > max_bytes:
            body = body[:max_bytes] + "…"
        out = f"status={r.status_code}\n{body}"
        _cache_put(url, out)
        return out
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

def search_web(repo: str, query: str, timeout_ms: int = 20000) -> str:
    import os, html
    base = os.environ.get("AGENT_WEB_SEARCH_URL", "https://duckduckgo.com/html/")
    qk = os.environ.get("AGENT_WEB_SEARCH_QS", "q")
    try:
        r = httpx.get(base, params={qk: query}, timeout=timeout_ms/1000)
        # Extract top result titles + links (best-effort)
        results = re.findall(r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', r.text, re.I)
        lines = [f"{i+1}. {re.sub('<.*?>','', html.unescape(title))}\n{link}" for i,(link,title) in enumerate(results[:10])]
        return "\n".join(lines) if lines else f"status={r.status_code}\nNo results parsed."
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"