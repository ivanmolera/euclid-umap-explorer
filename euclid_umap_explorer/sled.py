"""Read-only, public-only SLED lookups, isolated from Streamlit's script thread."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from threading import RLock
import time
from urllib.parse import urlencode

import requests
from astropy.coordinates import SkyCoord
import astropy.units as u

SLED_ENDPOINT = "https://sled.amnh.org/api/query-lenses-full/"
SEARCH_RADIUS_ARCSEC = 5.0
LOOKUP_TIMEOUT_SECONDS = 8.0
HTTP_TIMEOUT = (2.0, 4.0)
MATCH_CACHE_SECONDS = 24 * 3600
EMPTY_CACHE_SECONDS = 6 * 3600
ERROR_CACHE_SECONDS = 30
MAX_RESPONSE_BYTES = 256 * 1024


@dataclass(frozen=True)
class SledQuery:
    ra: float
    dec: float
    radius_arcsec: float = SEARCH_RADIUS_ARCSEC

    def __post_init__(self) -> None:
        values = (self.ra, self.dec, self.radius_arcsec)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("SLED coordinates and radius must be valid numbers.")
        if not (0 <= self.ra < 360 and -90 <= self.dec <= 90):
            raise ValueError("SLED coordinates must be in degrees (J2000).")
        if not 0 < self.radius_arcsec <= 60:
            raise ValueError("SLED search radius must be between 0 and 60 arcsec.")

    def parameters(self) -> dict:
        return {
            "lens-ra_centre": self.ra,
            "lens-dec_centre": self.dec,
            "lens-radius": self.radius_arcsec / 3600,
            "management-access_level": "PUB",
            "download-lens_options": ["ra", "dec", "name", "flag", "access_level"],
            "download-related": ["papers"],
        }

    def search_url(self) -> str:
        params = {key: value for key, value in self.parameters().items()
                  if not key.startswith("download-")}
        return "https://sled.amnh.org/lenses/query/?" + urlencode(params)


@dataclass(frozen=True)
class SledPaper:
    year: str
    author: str
    title: str
    reference: str


@dataclass(frozen=True)
class SledMatch:
    name: str
    flag: str
    ra: float
    dec: float
    separation_arcsec: float
    papers: tuple[SledPaper, ...] = ()


@dataclass(frozen=True)
class SledResult:
    status: str
    matches: tuple[SledMatch, ...] = ()
    checked_at: str = ""
    message: str = ""


def credentials_available() -> bool:
    return bool(os.getenv("SLED_USERNAME") and os.getenv("SLED_PASSWORD"))


def _result(status: str, message: str = "", matches: tuple = ()) -> SledResult:
    return SledResult(status, matches, datetime.now(timezone.utc).isoformat(), message)


def parse_response(payload: object, query: SledQuery) -> SledResult:
    if not isinstance(payload, dict) or payload.get("errors") != []:
        raise ValueError("Invalid SLED response.")
    records = payload.get("lenses")
    if not isinstance(records, list) or len(records) > 100:
        raise ValueError("Invalid SLED lens list.")
    origin = SkyCoord(query.ra * u.deg, query.dec * u.deg)
    matches = []
    for record in records:
        # Fail closed if the API ever ignores the public-only filter.
        if not isinstance(record, dict) or record.get("access_level") != "PUB":
            raise ValueError("SLED did not return explicitly public records.")
        position = SledQuery(float(record["ra"]), float(record["dec"]))
        separation = float(origin.separation(
            SkyCoord(position.ra * u.deg, position.dec * u.deg)
        ).arcsec)
        if separation > query.radius_arcsec + 1e-6:
            raise ValueError("SLED returned a record outside the search cone.")
        flag = str(record.get("flag", "")).upper()
        if flag not in {"CONFIRMED", "CANDIDATE", "CONTAMINANT"}:
            flag = "UNSPECIFIED"
        paper_records = record.get("papers", [])
        if not isinstance(paper_records, list):
            raise ValueError("Invalid SLED publications.")
        papers = tuple(
            SledPaper(str(p.get("year", ""))[:10],
                      str(p.get("first_author", ""))[:200],
                      str(p.get("title", ""))[:500],
                      str(p.get("cite_as", ""))[:200])
            for p in paper_records[:20] if isinstance(p, dict)
        )
        matches.append(SledMatch(str(record.get("name", "Unnamed"))[:200], flag,
                                 position.ra, position.dec, separation, papers))
    matches.sort(key=lambda match: match.separation_arcsec)
    return _result("matched" if matches else "no_match", matches=tuple(matches))


def query_sled(query: SledQuery) -> SledResult:
    """Only called by a worker. Neither credentials nor response bodies are logged."""
    username, password = os.getenv("SLED_USERNAME"), os.getenv("SLED_PASSWORD")
    if not username or not password:
        return _result("unavailable", "SLED credentials are not configured on the server.")
    deadline = time.monotonic() + LOOKUP_TIMEOUT_SECONDS
    try:
        with requests.post(
            SLED_ENDPOINT, json=query.parameters(),
            auth=requests.auth.HTTPBasicAuth(username, password),
            headers={"User-Agent": "ESCOPE-SLED/1.0", "Accept": "application/json"},
            timeout=HTTP_TIMEOUT, allow_redirects=False, stream=True,
        ) as response:
            if response.headers.get("cf-mitigated") == "challenge":
                return _result("error", "SLED's access protection blocked this server request. API access may need authorization by SLED.")
            if response.status_code == 401:
                return _result("error", "SLED denied access. Check the server credentials and account permissions.")
            if response.status_code == 403:
                return _result("error", "SLED denied access. Check account permissions or server-side access restrictions.")
            if response.status_code == 429:
                return _result("error", "SLED is limiting requests. Please try again later.")
            if response.status_code != 200:
                return _result("error", "SLED is temporarily unavailable. Please try again later.")
            body = bytearray()
            for chunk in response.iter_content(chunk_size=8192):
                if time.monotonic() >= deadline:
                    return _result("timeout", "SLED did not respond within the time limit.")
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES:
                    return _result("error", "SLED returned an unexpectedly large response.")
            return parse_response(json.loads(body), query)
    except requests.exceptions.Timeout:
        return _result("timeout", "SLED did not respond within the time limit.")
    except requests.exceptions.RequestException:
        return _result("error", "SLED could not be reached. Please try again later.")
    except (KeyError, TypeError, ValueError):
        return _result("error", "SLED returned an unexpected response. No match decision was made.")


@dataclass
class _Pending:
    future: Future
    deadline: float
    expired: bool = False


@dataclass
class _Cached:
    result: SledResult
    expires: float
    refresh_after: float


class SledClient:
    """Bounded per-process cache and workers, with no waiting on the caller thread."""

    def __init__(self, *, workers: int = 4, max_entries: int = 512,
                 lookup=query_sled, clock=time.monotonic) -> None:
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sled")
        self._workers = workers
        self._max_entries = max_entries
        self._lookup = lookup
        self._clock = clock
        self._lock = RLock()
        self._pending: dict[SledQuery, _Pending] = {}
        self._cache: OrderedDict[SledQuery, _Cached] = OrderedDict()

    def _work(self, query: SledQuery) -> tuple[float, SledResult]:
        try:
            result = self._lookup(query)
        except Exception:
            result = _result("error", "SLED could not complete this check. Please try again later.")
        return self._clock(), result

    def _save(self, query: SledQuery, result: SledResult, now: float) -> None:
        ttl = (MATCH_CACHE_SECONDS if result.status == "matched" else
               EMPTY_CACHE_SECONDS if result.status == "no_match" else ERROR_CACHE_SECONDS)
        self._cache[query] = _Cached(result, now + ttl,
                                     now + (5 if result.status in {"matched", "no_match"} else ttl))
        self._cache.move_to_end(query)
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    def _collect(self, now: float) -> None:
        for query, pending in list(self._pending.items()):
            if pending.future.done():
                finished, result = pending.future.result()  # Already done: never waits.
                if not pending.expired:
                    if finished > pending.deadline:
                        result = _result("timeout", "SLED did not respond within the time limit.")
                    self._save(query, result, finished)
                del self._pending[query]
            elif now >= pending.deadline and not pending.expired:
                self._save(query, _result("timeout", "SLED did not respond within the time limit."), now)
                pending.expired = True
        for query, cached in list(self._cache.items()):
            if now >= cached.expires:
                del self._cache[query]

    def get(self, query: SledQuery) -> SledResult | None:
        with self._lock:
            self._collect(self._clock())
            if query in self._pending and not self._pending[query].expired:
                return SledResult("pending")
            if query in self._cache:
                self._cache.move_to_end(query)
                return self._cache[query].result
            if query in self._pending:
                return _result("timeout", "SLED did not respond within the time limit.")
            return None

    def request(self, query: SledQuery, *, refresh: bool = False) -> SledResult:
        with self._lock:
            now = self._clock()
            result = self.get(query)
            cached = self._cache.get(query)
            if query in self._pending:
                return result
            if cached and (not refresh or now < cached.refresh_after):
                return cached.result
            if not credentials_available():
                return _result("unavailable", "SLED credentials are not configured on the server.")
            # No unbounded executor queue, even if a DNS/TLS operation stalls.
            if len(self._pending) >= self._workers:
                return SledResult("busy", message="Other SLED checks are running. Please try again shortly.")
            self._pending[query] = _Pending(
                self._executor.submit(self._work, query), now + LOOKUP_TIMEOUT_SECONDS
            )
            return SledResult("pending")

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
