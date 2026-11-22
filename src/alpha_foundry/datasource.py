from __future__ import annotations

import logging
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

SINA_LIST_URL = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeData"
)
TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://finance.sina.com.cn/",
}


class SinaUniverseClient:
    """Fetch a liquid A-share universe from Sina's public quote endpoint."""

    def __init__(self, session: requests.Session | None = None, timeout: int = 30):
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)
        self.timeout = timeout

    def fetch_page(self, page: int, num: int = 100) -> list[dict]:
        params = {
            "page": page,
            "num": num,
            "sort": "amount",
            "asc": 0,
            "node": "hs_a",
            "symbol": "",
            "_s_r_a": "page",
        }
        response = self.session.get(SINA_LIST_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected Sina universe response: {str(payload)[:200]}")
        return payload

    def fetch_universe(self, size: int) -> pd.DataFrame:
        rows: list[dict] = []
        pages = max(1, math.ceil(size / 100))
        for page in range(1, pages + 1):
            page_rows = self.fetch_page(page=page, num=min(100, size - len(rows)))
            rows.extend(page_rows)
            if len(page_rows) < 100:
                break
            time.sleep(0.2)

        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError("Sina returned an empty universe")

        keep = [
            "symbol",
            "code",
            "name",
            "trade",
            "amount",
            "mktcap",
            "nmc",
            "turnoverratio",
            "per",
            "pb",
        ]
        for col in keep:
            if col not in frame.columns:
                frame[col] = None
        frame = frame[keep].drop_duplicates(subset=["symbol"]).head(size).copy()
        frame["symbol"] = frame["symbol"].astype(str).str.lower()
        frame = frame[frame["symbol"].str.match(r"^(sh|sz)\d{6}$", na=False)]
        for col in ("trade", "amount", "mktcap", "nmc", "turnoverratio", "per", "pb"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["rank_by_amount"] = range(1, len(frame) + 1)
        return frame.reset_index(drop=True)


class TencentPriceClient:
    """Download forward-adjusted daily bars from Tencent's public K-line API."""

    def __init__(self, session: requests.Session | None = None, timeout: int = 30):
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)
        self.timeout = timeout

    def _get_json(self, symbol: str, start: str, end: str, retries: int = 4) -> dict:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                params = {"param": f"{symbol},day,{start},{end},640,qfq"}
                response = self.session.get(
                    TENCENT_KLINE_URL, params=params, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                sleep_s = 0.4 * (2**attempt) + random.random() * 0.3
                time.sleep(sleep_s)
        raise RuntimeError(f"Failed to fetch {symbol} {start}..{end}: {last_error}")

    @staticmethod
    def _parse_chunk(payload: dict, symbol: str) -> pd.DataFrame:
        node = (payload.get("data") or {}).get(symbol, {})
        rows = node.get("qfqday") or node.get("day") or []
        if not rows:
            return pd.DataFrame()
        parsed = []
        for row in rows:
            if len(row) < 6:
                continue
            parsed.append(
                {
                    "date": row[0],
                    "open": row[1],
                    "close": row[2],
                    "high": row[3],
                    "low": row[4],
                    "volume": row[5],
                }
            )
        frame = pd.DataFrame(parsed)
        if frame.empty:
            return frame
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for col in ("open", "close", "high", "low", "volume"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame.dropna(subset=["date", "close"]).sort_values("date")

    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        start_ts = pd.Timestamp(start)
        cursor_end = pd.Timestamp(end)
        chunks: list[pd.DataFrame] = []
        seen_dates: set[pd.Timestamp] = set()

        while cursor_end >= start_ts:
            payload = self._get_json(
                symbol=symbol,
                start=start_ts.strftime("%Y-%m-%d"),
                end=cursor_end.strftime("%Y-%m-%d"),
            )
            chunk = self._parse_chunk(payload, symbol)
            if chunk.empty:
                break
            chunk = chunk[~chunk["date"].isin(seen_dates)]
            if not chunk.empty:
                chunks.append(chunk)
                seen_dates.update(chunk["date"].tolist())
            first_date = chunk["date"].min() if not chunk.empty else None
            if first_date is None or first_date <= start_ts:
                break
            next_end = first_date - pd.Timedelta(days=1)
            if next_end >= cursor_end:
                break
            cursor_end = next_end
            time.sleep(0.05)

        if not chunks:
            return pd.DataFrame(
                columns=["date", "open", "close", "high", "low", "volume"]
            )
        frame = (
            pd.concat(chunks, ignore_index=True)
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        frame = frame[(frame["date"] >= start_ts) & (frame["date"] <= pd.Timestamp(end))]
        return frame

    def fetch_cached(
        self,
        symbol: str,
        start: str,
        end: str,
        cache_dir: Path,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{symbol}.csv"
        if path.exists() and not force_refresh:
            frame = pd.read_csv(path, parse_dates=["date"])
            if not frame.empty:
                return frame
        frame = self.fetch_daily(symbol=symbol, start=start, end=end)
        if not frame.empty:
            frame.to_csv(path, index=False)
        return frame


def download_universe(
    output_dir: Path,
    size: int,
    force_refresh: bool = False,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "universe.csv"
    if path.exists() and not force_refresh:
        return pd.read_csv(path)
    universe = SinaUniverseClient().fetch_universe(size=size)
    universe.to_csv(path, index=False)
    LOGGER.info("Saved universe with %d stocks to %s", len(universe), path)
    return universe


def download_prices(
    universe: pd.DataFrame,
    start: str,
    end: str,
    cache_dir: Path,
    index_symbol: str = "sh000300",
    max_workers: int = 4,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    client = TencentPriceClient()
    cache_dir.mkdir(parents=True, exist_ok=True)
    symbols = universe["symbol"].dropna().drop_duplicates().tolist()
    frames: list[pd.DataFrame] = []

    def load_one(symbol: str) -> pd.DataFrame:
        frame = client.fetch_cached(
            symbol=symbol,
            start=start,
            end=end,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )
        if not frame.empty:
            frame = frame.copy()
            frame["symbol"] = symbol
        return frame

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(load_one, symbol): symbol for symbol in symbols}
        for idx, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                frame = future.result()
                if not frame.empty:
                    frames.append(frame)
                if idx % 20 == 0 or idx == len(futures):
                    LOGGER.info("Downloaded %d/%d price files", idx, len(futures))
            except Exception as exc:  # pragma: no cover - network dependent
                LOGGER.warning("Skipping %s: %s", symbol, exc)

    if not frames:
        raise RuntimeError("No price data downloaded")
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.drop_duplicates(subset=["date", "symbol"]).sort_values(
        ["date", "symbol"]
    )
    prices.to_parquet(cache_dir / "prices.parquet", index=False)

    index_frame = client.fetch_cached(
        symbol=index_symbol,
        start=start,
        end=end,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )
    if index_frame.empty:
        LOGGER.warning("Index %s returned no data", index_symbol)
    else:
        index_frame.to_parquet(cache_dir / "index.parquet", index=False)
    return prices, index_frame


def _baostock_symbol(symbol: str) -> str:
    symbol = symbol.lower()
    if symbol.startswith("sh"):
        return f"sh.{symbol[2:]}"
    if symbol.startswith("sz"):
        return f"sz.{symbol[2:]}"
    raise ValueError(f"Unsupported Baostock symbol: {symbol}")


def download_prices_baostock(
    universe: pd.DataFrame,
    start: str,
    end: str,
    cache_dir: Path,
    index_symbol: str = "sh000300",
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download forward-adjusted daily bars through Baostock.

    Baostock is a free A-share data service. Its socket API is more stable for
    bulk downloads than public quote endpoints and provides an amount field.
    """
    import baostock as bs

    cache_dir.mkdir(parents=True, exist_ok=True)
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"Baostock login failed: {login.error_msg}")

    fields = "date,open,high,low,close,volume,amount"
    frames: list[pd.DataFrame] = []
    symbols = universe["symbol"].dropna().drop_duplicates().tolist()

    def query_one(symbol: str, adjustflag: str) -> pd.DataFrame:
        cache_path = cache_dir / f"{symbol}.csv"
        if cache_path.exists() and not force_refresh:
            frame = pd.read_csv(cache_path, parse_dates=["date"])
            if not frame.empty:
                return frame
        rs = bs.query_history_k_data_plus(
            _baostock_symbol(symbol),
            fields,
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag=adjustflag,
        )
        if rs.error_code != "0":
            raise RuntimeError(f"Baostock query failed for {symbol}: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame(columns=fields.split(","))
        frame = pd.DataFrame(rows, columns=fields.split(","))
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for col in ("open", "high", "low", "close", "volume", "amount"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["date", "close"]).sort_values("date")
        frame.to_csv(cache_path, index=False)
        return frame

    try:
        for idx, symbol in enumerate(symbols, start=1):
            try:
                frame = query_one(symbol, adjustflag="2")
                if not frame.empty:
                    frame = frame.copy()
                    frame["symbol"] = symbol
                    frames.append(frame)
                if idx % 20 == 0 or idx == len(symbols):
                    LOGGER.info("Baostock downloaded %d/%d price files", idx, len(symbols))
            except Exception as exc:  # pragma: no cover - network dependent
                LOGGER.warning("Baostock skipped %s: %s", symbol, exc)
            time.sleep(0.03)

        if not frames:
            raise RuntimeError("No Baostock price data downloaded")
        prices = pd.concat(frames, ignore_index=True)
        prices = prices.drop_duplicates(subset=["date", "symbol"]).sort_values(
            ["date", "symbol"]
        )
        prices.to_parquet(cache_dir.parent / "prices" / "prices.parquet", index=False)

        index_frame = query_one(index_symbol, adjustflag="3")
        if not index_frame.empty:
            index_frame.to_parquet(cache_dir.parent / "prices" / "index.parquet", index=False)
        else:
            LOGGER.warning("Baostock returned no index data for %s", index_symbol)
        return prices, index_frame
    finally:
        bs.logout()


def load_cached_market_data(
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe_path = raw_dir / "universe.csv"
    prices_path = raw_dir / "prices" / "prices.parquet"
    index_path = raw_dir / "prices" / "index.parquet"
    missing = [str(p) for p in (universe_path, prices_path, index_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing cached market data: {missing}")
    universe = pd.read_csv(universe_path)
    prices = pd.read_parquet(prices_path)
    index_frame = pd.read_parquet(index_path)
    return universe, prices, index_frame
