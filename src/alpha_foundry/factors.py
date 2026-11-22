from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .dsl import neutralize_cs, pct_change, safe_div


@dataclass(frozen=True)
class FactorSpec:
    spec_id: str
    name: str
    family: str
    hypothesis: str
    formula: str
    direction: float
    window: int
    category: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FAMILY_LIBRARY: list[dict[str, Any]] = [
    {
        "family": "ret",
        "prefix": "rev",
        "name": "短期反转",
        "windows": [3, 5, 10, 20],
        "direction": -1,
        "category": "reversal",
        "hypothesis": "过去 N 日涨幅过高的股票容易发生短期均值回归。",
    },
    {
        "family": "ret",
        "prefix": "mom",
        "name": "中期动量",
        "windows": [20, 60, 120],
        "direction": 1,
        "category": "momentum",
        "hypothesis": "中期相对强势具有持续性，但需控制反转和拥挤风险。",
    },
    {
        "family": "vol",
        "prefix": "lowvol",
        "name": "低波动",
        "windows": [10, 20, 60],
        "direction": -1,
        "category": "risk",
        "hypothesis": "高波动股票承担更多噪声风险，低波动组合有风险调整收益优势。",
    },
    {
        "family": "volume_ratio",
        "prefix": "volshock",
        "name": "成交量冲击",
        "windows": [5, 20, 60],
        "direction": -1,
        "category": "liquidity",
        "hypothesis": "异常放量往往伴随注意力交易和短期反转。",
    },
    {
        "family": "volume_trend",
        "prefix": "voltrend",
        "name": "成交量趋势",
        "windows": [10, 20, 60],
        "direction": 1,
        "category": "liquidity",
        "hypothesis": "成交量温和扩张可能确认价格趋势。",
    },
    {
        "family": "corr_ret_vol",
        "prefix": "corrretvol",
        "name": "收益-成交量相关",
        "windows": [10, 20, 60],
        "direction": -1,
        "category": "liquidity",
        "hypothesis": "收益与成交量高度同向可能反映拥挤交易，后续收益偏弱。",
    },
    {
        "family": "corr_close_vol",
        "prefix": "corrclosevol",
        "name": "价格-成交量相关",
        "windows": [20, 60],
        "direction": -1,
        "category": "liquidity",
        "hypothesis": "价格与成交量同步上升可能意味着情绪驱动，存在反转风险。",
    },
    {
        "family": "amihud",
        "prefix": "amihud",
        "name": "Amihud 非流动性",
        "windows": [20, 60],
        "direction": 1,
        "category": "liquidity",
        "hypothesis": "单位成交额带来的价格冲击越大，流动性溢价越高。",
    },
    {
        "family": "range",
        "prefix": "rangevol",
        "name": "日内振幅",
        "windows": [5, 20, 60],
        "direction": -1,
        "category": "risk",
        "hypothesis": "高振幅反映不确定性和投机交易，风险调整后收益偏弱。",
    },
    {
        "family": "clv",
        "prefix": "clv",
        "name": "收盘位置",
        "windows": [5, 20],
        "direction": 1,
        "category": "microstructure",
        "hypothesis": "收盘价位于日内区间上部，反映买盘承接较强。",
    },
    {
        "family": "max_ret",
        "prefix": "maxret",
        "name": "极端正收益",
        "windows": [20, 60],
        "direction": -1,
        "category": "lottery",
        "hypothesis": "近期出现极端正收益的股票具有彩票偏好，后续收益偏弱。",
    },
    {
        "family": "drawdown",
        "prefix": "drawdown",
        "name": "回撤",
        "windows": [20, 60, 120],
        "direction": 1,
        "category": "trend",
        "hypothesis": "距离区间高点越近，趋势和资金承接越强。",
    },
    {
        "family": "ma_dist",
        "prefix": "madist",
        "name": "均线偏离",
        "windows": [5, 20, 60],
        "direction": -1,
        "category": "reversal",
        "hypothesis": "价格相对均线过度偏离后存在均值回归。",
    },
    {
        "family": "gap",
        "prefix": "gaprev",
        "name": "隔夜跳空反转",
        "windows": [1, 5, 20],
        "direction": -1,
        "category": "reversal",
        "hypothesis": "隔夜跳空容易受到情绪和流动性冲击，随后部分反转。",
    },
    {
        "family": "intraday",
        "prefix": "intraday",
        "name": "日内动量",
        "windows": [5, 20],
        "direction": 1,
        "category": "momentum",
        "hypothesis": "日内收益具有短期持续性，反映订单流方向。",
    },
    {
        "family": "vwap_ret",
        "prefix": "vwapret",
        "name": "成交量加权收益",
        "windows": [5, 20, 60],
        "direction": 1,
        "category": "momentum",
        "hypothesis": "成交量加权的收益更接近真实资金方向。",
    },
    {
        "family": "vol_of_volume",
        "prefix": "volofvol",
        "name": "成交量波动",
        "windows": [20, 60],
        "direction": -1,
        "category": "liquidity",
        "hypothesis": "成交量不稳定反映交易拥挤和不确定性，后续收益偏弱。",
    },
    {
        "family": "skew",
        "prefix": "skew",
        "name": "收益偏度",
        "windows": [20, 60],
        "direction": -1,
        "category": "lottery",
        "hypothesis": "正偏度股票具有彩票特征，风险调整后收益偏低。",
    },
    {
        "family": "beta",
        "prefix": "lowbeta",
        "name": "低 Beta",
        "windows": [60, 120],
        "direction": -1,
        "category": "risk",
        "hypothesis": "高 Beta 股票在高波动环境中承担更高风险，低 Beta 有防御价值。",
    },
    {
        "family": "idio_vol",
        "prefix": "idiovol",
        "name": "特质波动",
        "windows": [20, 60],
        "direction": -1,
        "category": "risk",
        "hypothesis": "高特质波动股票存在套利限制和估值折价。",
    },
    {
        "family": "range_pos",
        "prefix": "rangepos",
        "name": "区间位置",
        "windows": [20, 60, 120],
        "direction": 1,
        "category": "momentum",
        "hypothesis": "价格位于区间上部时，趋势延续概率更高。",
    },
    {
        "family": "price_accel",
        "prefix": "priceaccel",
        "name": "价格加速度",
        "windows": [20],
        "direction": -1,
        "category": "reversal",
        "hypothesis": "短周期相对长周期的过度加速容易反转。",
    },
    {
        "family": "mom_vol",
        "prefix": "momvol",
        "name": "量价动量",
        "windows": [20, 60],
        "direction": 1,
        "category": "momentum",
        "hypothesis": "有成交量确认的价格动量更可能持续。",
    },
    {
        "family": "rev_vol",
        "prefix": "revvol",
        "name": "量价反转",
        "windows": [5, 20],
        "direction": -1,
        "category": "reversal",
        "hypothesis": "放量上涨后的短期反转更强。",
    },
    {
        "family": "overnight_mom",
        "prefix": "overnightmom",
        "name": "隔夜动量",
        "windows": [5, 20],
        "direction": 1,
        "category": "momentum",
        "hypothesis": "隔夜收益反映信息消化方向，具有短期持续性。",
    },
    {
        "family": "intraday_mom",
        "prefix": "intradaymom",
        "name": "日内动量均值",
        "windows": [5, 20],
        "direction": 1,
        "category": "momentum",
        "hypothesis": "日内订单流方向在短期内可能延续。",
    },
    {
        "family": "dist_high",
        "prefix": "disthigh",
        "name": "距区间高点",
        "windows": [60, 120],
        "direction": 1,
        "category": "trend",
        "hypothesis": "接近区间高点的股票更容易获得趋势资金关注。",
    },
    {
        "family": "dist_low",
        "prefix": "distlow",
        "name": "距区间低点",
        "windows": [60, 120],
        "direction": -1,
        "category": "reversal",
        "hypothesis": "接近区间低点的股票可能存在超跌反弹。",
    },
    {
        "family": "turnover_shock",
        "prefix": "turnovershock",
        "name": "换手冲击",
        "windows": [5, 20, 60],
        "direction": -1,
        "category": "liquidity",
        "hypothesis": "换手率异常放大后，交易拥挤和反转风险上升。",
    },
    {
        "family": "liquidity",
        "prefix": "illiquidity",
        "name": "低流动性溢价",
        "windows": [20, 60, 120],
        "direction": 1,
        "category": "liquidity",
        "hypothesis": "低流动性股票需要更高预期收益补偿交易成本。",
    },
    {
        "family": "residual_mom",
        "prefix": "residmom",
        "name": "残差动量",
        "windows": [20, 60, 120],
        "direction": 1,
        "category": "momentum",
        "hypothesis": "剥离 Beta 和流动性后的动量更接近特质 alpha。",
    },
]


def _format_formula(family: str, window: int, direction: float) -> str:
    sign = "" if direction > 0 else "-"
    formulas = {
        "ret": f"close / delay(close, {window}) - 1",
        "vol": f"ts_std(ret, {window})",
        "volume_ratio": f"volume / ts_mean(volume, {window})",
        "volume_trend": f"ts_mean(volume, 5) / ts_mean(volume, {window}) - 1",
        "corr_ret_vol": f"ts_corr(ret, volume, {window})",
        "corr_close_vol": f"ts_corr(close, volume, {window})",
        "amihud": f"ts_mean(abs(ret) / dollar_volume, {window}) * 1e9",
        "range": f"ts_mean((high - low) / close, {window})",
        "clv": f"ts_mean((2*close - high - low) / (high - low), {window})",
        "max_ret": f"ts_max(ret, {window})",
        "drawdown": f"close / ts_max(close, {window}) - 1",
        "ma_dist": f"close / ts_mean(close, {window}) - 1",
        "gap": f"ts_mean(open / delay(close, 1) - 1, {window})",
        "intraday": f"ts_mean(close / open - 1, {window})",
        "vwap_ret": f"ts_sum(ret * volume, {window}) / ts_sum(volume, {window})",
        "vol_of_volume": f"ts_std(volume, {window}) / ts_mean(volume, {window})",
        "skew": f"ts_skew(ret, {window})",
        "beta": f"rolling_beta(ret, market_ret, {window})",
        "idio_vol": f"ts_std(ret - beta_60 * market_ret, {window})",
        "range_pos": f"(close - ts_min(low, {window})) / (ts_max(high, {window}) - ts_min(low, {window}))",
        "price_accel": "ret_5 - ret_20",
        "mom_vol": f"ret_{window} * volume / ts_mean(volume, 20)",
        "rev_vol": f"ret_{window} * volume / ts_mean(volume, 20)",
        "overnight_mom": f"ts_mean(open / delay(close, 1) - 1, {window})",
        "intraday_mom": f"ts_mean(close / open - 1, {window})",
        "dist_high": f"close / ts_max(close, {window}) - 1",
        "dist_low": f"close / ts_min(close, {window}) - 1",
        "turnover_shock": f"dollar_volume / ts_mean(dollar_volume, {window}) - 1",
        "liquidity": f"-log(ts_mean(dollar_volume, {window}))",
        "residual_mom": f"neutralize(ret_{window}, [beta_60, log_adv20])",
    }
    return f"{sign}({formulas[family]})"


def generate_factor_specs() -> list[FactorSpec]:
    specs: list[FactorSpec] = []
    for item in FAMILY_LIBRARY:
        for window in item["windows"]:
            spec_id = f"{item['prefix']}_{window}"
            specs.append(
                FactorSpec(
                    spec_id=spec_id,
                    name=f"{item['name']} {window}D",
                    family=item["family"],
                    hypothesis=item["hypothesis"],
                    formula=_format_formula(item["family"], window, item["direction"]),
                    direction=float(item["direction"]),
                    window=int(window),
                    category=item["category"],
                    params={"prefix": item["prefix"], "direction": item["direction"]},
                )
            )
    seen: set[str] = set()
    for spec in specs:
        if spec.spec_id in seen:
            raise RuntimeError(f"Duplicate factor id: {spec.spec_id}")
        seen.add(spec.spec_id)
    return specs


def _ret_n(fields: dict[str, pd.DataFrame], window: int) -> pd.DataFrame:
    key = f"ret_{window}"
    return fields[key] if key in fields else pct_change(fields["close"], periods=window)


def _rolling_corr(
    left: pd.DataFrame, right: pd.DataFrame, window: int
) -> pd.DataFrame:
    return left.rolling(window, min_periods=window).corr(right)


def compute_base_factor(
    family: str, window: int, fields: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    close = fields["close"]
    open_ = fields["open"]
    high = fields["high"]
    low = fields["low"]
    volume = fields["volume"]
    ret = fields["ret"]
    dollar_volume = fields["dollar_volume"]

    if family == "ret":
        return _ret_n(fields, window)
    if family == "vol":
        return ret.rolling(window, min_periods=window).std()
    if family == "volume_ratio":
        return safe_div(volume, volume.rolling(window, min_periods=window).mean())
    if family == "volume_trend":
        short = volume.rolling(min(5, window), min_periods=min(5, window)).mean()
        long = volume.rolling(window, min_periods=window).mean()
        return safe_div(short, long) - 1.0
    if family == "corr_ret_vol":
        return _rolling_corr(ret, volume, window)
    if family == "corr_close_vol":
        return _rolling_corr(close, volume, window)
    if family == "amihud":
        return (safe_div(ret.abs(), dollar_volume)).rolling(
            window, min_periods=window
        ).mean() * 1e9
    if family == "range":
        return ((high - low) / close).rolling(window, min_periods=window).mean()
    if family == "clv":
        clv = safe_div(2.0 * close - high - low, high - low)
        return clv.rolling(window, min_periods=window).mean()
    if family == "max_ret":
        return ret.rolling(window, min_periods=window).max()
    if family == "drawdown":
        return safe_div(close, close.rolling(window, min_periods=window).max()) - 1.0
    if family == "ma_dist":
        return safe_div(close, close.rolling(window, min_periods=window).mean()) - 1.0
    if family == "gap":
        gap = safe_div(open_, close.shift(1)) - 1.0
        return gap.rolling(window, min_periods=window).mean()
    if family == "intraday":
        intraday = safe_div(close, open_) - 1.0
        return intraday.rolling(window, min_periods=window).mean()
    if family == "vwap_ret":
        numerator = (ret * volume).rolling(window, min_periods=window).sum()
        denominator = volume.rolling(window, min_periods=window).sum()
        return safe_div(numerator, denominator)
    if family == "vol_of_volume":
        return safe_div(
            volume.rolling(window, min_periods=window).std(),
            volume.rolling(window, min_periods=window).mean(),
        )
    if family == "skew":
        return ret.rolling(window, min_periods=window).skew()
    if family == "beta":
        key = f"beta_{window}"
        if key in fields:
            return fields[key]
        market = fields["market_ret"].iloc[:, 0]
        cov = ret.rolling(window, min_periods=window).cov(market)
        return cov.div(market.rolling(window, min_periods=window).var(), axis=0)
    if family == "idio_vol":
        return fields["idio_ret"].rolling(window, min_periods=window).std()
    if family == "range_pos":
        low_n = low.rolling(window, min_periods=window).min()
        high_n = high.rolling(window, min_periods=window).max()
        return safe_div(close - low_n, high_n - low_n)
    if family == "price_accel":
        return fields["ret_5"] - fields["ret_20"]
    if family in {"mom_vol", "rev_vol"}:
        volume_ratio = safe_div(volume, volume.rolling(20, min_periods=20).mean())
        return _ret_n(fields, window) * volume_ratio
    if family == "overnight_mom":
        gap = safe_div(open_, close.shift(1)) - 1.0
        return gap.rolling(window, min_periods=window).mean()
    if family == "intraday_mom":
        intraday = safe_div(close, open_) - 1.0
        return intraday.rolling(window, min_periods=window).mean()
    if family == "dist_high":
        return safe_div(close, close.rolling(window, min_periods=window).max()) - 1.0
    if family == "dist_low":
        return safe_div(close, close.rolling(window, min_periods=window).min()) - 1.0
    if family == "turnover_shock":
        return safe_div(dollar_volume, dollar_volume.rolling(window, min_periods=window).mean()) - 1.0
    if family == "liquidity":
        adv = dollar_volume.rolling(window, min_periods=window).mean()
        return -np.log(adv.clip(lower=1.0))
    if family == "residual_mom":
        controls = [fields["beta_60"], fields["log_adv20"]]
        return neutralize_cs(_ret_n(fields, window), controls)
    raise ValueError(f"Unknown factor family: {family}")


def compute_factor(spec: FactorSpec, fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
    factor = compute_base_factor(spec.family, spec.window, fields)
    factor = factor.replace([np.inf, -np.inf], np.nan)
    if spec.direction < 0:
        factor = -factor
    return factor
