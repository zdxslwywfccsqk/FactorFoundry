from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .agents import AgentContext, FactorRun


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return "-"
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in frame.columns) + " |")
    return "\n".join(lines)


def _best_single(context: AgentContext) -> FactorRun | None:
    accepted = [r for r in context.results if r.status == "accepted"]
    pool = accepted or context.results
    if not pool:
        return None
    return max(pool, key=lambda r: r.validation.metrics.get("icir", -np.inf))


def plot_equity(context: AgentContext, figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    best = _best_single(context)
    portfolio = context.portfolio_metrics
    if portfolio:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
        for ax, split_name in zip(axes, ("validation", "test"), strict=True):
            if best is not None:
                series = best.validation.returns if split_name == "validation" else best.test.returns
                if not series.empty:
                    ax.plot((1 + series).cumprod().to_numpy(), label=f"Best single: {best.spec.spec_id}")
            series = portfolio[f"{split_name}_returns"]
            if not series.empty:
                ax.plot((1 + series).cumprod().to_numpy(), label="Multi-agent portfolio")
            ax.set_title(f"{split_name.title()} equity curve (net of costs)")
            ax.set_xlabel("Rebalance period")
            ax.set_ylabel("Growth of 1")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = figures_dir / "equity_curves.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path.name)
    return paths


def plot_factor_ic(context: AgentContext, figures_dir: Path, top_n: int = 18) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in context.results:
        rows.append(
            {
                "factor_id": result.spec.spec_id,
                "mean_ic": result.validation.metrics.get("mean_ic", np.nan),
                "icir": result.validation.metrics.get("icir", np.nan),
                "status": result.status,
            }
        )
    frame = pd.DataFrame(rows).dropna(subset=["mean_ic"])
    if frame.empty:
        return []
    top = frame.sort_values("mean_ic", ascending=False).head(top_n).iloc[::-1]
    colors = ["#2b6cb0" if status == "accepted" else "#a0aec0" for status in top["status"]]
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(top) + 2))
    ax.barh(top["factor_id"], top["mean_ic"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top validation Rank IC by factor")
    ax.set_xlabel("Mean Rank IC (validation)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = figures_dir / "validation_ic_top.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [path.name]


def plot_icir_distribution(context: AgentContext, figures_dir: Path) -> list[str]:
    values = np.array(
        [r.validation.metrics.get("icir", np.nan) for r in context.results], dtype=float
    )
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return []
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=25, color="#90cdf4", edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Distribution of validation ICIR across the search space")
    ax.set_xlabel("Validation ICIR")
    ax.set_ylabel("Factor count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figures_dir / "validation_icir_distribution.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [path.name]


def plot_selected_correlation(context: AgentContext, figures_dir: Path) -> list[str]:
    selected = [r for r in context.results if r.spec.spec_id in context.selected_ids]
    if len(selected) < 2:
        return []
    returns = pd.concat(
        {r.spec.spec_id: r.validation.returns for r in selected}, axis=1
    ).dropna(how="all")
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("Selected factor return correlation (validation)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = figures_dir / "selected_factor_correlation.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [path.name]


def build_summary(context: AgentContext, figures: list[str]) -> dict[str, Any]:
    results = context.results
    accepted = [r for r in results if r.status == "accepted"]
    portfolio = context.portfolio_metrics
    best = _best_single(context)
    summary: dict[str, Any] = {
        "project": context.config.project_name,
        "run_id": context.run_id,
        "data": context.metadata,
        "n_factor_candidates": len(results),
        "n_accepted": len(accepted),
        "critic": context.critic_summary,
        "selected_factors": context.selected_ids,
        "portfolio": {
            "fallback": portfolio.get("fallback") if portfolio else None,
            "validation": portfolio.get("validation") if portfolio else None,
            "test": portfolio.get("test") if portfolio else None,
            "random_baseline": portfolio.get("random_baseline") if portfolio else None,
            "ablation_no_critic": portfolio.get("ablation_no_critic") if portfolio else None,
        },
        "best_single_factor": {
            "factor_id": best.spec.spec_id if best else None,
            "name": best.spec.name if best else None,
            "validation": best.validation.metrics if best else None,
            "test": best.test.metrics if best else None,
        },
        "figures": figures,
    }
    return summary


def write_report(context: AgentContext) -> tuple[Path, Path]:
    output_dir = context.config.paths.output_dir
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []
    figures.extend(plot_equity(context, figures_dir))
    figures.extend(plot_factor_ic(context, figures_dir))
    figures.extend(plot_icir_distribution(context, figures_dir))
    figures.extend(plot_selected_correlation(context, figures_dir))

    summary = build_summary(context, figures)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)

    results = context.results
    rows = []
    for result in results:
        rows.append(
            {
                "factor": result.spec.spec_id,
                "category": result.spec.category,
                "val_ic": result.validation.metrics.get("mean_ic"),
                "val_icir": result.validation.metrics.get("icir"),
                "val_sharpe": result.validation.metrics.get("net_sharpe"),
                "val_turnover": result.validation.metrics.get("turnover_annual"),
                "test_ic": result.test.metrics.get("mean_ic"),
                "test_sharpe": result.test.metrics.get("net_sharpe"),
                "q": result.q_value,
                "dsr": result.dsr,
                "status": result.status,
            }
        )
    frame = pd.DataFrame(rows)
    top_validation = frame.sort_values("val_icir", ascending=False).head(15).copy()
    top_validation = top_validation.rename(
        columns={
            "val_ic": "ValIC",
            "val_icir": "ValICIR",
            "val_sharpe": "ValSharpe",
            "val_turnover": "ValTurnover",
            "test_ic": "TestIC",
            "test_sharpe": "TestSharpe",
        }
    )
    accepted_frame = frame[frame["status"] == "accepted"].copy()
    status_counts = frame["status"].value_counts().rename_axis("status").reset_index(name="count")

    portfolio = context.portfolio_metrics
    val = portfolio.get("validation", {}) if portfolio else {}
    test = portfolio.get("test", {}) if portfolio else {}
    random_baseline = portfolio.get("random_baseline", {}) if portfolio else {}
    best = _best_single(context)
    audit_run = sum(1 for r in results if "passed" in r.audit)
    audit_pass = sum(1 for r in results if r.audit.get("passed") is True)
    shuffle_run = sum(1 for r in results if "passed" in r.shuffle)
    shuffle_pass = sum(1 for r in results if r.shuffle.get("passed") is True)

    lines: list[str] = []
    lines.append(f"# {context.config.project_name} 研究报告")
    lines.append("")
    lines.append(f"生成日期：{date.today().isoformat()}  ")
    lines.append(f"Run ID：`{context.run_id}`  ")
    lines.append(f"数据指纹：`{context.metadata.get('data_fingerprint', 'n/a')}`")
    lines.append("")
    lines.append("## 1. 执行摘要")
    lines.append("")
    lines.append(
        "本项目实现了一个可审计的多智能体因子研究框架：模板化 Hypothesis Agent 生成经济假设，"
        "Factor Engineer 用受限表达式实现因子，Backtest Agent 做成本后横截面回测，"
        "Critic Agent 做多重检验、相关系数、未来数据扰动和标签打乱审计，Portfolio Agent 在验证集上选因子并锁定测试集。"
    )
    lines.append("")
    lines.append(f"- 股票池：{context.metadata.get('n_selected_symbols')} 只 A 股，区间 {context.metadata.get('start_date')} 至 {context.metadata.get('end_date')}。")
    lines.append(f"- 候选因子：{len(results)} 个；通过全部审计：{len(accepted_frame)} 个。")
    lines.append(
        f"- 未来数据扰动审计：通过 {audit_pass}/{audit_run}（仅对候选因子运行）；"
        f"标签打乱审计：通过 {shuffle_pass}/{shuffle_run}。"
    )
    lines.append(f"- 验证集 PBO（回测过拟合概率）：{_fmt(context.critic_summary.get('pbo'))}。")
    if portfolio:
        lines.append(
            f"- 最终组合测试集：年化收益 {_fmt(test.get('net_ann_return'))}，"
            f"净 Sharpe {_fmt(test.get('net_sharpe'))}，最大回撤 {_fmt(test.get('max_drawdown'))}，"
            f"年化换手 {_fmt(test.get('turnover_annual'))}。"
        )
        lines.append(
            f"- 与同一搜索空间内 {random_baseline.get('n_portfolios', 0)} 个随机组合相比，所选组合测试集 Sharpe 位于第 "
            f"{_fmt(100 * random_baseline.get('selected_percentile', float('nan')), 1)} 百分位。"
        )
    lines.append("")
    lines.append("## 2. 数据与股票池")
    lines.append("")
    meta = context.metadata
    lines.append(
        f"使用新浪公开行情接口获取活跃 A 股列表，使用 Baostock 前复权日线接口获取历史行情。"
        f"新浪返回 {meta.get('requested_universe_size')} 只股票，其中 {meta.get('n_raw_symbols')} 只有可用行情；经过历史长度、成交额、价格和缺失率筛选后保留 "
        f"{meta.get('n_selected_symbols')} 只。历史长度、成交额和价格筛选只使用 {meta.get('selection_end_date', '2020-12-31')} 之前的数据；缺失率条件当前按全样本计算，是已知的 PIT 风险。"
    )
    lines.append("")
    lines.append("**已知限制：** 当前股票池来自当前活跃股票列表，仍存在幸存者偏差；财务数据和行业分类未纳入 MVP；交易成本采用固定 bps 假设，未做完整市场冲击模型。")
    lines.append("")
    lines.append("## 3. 多智能体架构")
    lines.append("")
    lines.append("| Agent | 职责 | 关键约束 |")
    lines.append("|---|---|---|")
    lines.append("| Data Steward | PIT 面板、可交易性、数据质量 | 只读版本化数据 |")
    lines.append("| Planner | 研究议程、训练/验证/测试隔离 | 测试集锁定 |")
    lines.append("| Hypothesis | 经济机制和因子模板 | 结构化假设 |")
    lines.append("| Factor Engineer | 因子表达式实现 | 只使用历史滚动窗口 |")
    lines.append("| Backtest | 成本后横截面多空回测 | 统一评估协议 |")
    lines.append("| Critic | FDR、DSR、PBO、未来数据扰动、标签打乱 | 独立对抗式审查 |")
    lines.append("| Portfolio | 低相关因子组合 | 只在验证集选择 |")
    lines.append("| Librarian | SQLite/CSV 实验血缘 | 全链路记录 |")
    lines.append("")
    lines.append("说明：本 demo 默认使用确定性的经济假设模板库，保证可复现和零 API 成本；`HypothesisAgent` 预留了替换为 LLM 结构化生成的接口。")
    lines.append("")
    lines.append("## 4. 验证协议")
    lines.append("")
    lines.append("- 标签：`t` 日收盘至 `t+5` 日收盘的收益；信号使用 `t` 日及之前数据，隐含在 `t` 日收盘附近成交，存在轻微执行乐观偏差。")
    if context.config.backtest.portfolio_mode == "rank":
        lines.append(
            f"- 组合：每 {context.config.backtest.horizon} 日按截面排序构造 rank-weighted 多空组合（总敞口 2）。"
        )
    else:
        lines.append(
            f"- 组合：每 {context.config.backtest.horizon} 日截面排序，多空各 "
            f"{100 * context.config.backtest.quantile:.0f}%，等权。"
        )
    lines.append(f"- 成本：单边 {context.config.backtest.cost_bps_per_side:.0f} bps，按换手率扣减。")
    lines.append(
        f"- 训练：{context.config.splits.train_start} 至 {context.config.splits.train_end}；"
        f"验证：{context.config.splits.validation_start} 至 {context.config.splits.validation_end}；"
        f"测试：{context.config.splits.test_start} 至 {context.config.splits.test_end}。"
    )
    lines.append("- 多重检验：Benjamini-Hochberg FDR；Deflated Sharpe Ratio；PBO。")
    lines.append("- 泄漏审计：把 cutoff 之后的数据替换为噪声，检查 cutoff 当日因子值是否变化。")
    lines.append("- 标签审计：在截面内打乱未来收益，检查因子 IC 是否显著高于打乱后的分布。")
    lines.append("")
    lines.append("## 5. 搜索结果")
    lines.append("")
    lines.append("### 5.1 验证集 ICIR 前 15 名")
    lines.append("")
    lines.append(markdown_table(top_validation))
    lines.append("")
    lines.append("### 5.2 状态分布")
    lines.append("")
    lines.append(markdown_table(status_counts))
    lines.append("")
    if not accepted_frame.empty:
        lines.append("### 5.3 通过审计的因子")
        lines.append("")
        lines.append(markdown_table(accepted_frame))
        lines.append("")
    lines.append("## 6. 最终组合")
    lines.append("")
    if portfolio:
        lines.append(f"- 选中因子：`{', '.join(context.selected_ids)}`")
        lines.append(f"- 是否回退到候选因子：`{portfolio.get('fallback')}`")
        if len(context.selected_ids) == 1:
            lines.append(
                "- 说明：其他通过审计的因子没有在验证集上提升组合 Sharpe，因此没有为了“多因子”而强行加入；这是组合构建的克制，而不是流程缺失。"
            )
        lines.append(f"- 验证集净 Sharpe：{_fmt(val.get('net_sharpe'))}；测试集净 Sharpe：{_fmt(test.get('net_sharpe'))}")
        lines.append(f"- 测试集年化收益：{_fmt(test.get('net_ann_return'))}；最大回撤：{_fmt(test.get('max_drawdown'))}")
        lines.append(f"- 测试集平均 IC：{_fmt(test.get('mean_ic'))}；ICIR：{_fmt(test.get('icir'))}")
        lines.append(
            f"- 随机组合基线：{random_baseline.get('n_portfolios', 0)} 个组合测试集 Sharpe 均值 {_fmt(random_baseline.get('mean_test_net_sharpe'))}，"
            f"中位数 {_fmt(random_baseline.get('median_test_net_sharpe'))}，所选组合百分位 {_fmt(100 * random_baseline.get('selected_percentile', float('nan')), 1)}%。"
        )
        ablation = portfolio.get("ablation_no_critic", {})
        if ablation:
            ablation_test = ablation.get("test", {})
            lines.append(
                f"- 消融对照（去掉 Critic 审计，直接选验证集 ICIR 前 5）：测试集净 Sharpe "
                f"{_fmt(ablation_test.get('net_sharpe'))}，年化收益 {_fmt(ablation_test.get('net_ann_return'))}。"
            )
    else:
        lines.append("_No portfolio was constructed._")
    lines.append("")
    if best is not None:
        lines.append("### 6.1 验证集 ICIR 最高单因子对照")
        lines.append("")
        lines.append(
            f"按验证集 ICIR 选出的最高单因子为 `{best.spec.spec_id}`（{best.spec.name}）。验证集 ICIR {_fmt(best.validation.metrics.get('icir'))}，"
            f"测试集净 Sharpe {_fmt(best.test.metrics.get('net_sharpe'))}，测试集年化收益 {_fmt(best.test.metrics.get('net_ann_return'))}。"
        )
        lines.append("")
    lines.append("## 7. 图表")
    lines.append("")
    for name in figures:
        lines.append(f"![{name}](figures/{name})")
        lines.append("")
    lines.append("## 8. 结论与限制")
    lines.append("")
    lines.append("- 框架层面：完成了从假设生成、因子实现、成本后回测、对抗式审计、组合构建到实验血缘的完整闭环。")
    lines.append("- 研究层面：结果应被视为对搜索效率和验证流程的演示，而不是可实盘交易的 alpha 结论。")
    lines.append("- 数据层面：当前股票池不是完整的 PIT 全市场股票池，存在幸存者偏差；未覆盖财务因子、行业中性化和完整涨跌停/停牌建模。")
    lines.append("- 统计层面：尽管使用了 FDR、DSR、PBO 和 holdout，仍不能消除所有数据挖掘偏差；实盘前需要更长样本、更多市场和真实执行成本验证。")
    if not accepted_frame.empty:
        lead = accepted_frame.sort_values("val_sharpe", ascending=False).iloc[0]
        lines.append(
            f"- 证据强度：最终因子的 FDR q 值为 {_fmt(lead['q'])}，DSR 为 {_fmt(lead['dsr'])}，"
            f"验证集 PBO 为 {_fmt(context.critic_summary.get('pbo'))}；因此应把结果视为中等强度的研究证据，而不是确定性 alpha。"
        )
    lines.append("- 工程层面：默认模板 Agent 可复现、低成本；若接入 LLM，应保留相同的确定性验证层和预算控制。")
    lines.append("")
    lines.append("## 9. 简历可用表述")
    lines.append("")
    lines.append(
        f"- 独立实现 `{context.config.project_name}` 多智能体因子研究框架，覆盖假设生成、因子实现、成本后回测、"
        f"FDR/DSR/PBO 审计、组合构建与 SQLite 实验血缘；在 {meta.get('n_selected_symbols')} 只 A 股、"
        f"{meta.get('start_date')} 至 {meta.get('end_date')} 日频数据上评估 {len(results)} 个候选因子。"
    )
    lines.append(
        f"- 构建未来数据扰动和标签打乱审计，通过全部审计的因子 {len(accepted_frame)} 个；"
        f"最终组合测试集净 Sharpe {_fmt(test.get('net_sharpe'))}，年化收益 {_fmt(test.get('net_ann_return'))}，"
        f"最大回撤 {_fmt(test.get('max_drawdown'))}，换手率 {_fmt(test.get('turnover_annual'))}。"
    )
    lines.append(
        f"- 在固定搜索空间内与 {random_baseline.get('n_portfolios', 0)} 个随机组合对照，所选组合测试集 Sharpe 位于第 "
        f"{_fmt(100 * random_baseline.get('selected_percentile', float('nan')), 1)} 百分位；"
        f"验证集 PBO 为 {_fmt(context.critic_summary.get('pbo'))}。"
    )
    lines.append("")
    lines.append("## 10. 复现")
    lines.append("")
    lines.append("```bash")
    lines.append("cd factorfoundry")
    lines.append("python3 -m alpha_foundry.pipeline --config config.json --stage all")
    lines.append("```")
    lines.append("")
    lines.append("实验明细见 `experiments/factor_results.csv`，机器可读摘要见 `output/summary.json`。")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_resume_bullets(context, summary, output_dir)
    return report_path, summary_path


def write_resume_bullets(
    context: AgentContext, summary: dict[str, Any], output_dir: Path
) -> Path:
    portfolio = summary.get("portfolio", {}).get("test", {}) or {}
    random_baseline = summary.get("portfolio", {}).get("random_baseline", {}) or {}
    data = summary.get("data", {})
    critic = summary.get("critic", {})
    n_factors = summary.get("n_factor_candidates", 0)
    n_accepted = summary.get("n_accepted", 0)
    selected = ", ".join(summary.get("selected_factors", []))
    lines = [
        "# Resume bullets",
        "",
        "以下表述只使用本次运行的实际结果；若测试集收益不理想，优先保留工程与验证流程类表述，不要虚构收益。",
        "",
        "## 中文版本",
        "",
        f"- 独立实现 `{context.config.project_name}` 多智能体因子研究框架，覆盖经济假设生成、因子实现、成本后横截面回测、"
        f"FDR/DSR/PBO 审计、组合构建与 SQLite 实验血缘；在 {data.get('n_selected_symbols')} 只 A 股、"
        f"{data.get('start_date')} 至 {data.get('end_date')} 日频数据上评估 {n_factors} 个候选因子。",
        f"- 设计未来数据扰动和截面标签打乱审计，形成可复现的因子验收流程；{n_accepted} 个因子通过全部筛选，"
        f"验证集 PBO 为 {_fmt(critic.get('pbo'))}。",
        f"- 在验证集上选择低相关因子并锁定测试集，组合测试集净 Sharpe {_fmt(portfolio.get('net_sharpe'))}、"
        f"年化收益 {_fmt(portfolio.get('net_ann_return'))}、最大回撤 {_fmt(portfolio.get('max_drawdown'))}、"
        f"年化换手 {_fmt(portfolio.get('turnover_annual'))}；与同一搜索空间的随机组合基线对照，"
        f"所选组合位于第 {_fmt(100 * random_baseline.get('selected_percentile', float('nan')), 1)} 百分位。",
        "",
        "## English version",
        "",
        f"- Built `{context.config.project_name}`, an auditable multi-agent alpha research framework covering hypothesis generation, "
        f"factor implementation, cost-aware cross-sectional backtesting, FDR/DSR/PBO checks, portfolio construction, and SQLite experiment lineage; "
        f"evaluated {n_factors} candidates on {data.get('n_selected_symbols')} A-share names from {data.get('start_date')} to {data.get('end_date')}.",
        f"- Designed future-data perturbation and cross-sectional label-shuffle audits; {n_accepted} factors passed the full screen, "
        f"with a validation PBO of {_fmt(critic.get('pbo'))}.",
        f"- Selected low-correlation factors on validation only and locked the holdout; the portfolio delivered test net Sharpe "
        f"{_fmt(portfolio.get('net_sharpe'))}, annualized return {_fmt(portfolio.get('net_ann_return'))}, max drawdown "
        f"{_fmt(portfolio.get('max_drawdown'))}, and annualized turnover {_fmt(portfolio.get('turnover_annual'))}. "
        f"It ranked in the {_fmt(100 * random_baseline.get('selected_percentile', float('nan')), 1)}th percentile against random portfolios from the same search space.",
        "",
        "## Interview narrative",
        "",
        "> 因子研究的瓶颈不是回测速度，而是高质量假设供给、实验可复现性和过拟合控制。我把研究流程拆成假设、实现、回测、Critic 审计和组合五个角色；Agent 只负责前两步，收益和统计检验全部由确定性代码完成。系统记录每次实验的代码、数据版本和拒绝原因，并用未来数据扰动、标签打乱、FDR、DSR 和 PBO 守住验证层。",
        "",
        "## 不要写",
        "",
        "- 不要写“AI 自动挖出高收益因子”。",
        "- 不要把回测收益写成实盘收益。",
        "- 不要省略样本外、交易成本和多重检验。",
        f"- 本次选中因子：`{selected}`。",
        "",
    ]
    path = output_dir / "resume_bullets.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
