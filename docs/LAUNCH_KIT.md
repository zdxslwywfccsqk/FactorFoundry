# GitHub and community launch kit

This file contains copy-ready material for presenting FactorFoundry without
overstating the research result. Update numbers only from a committed reference
run with an exact run id, config hash, and data fingerprint.

## Repository metadata

**About**

> Open-source multi-agent research lab for auditable alpha discovery. Agents
> propose; deterministic tools compute, audit, and lock the holdout.

**Suggested topics**

`quantitative-finance` `alpha-research` `factor-model` `multi-agent` `llm-agents`
`a-share` `backtesting` `research-tools` `reproducibility` `mini-agent`

**Social preview**

Use [`assets/social-preview.png`](assets/social-preview.png), 1280×640. Regenerate
with:

```bash
python3 scripts/render_brand_assets.py
```

**Pinned repository links**

- release: `v0.3.0`
- benchmark: `docs/REFERENCE_RUN.md`
- architecture: `docs/ARCHITECTURE.md`
- contribution guide: `CONTRIBUTING.md`

## One-sentence pitches

English:

> FactorFoundry is an open-source multi-agent research lab that turns economic
> hypotheses into reproducible, leakage-audited alpha experiments.

中文：

> FactorFoundry 是一个开源多智能体 Alpha 研究实验室，把经济假设转化为可复现、
> 可审计、锁定 Holdout 的因子实验。

## English launch post

### Short version

Most factor-mining demos optimize for a prettier equity curve. FactorFoundry
optimizes for whether the result survives contact with reality.

It is an open-source multi-agent research lab for A-share alpha discovery:

- LLM agents propose and critique;
- deterministic tools compute every metric;
- the test split is locked by a runtime capability guard;
- rejected factors and failed trials remain in the evidence chain;
- FDR, DSR, PBO, and leakage audits are part of acceptance.

The v0.3 reference run reports 0.7567 net Sharpe and 20.46% annualized net return on
the holdout. It also receives only 46/100 on our Alpha Evidence Scorecard and the
certification **Benchmark only**. None of the 73 candidates passed every critic gate,
so the portfolio is an explicit fallback. PBO is 0.56, DSR is 0.25, FDR q is 0.33,
and the holdout Sharpe interval is [-0.09, 0.32]. The point is not to market a number.
The point is to make the evidence inspectable.

GitHub: https://github.com/zdxslwywfccsqk/FactorFoundry

### Thread outline

1. The problem: agent demos generate more backtests, not necessarily better
   evidence.
2. The design: bounded roles, typed artifacts, deterministic truth layer.
3. The hard part: holdout lock, negative evidence, multiple-testing correction.
4. The reference run: headline result plus the caveats that weaken it.
5. The ask: review the protocol, submit a falsification test, or contribute a
   point-in-time benchmark.

## 中文发布文案

### 短版

很多因子挖掘项目追求更漂亮的净值曲线。FactorFoundry 追求的是：一个结果在接触
真实世界后仍然存活的概率。

这是一个面向 A 股 Alpha 挖掘的开源多智能体研究框架：

- Agent 负责提出经济假设、实现选择和对抗式审查；
- 所有收益、显著性和组合指标由确定性工具计算；
- 测试集由运行时能力锁保护，必须显式释放；
- 被拒绝的因子和失败试验也会进入证据链；
- FDR、DSR、PBO、未来数据扰动和标签打乱是验收的一部分。

v0.3 参考运行在 Holdout 上报告 0.7567 净 Sharpe、20.46% 年化净收益，但在
Alpha Evidence Scorecard 上只有 46/100，认证等级为 **Benchmark only**。73 个
候选中没有任何因子通过全部 Critic 门槛，页面中的组合是显式 fallback；PBO 为
0.56、DSR 为 0.25、FDR q 为 0.33，Holdout Sharpe 的 95% 区间为 [-0.09, 0.32]。
我们不把单次历史结果包装成稳定 Alpha，而是把完整证据链公开出来，接受审阅和证伪。

GitHub：https://github.com/zdxslwywfccsqk/FactorFoundry

### 社区讨论标题

- 为什么 Agent 不应成为因子研究里的“数值真相来源”
- 一个开源的多智能体 Alpha 研究协议：如何锁定 Holdout 并保留失败证据
- 从 73 个候选到 1 个组合：一次可复现的 A 股因子搜索
- 我们公开了 PBO 0.56、DSR 0.25 和 46/100 的评分，因为这比只展示 Sharpe 更重要

## Claims to avoid

Do not describe the project as:

- a guaranteed profitable strategy;
- a live trading or production execution system;
- proof that LLM agents can autonomously discover durable alpha;
- a result that is unaffected by survivorship, point-in-time, or short-sale bias;
- investment advice.

Prefer language such as “reproducible benchmark”, “conditional evidence”,
“research infrastructure”, and “falsifiable protocol”.
