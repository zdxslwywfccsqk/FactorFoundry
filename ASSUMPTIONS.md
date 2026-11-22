# FactorFoundry 假设汇总

本文档把项目中的**经济假设、交易假设、统计假设和已知偏差**汇总为一张表。
表格是解释层，实际执行以 `config.json` 和 `src/alpha_foundry/*.py` 为准；若文档与代码不一致，应优先修正文档或代码，而不是假设两者等价。

## 假设总表

| # | 层面 | 假设 | 具体设定 | 代码/配置依据 | 影响或风险 |
|---:|---|---|---|---|---|
| 1 | 市场 | A 股横截面存在可预测的 5 日收益差异 | 因子值越高或越低，未来 5 日收益按预定方向单调变化；用 Rank IC 和多空组合检验 | `src/alpha_foundry/backtest.py:21`、`src/alpha_foundry/backtest.py:126` | 假设因子与收益的关系在训练、验证、测试期足够稳定；未证明市场无效 |
| 2 | 交易 | 信号和成交都发生在 `t` 日收盘附近 | 使用 `t` 日及之前数据计算信号，假设可在 `t` 日收盘成交，没有执行延迟 | `README.md:53` | 执行乐观偏差；无法真实获得收盘价和当日成交量后再按同一收盘价成交 |
| 3 | 交易 | 每 5 个交易日调仓并持有 5 日 | 未来收益为 `close(t+5)/close(t)-1`；评价日期按 5 日切片；年化使用 `252/5` 期 | `config.json:32`、`src/alpha_foundry/backtest.py:61` | 忽略周内路径、盘中风控和临时停牌；收益序列近似非重叠 |
| 4 | 组合 | 使用 rank-weighted 多空组合，总敞口为 2 | 截面排名中心化后归一化，多空总绝对权重约 2；quantile 模式可选但当前未启用 | `src/alpha_foundry/backtest.py:88` | 近似 dollar-neutral，但 rank 求和不为零，存在约 `1/n` 量级的净多头 |
| 5 | 组合 | 多因子信号可以线性等权合成 | 每个因子先做截面 z-score，缺失填 0，再等权相加，最后对合成信号重新排序 | `src/alpha_foundry/portfolio.py:11`、`src/alpha_foundry/agents.py:463` | 注释写的是 equal-risk，实际是等权信号；没有风险平价或因子协方差矩阵 |
| 6 | 交易 | 多空两条腿都可执行，且可维持 2 倍总敞口 | 假设可以融券做空、没有借券费、融资成本、保证金或杠杆上限 | `output/detailed_report.md:1048` | A 股实际做空和杠杆受限；空头腿可能不可复制 |
| 7 | 交易 | 交易成本为固定 10 bps/边 | 换手按半换手计算，成本为 `0.5 × Σabs(Δw) × 10bps`；没有印花税、佣金、价差、冲击成本和容量约束 | `config.json:35`、`src/alpha_foundry/backtest.py:110` | 对总敞口 2 的组合，初始建仓只扣约 10 bps；若 10 bps 是每腿单边费率，实际成本应接近 20 bps |
| 8 | 交易 | 涨跌停和停牌可以用收益阈值近似 | 主板 `abs(ret) ≤ 9.5%`、创业板/科创板 `abs(ret) ≤ 19.5%`，且 `volume > 0`、`close` 非缺失 | `src/alpha_foundry/data.py:127` | 未处理 ST 5% 涨跌停、IPO 首日、T+1、方向性涨跌停、精确停牌状态和盘口流动性 |
| 9 | 数据 | 当前活跃、流动性较高的 A 股可以代表历史可投资域 | 从当前成交额靠前的约 200 只股票中筛选 155 只；历史长度、平均成交额和价格使用 2020-12-31 之前数据 | `src/alpha_foundry/data.py:35`、`config.json:12` | 存在幸存者偏差和当前流动性选择偏差；不是 PIT 全市场成分股 |
| 10 | 数据 | 缺失率筛选不引入显著未来信息 | `missing_ratio` 实际按全样本计算，而不是只按 `selection_end_date` 之前计算 | `src/alpha_foundry/data.py:44` | 已知的小样本 PIT 泄漏风险；本轮该条件未改变最终名单，但生产版本应修正 |
| 11 | 数据 | 前复权价格和沪深 300 足以构造量价因子 | 使用 Baostock 前复权日线，沪深 300 收益广播为市场收益；人民币成交额作为 `dollar_volume` 代理 | `src/alpha_foundry/data.py:88`、`config.json:16` | 没有财务、行业、分析师、宏观或另类数据；复权处理可能与真实持仓收益有差异 |
| 12 | 风险 | 可以用零无风险利率和简单波动率衡量风险调整收益 | Sharpe 使用 `mean(net_return)/std(net_return)`，不扣无风险利率，按 `sqrt(252/h)` 年化 | `src/alpha_foundry/backtest.py:171` | 没有风险模型、Beta 中性、行业中性或市值中性；除残差动量外，因子可能承载风格暴露 |
| 13 | 研究流程 | 训练、验证、测试集严格隔离 | 训练 2019-07-01 至 2021-12-31；验证 2022-01-01 至 2024-12-31；测试 2025-01-01 至 2026-09-30 | `config.json:23` | 测试集未进入程序选择逻辑，但研究者看到结果后仍可能影响下一轮实验设计 |
| 14 | 统计 | 多重检验可以用 BH-FDR、DSR 和 PBO 控制 | 73 个去重候选统一做 BH-FDR，FDR 阈值 0.20；用验证集收益计算 DSR 和 PBO | `src/alpha_foundry/agents.py:280`、`src/alpha_foundry/stats.py:43`、`src/alpha_foundry/stats.py:77` | 这些是近似控制；搜索空间、窗口和方向先验仍带来研究者自由度 |
| 15 | 统计 | 因子没有使用未来数据 | 在 cutoff 之后注入噪声，比较 cutoff 时点的因子值是否不变；并做截面标签打乱检验 | `src/alpha_foundry/agents.py:337` | 只能检测实现层面的未来数据使用，不能消除股票池和研究流程层面的选择偏差 |
| 16 | 统计 | IC 和收益观测近似独立同分布 | IC t 检验使用简单 t 统计量；Sharpe 使用算术均值和标准差；Bootstrap 使用 IID 重采样 | `src/alpha_foundry/backtest.py:163`、`src/alpha_foundry/stats.py:110` | 未做 Newey-West、block bootstrap 或序列相关修正；测试期短时置信区间很宽 |
| 17 | 组合 | 缺失因子值可以视为中性 | 截面 z-score 后把缺失值填为 0；单因子回测则在可用股票截面内计算 | `src/alpha_foundry/portfolio.py:11` | 缺失可能不是随机缺失；合成信号可能给可用股票更高权重 |
| 18 | 组合 | 低相关因子组合优于单因子 | 最多选 5 个因子，两两收益相关性上限 0.65，加入新因子需使验证集净 Sharpe 至少提升 0.01 | `config.json:48`、`src/alpha_foundry/agents.py:430` | 相关性用验证集估计，可能不稳定；组合选择本身也会过拟合验证集 |
| 19 | 搜索空间 | 30 个经济因子家族展开为 77 个参数化定义，去重后 73 个候选 | 每个家族按多个窗口生成候选，按公式和方向去重后统一进入回测和多重检验 | `src/alpha_foundry/factors.py:28` | `intraday`/`intraday_mom`、`drawdown`/`dist_high` 等存在重复或反向重复，有效独立假设少于 73 个 |
| 20 | 经济机制 | 短期反转：过去涨幅过高会均值回归 | `rev_3/5/10/20`，方向 `-1`；公式为 `close/delay(close,N)-1` 取负 | `src/alpha_foundry/factors.py:29` | 对交易成本和微观结构噪声敏感；短窗口换手较高 |
| 21 | 经济机制 | 中期动量：相对强势具有持续性 | `mom_20/60/120`，方向 `+1`；使用过去 N 日收益 | `src/alpha_foundry/factors.py:38` | 可能受反转、拥挤和风格切换影响；A 股中期动量并不稳定 |
| 22 | 经济机制 | 低波动异象：高波动股票风险调整后收益更差 | `lowvol_10/20/60`，方向 `-1`；使用日收益滚动标准差 | `src/alpha_foundry/factors.py:47` | 可能只是低 Beta/防御风格暴露；不同市场环境下失效 |
| 23 | 经济机制 | 成交量冲击：异常放量后短期反转 | `volshock_5/20/60`，方向 `-1`；成交量为 N 日均值的倍数 | `src/alpha_foundry/factors.py:56` | 事件驱动强，受公告、新闻和停牌复牌影响 |
| 24 | 经济机制 | 成交量趋势：温和放量确认价格趋势 | `voltrend_10/20/60`，方向 `+1`；短期均量相对长期均量 | `src/alpha_foundry/factors.py:65` | 量价关系可能因市场状态而反转；没有区分主动买卖盘 |
| 25 | 经济机制 | 收益-成交量相关：高同向性反映拥挤 | `corrretvol_10/20/60`，方向 `-1`；日收益与成交量的滚动相关 | `src/alpha_foundry/factors.py:74` | 相关性估计噪声大；无法区分知情交易和情绪交易 |
| 26 | 经济机制 | 价格-成交量相关：价量同步上升存在反转风险 | `corrclosevol_20/60`，方向 `-1`；收盘价与成交量的滚动相关 | `src/alpha_foundry/factors.py:83` | 该因子验证集表现强但测试集失效，说明机制可能不稳定 |
| 27 | 经济机制 | Amihud 非流动性：单位成交额冲击越大，流动性溢价越高 | `amihud_20/60`，方向 `+1`；`abs(ret)/dollar_volume` 的滚动均值 | `src/alpha_foundry/factors.py:92` | 与低流动性因子高度相关；可能承载规模和价值暴露 |
| 28 | 经济机制 | 日内振幅：高振幅反映不确定性和投机 | `rangevol_5/20/60`，方向 `-1`；`(high-low)/close` 的滚动均值 | `src/alpha_foundry/factors.py:101` | 与波动率、彩票偏好和微观结构噪声相关 |
| 29 | 经济机制 | 收盘位置：收盘价靠近日内高位代表买盘承接 | `clv_5/20`，方向 `+1`；`(2*close-high-low)/(high-low)` 的滚动均值 | `src/alpha_foundry/factors.py:110` | 依赖收盘集合竞价和日内路径；受买卖价差影响 |
| 30 | 经济机制 | 极端正收益：彩票偏好导致后续收益偏弱 | `maxret_20/60`，方向 `-1`；过去 N 日最大单日收益 | `src/alpha_foundry/factors.py:119` | 容易被事件、涨停和少量极端样本驱动 |
| 31 | 经济机制 | 回撤/趋势：越接近区间高点，趋势资金越强 | `drawdown_20/60/120`，方向 `+1`；`close/rolling_max(close,N)-1` | `src/alpha_foundry/factors.py:128` | 与 `disthigh_60/120` 完全重复；趋势和反转机制可能互相抵消 |
| 32 | 经济机制 | 均线偏离：价格过度偏离均线后均值回归 | `madist_5/20/60`，方向 `-1`；`close/rolling_mean(close,N)-1` 取负 | `src/alpha_foundry/factors.py:137` | 对趋势市场和波动状态敏感；窗口选择影响大 |
| 33 | 经济机制 | 隔夜跳空：情绪和流动性冲击后部分反转 | `gaprev_1/5/20`，方向 `-1`；`open/delay(close,1)-1` 的滚动均值取负 | `src/alpha_foundry/factors.py:146` | 与 `overnightmom_5/20` 是同一信号的反向版本；受隔夜信息和停牌影响 |
| 34 | 经济机制 | 日内动量：日内收益和订单流方向短期持续 | `intraday_5/20`，方向 `+1`；`close/open-1` 的滚动均值 | `src/alpha_foundry/factors.py:155` | 与 `intradaymom_5/20` 公式完全重复；容易受开盘价噪声影响 |
| 35 | 经济机制 | 成交量加权收益更接近真实资金方向 | `vwapret_5/20/60`，方向 `+1`；收益按成交量加权的滚动均值 | `src/alpha_foundry/factors.py:164` | 使用日频收益和日成交量近似 VWAP，不是真正的逐笔成交 |
| 36 | 经济机制 | 成交量波动：成交量不稳定反映拥挤和不确定性 | `volofvol_20/60`，方向 `-1`；成交量的滚动标准差/均值 | `src/alpha_foundry/factors.py:173` | 与流动性、波动率和事件交易相关；没有区分成交量的方向 |
| 37 | 经济机制 | 收益偏度：正偏度股票具有彩票特征 | `skew_20/60`，方向 `-1`；日收益的滚动偏度 | `src/alpha_foundry/factors.py:182` | 偏度估计对极端值敏感；样本短时不稳定 |
| 38 | 经济机制 | 低 Beta：高 Beta 股票承担更高风险，低 Beta 有防御价值 | `lowbeta_60/120`，方向 `-1`；对沪深 300 的滚动 Beta | `src/alpha_foundry/factors.py:191` | Beta 估计依赖市场基准和窗口；可能与其他低风险因子高度相关 |
| 39 | 经济机制 | 特质波动：高特质波动股票存在套利限制和估值折价 | `idiovol_20/60`，方向 `-1`；剥离 Beta 后残差收益的滚动标准差 | `src/alpha_foundry/factors.py:200` | 残差只剥离市场 Beta，没有剥离行业和风格；仍可能承载风险暴露 |
| 40 | 经济机制 | 区间位置：价格位于区间上部时趋势延续 | `rangepos_20/60/120`，方向 `+1`；`(close-min_low)/(max_high-min_low)` | `src/alpha_foundry/factors.py:209` | 与动量、回撤和距高点因子相关；震荡市容易反转 |
| 41 | 经济机制 | 价格加速度：短周期相对长周期过度加速后反转 | `priceaccel_20`，方向 `-1`；`ret_5-ret_20` 取负 | `src/alpha_foundry/factors.py:218` | 窗口固定且机制较窄；对趋势和波动状态敏感 |
| 42 | 经济机制 | 量价动量：有成交量确认的动量更容易持续 | `momvol_20/60`，方向 `+1`；过去 N 日收益乘以相对成交量 | `src/alpha_foundry/factors.py:227` | 与 `revvol_20` 是同一公式的反向版本；量价交互可能不稳定 |
| 43 | 经济机制 | 量价反转：放量上涨后的短期反转更强 | `revvol_5/20`，方向 `-1`；过去 N 日收益乘以相对成交量后取负 | `src/alpha_foundry/factors.py:236` | 与 `momvol_20` 反向重复；对异常成交量敏感 |
| 44 | 经济机制 | 隔夜动量：隔夜收益反映信息消化方向并短期持续 | `overnightmom_5/20`，方向 `+1`；`open/delay(close,1)-1` 的滚动均值 | `src/alpha_foundry/factors.py:245` | 与 `gaprev_5/20` 是同一信号的反向版本；无法区分隔夜信息类型 |
| 45 | 经济机制 | 日内动量均值：日内订单流方向短期延续 | `intradaymom_5/20`，方向 `+1`；`close/open-1` 的滚动均值 | `src/alpha_foundry/factors.py:254` | 与 `intraday_5/20` 公式完全重复，属于重复试验 |
| 46 | 经济机制 | 距区间高点：接近高点更容易获得趋势资金关注 | `disthigh_60/120`，方向 `+1`；`close/rolling_max(close,N)-1` | `src/alpha_foundry/factors.py:263` | 与 `drawdown_60/120` 完全重复；趋势拥挤时可能反转 |
| 47 | 经济机制 | 距区间低点：接近低点可能存在超跌反弹 | `distlow_60/120`，方向 `-1`；`close/rolling_min(close,N)-1` 取负 | `src/alpha_foundry/factors.py:272` | 可能买到基本面恶化或退市风险股票；需要财务和事件过滤 |
| 48 | 经济机制 | 换手冲击：换手异常放大后拥挤和反转风险上升 | `turnovershock_5/20/60`，方向 `-1`；成交额相对 N 日均值的倍数 | `src/alpha_foundry/factors.py:281` | 用成交额代替真实换手率；没有股本和自由流通市值数据 |
| 49 | 经济机制 | 低流动性溢价：低流动性股票需要更高预期收益补偿成本 | `illiquidity_20/60/120`，方向 `+1`；`-log(mean(dollar_volume,N))` | `src/alpha_foundry/factors.py:290` | 本轮最终选中 `illiquidity_20`；可能主要是规模/流动性风险溢价，且空头腿难执行 |
| 50 | 经济机制 | 残差动量：剥离 Beta 和流动性后的动量更接近特质 alpha | `residmom_20/60/120`，方向 `+1`；对 `beta_60` 和 `log_adv20` 做截面中性化后的 N 日收益 | `src/alpha_foundry/factors.py:299` | 只剥离两个控制变量；没有行业、规模、估值等完整风险模型 |

## 使用建议

- 讨论“因子是否有效”时，引用第 20–50 条的经济机制，不要只引用回测 Sharpe。
- 讨论“结果能否实盘”时，优先检查第 2–8、12、18 条的交易和组合假设。
- 讨论“统计显著性”时，引用第 13–19 条，并结合 `output/detailed_report.md` 第 10 节的偏差清单。
- 如果修改了 `config.json` 或核心代码，应同步更新本表，避免文档与实现漂移。
