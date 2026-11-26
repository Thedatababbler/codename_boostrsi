# Boosted Topology：实现文档

Nov 19, 2026 · @dogecheemse

本文档说明如何在 SelfImprove-Bench 上实现“像 boosting 一样进化 MAS 子图”：先用 mini-swe-agent 搭好基础代码，再实现 GBDT 版本和 AdaBoost 版本，并预留随机森林式宏改动的接口。

## 1. 目标与范围

第一版要交付一个能在 SelfImprove-Bench 的 Farm 游戏上跑通的 boosting 式子图进化框架：两种组合模式共用一套代码，变动选择做成可替换的接口。

### 1.1 三个版本的关系

| 版本 | 残差体现在哪里 | 子图怎么组合 | 新子图的目标 |
| --- | --- | --- | --- |
| GBDT | 输入的解：新子图拿到上一阶段改过的解 | 串联，逐阶段修正 | 在已改过的解上继续缩小差距 |
| AdaBoost | 题目权重：做得差的题权重高 | 各自独立运行，取分数最高的 | 补上当前集成做不好的题 |
| 随机森林插件 | 不改变残差定义 | 不改变组合方式 | 让每次搜索在随机子空间里进行，并允许宏改动 |

前两个是**组合模式**，二选一；随机森林是**变动选择方式**，可以叠加在任意一种模式上，也可以关闭。

### 1.2 第一版不做的事

- 第一版（M0 到 M3）只做**拓扑层**的自我改进：子图和集成由 boosting 进化，搜索策略 f 是手写的 f0，保持不变。**元层**的自我改进放在里程碑 M4，方式是 playbook（第 9.5 节）：把每个被接受的子图沉淀成可复用的编排原则，用它来驱动后续搜索；
- 不进化角色提示词，角色库固定；
- 不接数学题族和 RSI-Forge，先只用 SelfImprove-Bench；
- M4 之前不做路由器，AdaBoost 模式下集成里的每个子图都运行；M4 起由 playbook 的分类树负责路由。

### 1.3 术语

- **题（item）**：SelfImprove-Bench 生成的一个游戏实例；
- **解**：智能体对一道题给出的方案，比如 Farm 的按天种植计划；
- **子图**：2 到 3 个角色节点组成的小型 MAS，作用是把一个解改得更好；
- **集成**：已被接受的子图的有序列表；
- **缺口闭合度 g**：一道题上，从起始分数到参考分数之间的差距被补上了多少，取值 0 到 1。

## 2. 系统总览

整个系统是一个外层循环：每一轮算残差、长出若干候选子图、挑一个、在留出题上验证，通过就冻结进集成。GBDT 和 AdaBoost 两个版本只在第 1、2 步和“子图拿到什么输入”上不同，其余步骤共用。

&#91;embedded content: 一轮 boosting · 6 个步骤\]

### 2.1 模块分层

| 层 | 模块 | 职责 | 谁可以改 |
| --- | --- | --- | --- |
| 环境层 | `bench/` | 生成题目、执行游戏、返回分数 | 固定 |
| 执行层 | `graph/`、`roles/` | 子图的数据结构、角色提示词、按拓扑序运行节点 | 固定 |
| 评估层 | `evalx/` | 运行子图或集成、缓存逐题结果、计算 g 与成本 | 固定 |
| Boosting 层 | `boost/` | 两种组合模式、残差、抽题、接受准则、子图库 | 固定（超参数可配） |
| 搜索层 | `search/` | 提议器、长子图的贪心循环 | 提议器可替换 |
| 策略层 | `policy/` | 策略 f：给候选变动排序。f0 是手写规则 | M4 之前固定 |
| 元层 | `playbook/` | 编排原则库、分类树、权重与生命周期、推理路由 | **进化的对象**（M4） |

M4 之前，变化的只有子图和集成。M4 起，playbook 成为跨运行持续变化的部分：它通过提议器和策略两个接口接入搜索，其余各层保持固定，这样不同版本的 playbook 才能在相同条件下比较。

## 3. 用 mini-swe-agent 搭基础代码

基础代码拆成 8 张任务单，每张都带可执行的验收测试，逐张交给 mini-swe-agent 完成。人只负责写任务单和审查合并，不手写实现。

### 3.1 工作方式

mini-swe-agent 是一个只用 bash 完成任务的极简编码智能体，既有 `mini` 命令行，也可以作为 Python 库调用，通过 litellm 接各家模型。每张任务单的流程是：

1. 在 `tasks/Tn.md` 里写清目标、要改的文件、接口签名、验收命令；
2. 新开一个 git 分支，在仓库根目录运行 mini-swe-agent，把任务单全文作为任务输入；
3. 智能体结束后，人工运行验收命令并审查 diff，通过才合并。

```bash
pip install mini-swe-agent
mini-extra config setup                 # 首次配置模型和密钥
git checkout -b t3-graph
mini -t "$(cat tasks/T3.md)"            # 具体参数以 mini --help 为准
pytest tests/test_graph.py -q           # 人工验收
```

需要批量跑时改用 Python 接口：构造模型和本地环境，然后 `agent.run(task)`。

### 3.2 任务单的固定格式

```markdown
# T3 子图数据结构与执行器
## 目标
实现 Subgraph、Edit、apply_edit、run_subgraph。
## 只允许修改
src/boosttopo/graph/、tests/test_graph.py
## 接口（不得更改签名）
见 docs/interfaces.md 第 2 节
## 约束
- 不调用真实 LLM，测试一律用 FakeLLM
- 不新增第三方依赖
## 验收
pytest tests/test_graph.py -q 全部通过
```

三条写任务单的规则：接口签名由人在 `docs/interfaces.md` 里先定好，智能体只填实现；每张单只允许改指定目录；验收必须是一条能在本地跑的命令。

### 3.3 仓库结构

```text
boosttopo/
  docs/interfaces.md        # 人写：全部接口签名，第 5 节的内容
  tasks/T1.md … T8.md       # 人写：任务单
  configs/farm_gbdt.yaml    # 运行配置
  src/boosttopo/
    llm.py                  # LLM 客户端 + FakeLLM + 调用缓存
    bench/                  # SelfImprove-Bench 适配层
    roles/                  # 固定角色库与提示词
    graph/                  # Subgraph、Edit、执行器
    evalx/                  # 评估器、逐题结果缓存、g 的计算
    boost/                  # gbdt.py、adaboost.py、residual.py、library.py
    search/                 # grow.py、proposers.py
    policy/                 # f0.py（手写初始策略）
    rsi/                    # M4：回放报告与外层循环
  tests/
  runs/                     # 每次运行的配置、日志、结果
```

### 3.4 任务单列表

| 任务 | 内容 | 验收 |
| --- | --- | --- |
| T1 | 项目骨架、配置加载、`llm.py`（含 FakeLLM 和磁盘缓存） | 相同请求第二次不触发调用 |
| T2 | `bench/` 适配层：生成 Farm 题、校验并打分一个计划 | 手写的合法计划得分为正，非法计划得 0 |
| T3 | `graph/`：子图、原子编辑、宏编辑、执行器 | 6 种编辑的合法性测试；执行顺序符合拓扑序 |
| T4 | `evalx/`：运行子图、逐题缓存、计算 g 和成本 | 重复评估命中缓存；g 被截断在 0 到 1 |
| T5 | `search/grow.py` 与默认提议器、`policy/f0.py` | 在 FakeLLM 的玩具题上，子图分数单调不降 |
| T6 | `boost/gbdt.py`：串联模式全流程 | 玩具题上 3 轮后留出分数高于起点 |
| T7 | `boost/adaboost.py` 与 `library.py` | 权重随 g 升高而降低；离线组合选择复现手算结果 |
| T8 | 随机森林提议器插件、K 路并行与袋外挑选 | 配置开关可切换；关闭时结果与 T6、T7 完全一致 |

T1 到 T4 是基础代码，T5 之后才是算法。T2 完成后先人工跑通一次“单写者解 Farm 题”，确认分数量级正常再往下走。

### 3.5 在 Slurm 服务器上用 mini-swe-agent 写代码

服务器没有容器功能，mini-swe-agent 用本地执行模式：它会以你的账号直接在机器上运行 bash 命令，没有隔离。在共享集群上要守住下面几条。

| 事项 | 做法 |
| --- | --- |
| 在哪里跑 | 用 `salloc` 或 `srun --pty` 申请一个交互式作业，在计算节点上跑；不要在登录节点上长时间运行 |
| 网络 | 智能体要调 LLM 接口。先确认计算节点能否出网；不能的话，走集群代理，或在能出网的节点上运行 |
| 工作目录 | 只在项目仓库目录里启动；任务单里写明“只允许修改”的路径（第 3.2 节） |
| 确认模式 | 先用逐条确认命令的模式跑 T1、T2，熟悉它的行为后再考虑放开自动执行 |
| 环境 | 用独立的 conda 或 venv 环境；启动前只导出 LLM 的密钥，不要带着其他凭据（云账号、SSH 代理转发） |
| 可回滚 | 每张任务单一个 git 分支，启动前先提交；出问题直接丢弃分支 |
| 资源 | 申请作业时设好时间和内存上限，智能体跑测试失控时由 Slurm 兜底终止 |
| 可选的隔离 | 如果集群装了 Apptainer 或 Singularity，可以用它做执行环境；没有就用上面的办法 |

```bash
salloc --time=02:00:00 --cpus-per-task=4 --mem=16G     # 交互式作业
conda activate boosttopo
cd $HOME/projects/boosttopo && git checkout -b t3-graph
curl -sS -m 10 https://api.openai.com >/dev/null && echo "可出网" || echo "不可出网，需代理"
mini -t "$(cat tasks/T3.md)"
```

探测出网时把地址换成你实际使用的 LLM 服务地址。

## 4. SelfImprove-Bench 适配层

适配层把基准包成三个函数：生成题、给一个解打分、给出起始解。第一版只接 Farm，因为它是一次性输出完整计划、得到连续分数，最贴近“拿到一个解再改进”。

### 4.1 为什么先用 Farm

- **单次、非交互：** 智能体一次拿到完整题面（初始资金、地块数、天数、作物表），返回按天索引的 JSON 计划；
- **分数连续：** 分数是最终利润与基准自带参考解利润的比值。参考解来自贪心搜索，不是最优，所以分数可以超过 1；
- **难度可控：** 难度 d 取 1 到 5，作物种类为 2d+2，地块数和天数都是 4d+4；
- **打分不花钱：** 打分只是跑一遍模拟器，成本全部在 LLM 调用上。

Combat 是多步交互（战斗与商店交替），放到 Farm 跑通之后再接，用来做同类能力内的迁移测试。

### 4.2 题目划分

| 集合 | 来源 | 数量（建议） | 用途 | 谁能看到分数 |
| --- | --- | --- | --- | --- |
| 训练题 | Farm，难度 3，训练种子 | 40 | 算残差、搜索子图、袋内与袋外 | 搜索过程、策略 f |
| 留出题 | Farm，难度 3，留出种子 | 20 | 每轮决定是否接受新子图、是否早停 | 只有接受准则 |
| 终测题 G1 | Farm，难度 3，另一组种子 | 20 | 全部结束后报告一次 | 无 |
| 终测题 G2 | Farm，难度 4 和 5 | 各 10 | 更高难度的泛化 | 无 |
| 终测题 G3 | Combat，难度 3 | 20 | 同类能力内跨游戏的泛化 | 无 |

留出题每一轮都被用来做接受判断，用多了也会被过拟合。所以另设终测题，整个实验只在最后跑一次。三组种子必须互不重叠，生成后把题目清单写入 `runs/<id>/manifest.json`。

### 4.3 接口

```python
@dataclass(frozen=True)
class Item:
    item_id: str
    game: str            # "farm"
    difficulty: int
    seed: int
    prompt: str          # 给智能体看的完整题面
    payload: dict        # 原始实例，用于打分

class BenchAdapter(Protocol):
    def make_items(self, game: str, difficulty: int, seeds: list[int]) -> list[Item]: ...
    def score(self, item: Item, solution: str) -> ScoreResult: ...   # 分数 + 诊断

@dataclass
class ScoreResult:
    score: float         # 基准的原始分数；非法解为 0
    valid: bool
    diag: str            # 例如 "第 5 天资金不足"、"JSON 解析失败"
```

`diag` 要尽量保留基准返回的失败原因，后面的残差报告按它聚类。

### 4.4 起始解与缺口闭合度

每道题的起始解 y0 由“单写者”基线生成一次并冻结，记下它的分数 s0。之后所有比较都以它为起点。

```latex
g_i = \mathrm{clip}\!\left(\frac{s_i - s_i^0}{b_i - s_i^0},\ 0,\ 1\right), \qquad b_i = \max(1,\ s_i^0 + 0.05)
```

- 参考分数 b 取 1，即基准参考解的水平；起始解已经接近或超过 1 的题，b 取 s0 + 0.05，避免分母过小；
- 超过参考分数时 g 截断为 1，原始分数另存，报告时两者都给；
- 这个定义和 SelfImprove-Bench 的 HCI 指标是同一个公式，结果可以直接和它论文里的数字对照。

## 5. 核心数据结构与子图执行

子图是一个带角色的小型有向无环图，输入是（题，当前解），输出是一个新解。所有版本共用这一套结构和同一个评估缓存。

### 5.1 角色库（固定）

| 角色 | 类型 | 输入 | 输出 |
| --- | --- | --- | --- |
| `writer` | 写者 | 题面、当前解、上游消息 | 一个完整的新计划 |
| `repairer` | 写者 | 题面、当前解、打分器的报错 | 修好合法性问题的计划 |
| `critic` | 读者 | 题面、当前解 | 指出计划里浪费资金或地块的地方 |
| `analyst` | 读者 | 题面 | 每种作物的单位地块日收益等定量分析 |

写者产出可以打分的解，读者只产出消息。输出节点必须是写者。

### 5.2 子图与编辑

```python
@dataclass(frozen=True)
class Node:
    nid: str
    role: str                      # 角色库里的名字

@dataclass(frozen=True)
class Subgraph:
    nodes: tuple[Node, ...]
    edges: tuple[tuple[str, str], ...]   # (u, v)：v 读取 u 的输出；"IN" 表示子图输入
    out: str                              # 输出节点，必须是写者
    def key(self) -> str: ...             # 规范化后的哈希，用作缓存键和库的主键

@dataclass(frozen=True)
class Edit:
    kind: str      # add_node | add_reader | remove_node | add_edge | remove_edge | change_role
    args: dict

@dataclass(frozen=True)
class MacroEdit:
    steps: tuple[Edit, ...]        # 按顺序应用；任何一步不合法则整体不合法

def legal_edits(h: Subgraph, limits: Limits) -> list[Edit]: ...
def apply_edit(h: Subgraph, e: Edit | MacroEdit) -> Subgraph: ...
```

约束写在 `Limits` 里：节点数不超过 3，每个节点入度不超过 2，必须无环，输出节点从 `IN` 可达。起点子图是单个 `writer`。

### 5.3 执行与门控

```python
def run_subgraph(h: Subgraph, item: Item, y_in: str, llm: LLM, bench: BenchAdapter) -> StageOutput:
    msgs = {"IN": y_in}
    cost = 0
    for node in topo_order(h):
        upstream = [msgs[u] for (u, v) in h.edges if v == node.nid]
        diag = bench.score(item, y_in).diag if node.role == "repairer" else None
        out, tokens = llm.call(role_prompt(node.role, item.prompt, y_in, upstream, diag))
        msgs[node.nid] = out
        cost += tokens
    y_out = msgs[h.out]
    s_in, s_out = bench.score(item, y_in).score, bench.score(item, y_out).score
    y_final = y_out if s_out > s_in else y_in          # 门控：只在变好时采用
    return StageOutput(solution=y_final, score=max(s_in, s_out), raw_score=s_out, cost=cost)
```

门控对应 boosting 里的收缩：一个子图在任何一道题上都不会让分数变差。`raw_score` 是不经门控的分数，单独保留，用来分析子图本身的好坏。

### 5.4 评估结果与缓存

```python
@dataclass
class ItemResult:
    item_id: str
    score: float          # 门控后的分数，n_repeats 次的平均
    raw_score: float
    g: float
    cost: float           # 平均 token 数
    diag: str

class Evaluator:
    def run(self, h: Subgraph, items: list[Item], inputs: dict[str, str]) -> dict[str, ItemResult]: ...
```

- 缓存键是（子图哈希，题号，输入解的哈希，重复序号），落盘保存；
- 每个（子图，题）默认重复 2 次取平均，LLM 温度固定；
- `inputs` 是每道题的输入解：AdaBoost 模式下永远是 y0，GBDT 模式下是上一阶段的输出。

## 6. GBDT 版本（先实现）

GBDT 版本把子图串起来：第 m 个子图拿到前 m−1 个子图改过的解，只负责继续缩小剩下的差距。它是最简单的版本，不需要题目权重，先用它把整条流程跑通。

### 6.1 长一个子图（两个版本共用）

```python
def grow_subgraph(seed, items, inputs, fitness, proposer, policy, cfg) -> Subgraph:
    h, fit_h = seed, fitness(seed, items, inputs)
    for step in range(cfg.max_steps):
        ctx = GrowContext(h=h, items=items, inputs=inputs, step=step, obs=make_obs(...))
        cands = proposer.propose(h, ctx)                 # 第 8 节：可替换
        ranked = policy.rank(h, cands, ctx.obs)[: cfg.top_k]   # 策略 f：只挑前几个去评估
        scored = [(fitness(apply_edit(h, e), items, inputs), e) for e in ranked]
        best_fit, best_e = max(scored, key=lambda t: t[0])
        log_edit_outcomes(h, scored)                     # 每条都记下来，供 f 进化使用
        if best_fit - fit_h <= cfg.min_gain:             # 增益不够就停
            break
        h, fit_h = apply_edit(h, best_e), best_fit
    return h
```

这段循环对应决策树的贪心生长：提议器给出候选变动（相当于候选分裂），f 预估哪几个值得试，真正评估后保留增益最大的一个，增益不够就停。默认 `top_k = 3`、`max_steps = 3`、`min_gain = 0.005`。

### 6.2 适应度

```latex
\mathrm{fit}_{\mathrm{GBDT}}(h) = \frac{1}{|S|}\sum_{i \in S}\big[g_i(h(x_i, y_i^{m-1})) - g_i(y_i^{m-1})\big] - \lambda \cdot \mathrm{cost}(h)
```

S 是这次搜索用的题，y 的上标 m−1 表示上一阶段的输出。因为有门控，方括号里的量不会为负。

### 6.3 主循环

```python
def fit_gbdt(train, heldout, cfg, policy, proposer, ev) -> list[Subgraph]:
    ensemble = []
    cur_tr = {i.item_id: y0[i.item_id] for i in train}       # 训练题上的当前解
    cur_ho = {i.item_id: y0[i.item_id] for i in heldout}     # 留出题上的当前解
    stall = 0
    for m in range(cfg.rounds):
        active = [i for i in train if g(cur_tr[i.item_id]) < 1.0]   # 已达标的题不再参与
        cands = []
        for k in range(cfg.forest.k):                        # 默认 k = 1
            bag, oob = cfg.sampler.split(active, weights=None, seed=(m, k))
            h = grow_subgraph(single_writer(), bag, cur_tr, fit_gbdt_fn, proposer, policy, cfg)
            h_gain = fit_gbdt_fn(h, oob or bag, cur_tr)      # 没有袋外题时退回袋内
            cands.append((h_gain, h))
            library.add(h, mode="gbdt", stage=m)
        best = max(cands, key=lambda t: t[0])[1]

        out_ho = ev.run(best, heldout, cur_ho)               # 只有这里碰留出题
        gain = paired_gain(out_ho, cur_ho)                   # 逐题配对的 g 增量
        if lower_ci(gain) <= cfg.accept_margin:
            stall += 1
            if stall >= cfg.patience: break
            continue                                         # 不接受，换随机种子再来一轮
        stall = 0
        ensemble.append(best)
        cur_tr = solutions(ev.run(best, train, cur_tr))      # 残差更新：下一轮的输入
        cur_ho = solutions(out_ho)
    return ensemble
```

### 6.4 四个要点

- **残差更新就是替换输入。** 接受一个子图后，把所有训练题的当前解换成它的输出。下一轮的子图自然只能在这些“改过一轮”的解上找提升；
- **已达标的题退出。** g 已经为 1 的题不再进入搜索，推理时也跳过后续阶段，省成本；
- **接受看置信下界。** 留出题上逐题配对的 g 增量，用自助法求 90% 置信下界，高于 `accept_margin`（默认 0.01）才接受；
- **串联的记录不能跨阶段复用。** 第 m 阶段的评估结果依赖前面的阶段，入库时带上阶段号，只用于统计，不用于离线组合。

### 6.5 推理

对一道新题：先生成 y0，然后按顺序运行集成里的每个子图，每一步都过门控；分数达到参考分数就提前结束。

## 7. AdaBoost 版本

AdaBoost 版本让每个子图都从起始解 y0 独立出发，系统输出取分数最高的那个。每一轮把当前集成做得差的题权重调高，新子图专门去补这些题。它只在 GBDT 版本上改三处：残差权重、适应度、组合方式。

### 7.1 残差权重

```latex
g_i^{\mathrm{ens}} = \max_{h \in \mathrm{ensemble}} g_i(h(x_i, y_i^0)), \qquad r_i = U'(g_i^{\mathrm{ens}}) = \frac{1}{\varepsilon + g_i^{\mathrm{ens}}}
```

```python
def residual_weights(g_ens: dict[str, float], cfg) -> dict[str, float]:
    r = {i: U_PRIME[cfg.utility](g, cfg.eps) for i, g in g_ens.items()}   # 默认 log，eps = 0.1
    cap = cfg.weight_cap * mean(r.values())                              # 默认 5 倍均值
    r = {i: min(v, cap) for i, v in r.items()}
    z = sum(r.values())
    return {i: v / z for i, v in r.items()}
```

| `utility` | U(g) | 权重 r | 适用场景 |
| --- | --- | --- | --- |
| `log`（默认） | log(ε + g) | 1 / (ε + g) | 平滑地偏向难题 |
| `threshold` | min(g, τ) | g < τ 时为 1，否则为 0 | 只关心“达标的题数” |
| `exp` | −exp(−αg) | α·exp(−αg) | 经典 AdaBoost 的指数权重，对噪声最敏感 |

权重上限是为了防噪声：一道题分数低可能只是那两次运行运气不好，不能让它吸走全部权重。

### 7.2 适应度

```latex
\mathrm{fit}_{\mathrm{Ada}}(h) = \sum_{i \in S} r_i \cdot \big[\max(g_i^{\mathrm{ens}},\ g_i(h)) - g_i^{\mathrm{ens}}\big] - \lambda \cdot \mathrm{cost}(h)
```

只有在某道题上超过当前集成，才算有贡献；贡献按那道题的权重放大。一个平均分不高、但恰好补上两道难题的子图，会比一个处处略好的通用子图得分更高。

默认做法是**重加权**：均匀抽题，权重放在适应度里。配合随机森林插件时改为**重采样**：按权重有放回抽题，适应度里不再乘权重。两种不要同时用，否则难题被强调两次。

### 7.3 主循环

```python
def fit_adaboost(train, heldout, cfg, policy, proposer, ev) -> list[Subgraph]:
    ensemble, stall = [], 0
    g_tr = {i.item_id: 0.0 for i in train}        # 集成在训练题上的逐题 g（起始解的 g 为 0）
    g_ho = {i.item_id: 0.0 for i in heldout}
    y0_tr, y0_ho = start_solutions(train), start_solutions(heldout)
    for m in range(cfg.rounds):
        r = residual_weights(g_tr, cfg)
        report = residual_report(train, g_tr, r)             # 哪些题差、失败类型分布
        cands = []
        for k in range(cfg.forest.k):
            bag, oob = cfg.sampler.split(train, weights=r, seed=(m, k))
            seed = pick_seed(library, r, cfg)                # 单写者，或库里专攻当前难题的子图
            fit = lambda h, items, inp: fit_ada(h, items, inp, g_tr, r, cfg)
            h = grow_subgraph(seed, bag, y0_tr, fit, proposer, policy, cfg)
            cands.append((fit(h, oob or bag, y0_tr), h))
            library.add(h, mode="ada", per_item=ev.run(h, bag, y0_tr))
        best = max(cands, key=lambda t: t[0])[1]

        res_ho = ev.run(best, heldout, y0_ho)
        new_g_ho = {i: max(g_ho[i], res_ho[i].g) for i in g_ho}
        if lower_ci(paired(new_g_ho, g_ho)) <= cfg.accept_margin:
            stall += 1
            if stall >= cfg.patience: break
            continue
        stall = 0
        ensemble.append(best)
        res_tr = ev.run(best, train, y0_tr)                  # 补全它在全部训练题上的结果
        g_tr = {i: max(g_tr[i], res_tr[i].g) for i in g_tr}  # 残差更新：逐题取最大
        g_ho = new_g_ho
        if len(ensemble) >= cfg.max_members: break
    return ensemble
```

### 7.4 子图库与离线组合选择

因为每个子图都在 y0 上独立运行，它的逐题结果不依赖集成里的其他成员，可以直接复用。库里的每条记录是（子图，逐题 g 向量，成本）。

```python
def select_ensemble(library, train_ids, cfg) -> list[Subgraph]:
    pool = [e for e in library.entries(mode="ada") if e.covers(train_ids)]
    chosen, g = [], {i: 0.0 for i in train_ids}
    while len(chosen) < cfg.max_members:
        def gain(e): return mean(max(g[i], e.g[i]) - g[i] for i in train_ids) - cfg.lam * e.cost
        best = max(pool, key=gain)
        if gain(best) <= 0: break
        chosen.append(best.subgraph)
        g = {i: max(g[i], best.g[i]) for i in train_ids}
    return chosen
```

- 这是不花任何 LLM 调用的贪心选择。训练结束后跑一次，得到的组合再在留出题上验证一次，比逐轮得到的集成更好才替换；
- 库里很多子图只在袋内题上评估过。进入候选池之前，先给适应度排名靠前的若干个补全其余训练题；
- `pick_seed` 也用这个库：按当前权重 r 计算每条记录的加权收益，最高的那个可以作为下一轮的起点，这样前几轮被淘汰的“专才”有机会被重新启用。

### 7.5 推理

对一道新题：生成 y0，集成里的每个子图各自在 y0 上运行，取分数最高的解。成本是所有成员之和，所以 `max_members` 默认只取 4。

## 8. 变动选择接口：默认方式与随机森林插件

“每一步考虑哪些变动”被抽成两个可替换的接口：提议器决定候选变动，抽题器决定用哪些题。默认实现服务于 GBDT 和 AdaBoost；随机森林插件通过配置接入，不改主循环的任何代码。

### 8.1 两个接口

```python
class EditProposer(Protocol):
    def start_search(self, seed: int) -> None: ...                 # 每次长子图前调用一次
    def propose(self, h: Subgraph, ctx: GrowContext) -> list[Edit | MacroEdit]: ...

class ItemSampler(Protocol):
    def split(self, items: list[Item], weights: dict | None, seed) -> tuple[list[Item], list[Item]]: ...
    # 返回（袋内题，袋外题）；不抽样时袋外为空

PROPOSERS: dict[str, type[EditProposer]] = {}
def register_proposer(name: str): ...                              # 装饰器，供以后扩展
```

主循环只依赖这两个协议。新增一种变动选择方式，只需要写一个类并注册。

### 8.2 默认提议器：全部原子编辑

```python
@register_proposer("atomic")
class AtomicProposer:
    def start_search(self, seed): pass
    def propose(self, h, ctx):
        return legal_edits(h, ctx.limits)          # 6 种原子编辑的全部合法实例
```

默认方式下不做任何随机限制，“选哪个变动”完全交给策略 f：f 看残差报告和历史记录给候选排序，只有前 `top_k` 个被真正评估。这是 GBDT 和 AdaBoost 版本的默认变动选择。

### 8.3 随机森林插件：随机子空间 + 宏改动

```python
@register_proposer("random_subspace")
class RandomSubspaceProposer:
    def __init__(self, role_frac, edit_type_frac, resample, macro_rate, macros): ...

    def start_search(self, seed):
        self.rng = Random(seed)
        self._draw()                               # 抽这次搜索允许的角色子集和编辑类型子集

    def propose(self, h, ctx):
        if self.resample == "per_step":
            self._draw()                           # 每一步重抽，对应“每次分裂换一批特征”
        atomic = [e for e in legal_edits(h, ctx.limits)
                  if e.kind in self.edit_types and roles_of(e) <= self.roles]
        macro = [m for name in self.macros
                 for m in expand_macro(name, h, ctx.limits)
                 if roles_of(m) <= self.roles]
        return atomic + self.rng.sample(macro, k=round(len(macro) * self.macro_rate))
```

宏改动是预先定义的模板，一步完成几个原子编辑：

| 宏 | 展开成的原子编辑 | 直观含义 |
| --- | --- | --- |
| `critique_and_rewrite` | 加一个 `critic` 读输入；加一个 `writer` 读它并成为新的输出节点 | 先挑毛病，再重写 |
| `analyze_first` | 加一个 `analyst` 读输入；把它连到输出节点 | 先做定量分析，再写计划 |
| `repair_tail` | 在输出节点后加一个 `repairer`；再给它连一条来自输入的边 | 最后补一道合法性修复 |

宏改动在记录里保留它的原子步骤。配置 `macro.ablate_on_accept: true` 时，一个宏被采纳后会额外评估它的每个前缀，用来判断是哪一步在起作用。

### 8.4 森林的另外两个部件

- **抽题器：** `all`（不抽样）、`subsample`（无放回抽一部分）、`weighted_bootstrap`（按残差权重有放回抽样）；
- **K 路并行与袋外挑选：** `forest.k > 1` 时，每一轮并行长 K 个子图，各用不同的抽题和子空间种子，按袋外题上的适应度挑一个。并列时选逐题收益与当前集成相关性最低的那个。

### 8.5 配置开关

```yaml
mode: gbdt                  # gbdt | adaboost
search:
  proposer: atomic          # atomic | random_subspace
  top_k: 3
  random_subspace:
    role_frac: 0.5
    edit_type_frac: 0.67
    resample: per_search    # per_search | per_step
    macro_rate: 0.5
    macros: [critique_and_rewrite, analyze_first, repair_tail]
forest:
  k: 1
sampler:
  kind: all                 # all | subsample | weighted_bootstrap
  frac: 0.6
```

| 预设 | `proposer` | `forest.k` | `sampler.kind` | 说明 |
| --- | --- | --- | --- | --- |
| GBDT 默认 | `atomic` | 1 | `all` | 最简单，先跑通 |
| AdaBoost 默认 | `atomic` | 1 | `all` | 权重放在适应度里 |
| GBDT + 森林 | `random_subspace` | 3 | `subsample` | 多样性来自子空间和抽题 |
| AdaBoost + 森林 | `random_subspace` | 3 | `weighted_bootstrap` | 权重改由抽题体现，适应度不再乘权重 |

验收要求：关闭插件时（前两行），运行结果必须和接入插件之前逐位一致。

## 9. 进化是怎么发生的

系统里有三样东西在变，变化的速度不同：一次搜索内变的是**子图**，轮与轮之间变的是**集成和残差**，代与代之间变的是 **playbook**，也就是搜索策略 f 所依据的知识。前两层是拓扑层的自我改进，由 boosting 完成；第三层是元层的自我改进，改的是搜索方法本身。两层都属于递归自我改进，区别在于前者会收敛，后者的收益会累积到以后的每一次搜索。

### 9.1 三个时间尺度

| 尺度 | 什么在变 | 靠什么决定 | 信号来自 | 实现位置 |
| --- | --- | --- | --- | --- |
| 一次搜索内（几步） | 一个子图的结构 | 提议器出候选，f 排序，贪心保留增益最大的 | 训练题（袋内） | `search/grow.py` |
| 轮与轮之间（几轮） | 集成多一个子图；残差随之改变 | 袋外挑选 + 留出题接受准则 | 袋外题、留出题 | `boost/` |
| 代与代之间（几代） | playbook：原则增减、权重更新、树的结构 | 试用记录累计，按后验分位数启用或归档 | 多个任务族上的运行 | `playbook/`（M4） |

### 9.2 一轮的数字走查（AdaBoost 模式，6 道训练题）

**第 1 轮。** 集成为空，所有题的 g 都是 0，权重相等。从单写者出发长子图：f 排在最前的三个候选里，“加一个 `analyst`”的适应度最高，被保留；再下一步最好的候选只多出 0.002，低于 `min_gain`，停止。得到 h1 = `analyst → writer`，留出题上置信下界远高于阈值，接受。

h1 在训练题上的结果，以及由此算出的第 2 轮权重（ε = 0.1）：

| 题 | h1 之后的 g | 打分器诊断 | 1 / (0.1 + g) | 归一化权重 |
| --- | --- | --- | --- | --- |
| i1 | 0.9 | 合法 | 1.00 | 0.05 |
| i2 | 0.8 | 合法 | 1.11 | 0.05 |
| i3 | 0.7 | 合法 | 1.25 | 0.06 |
| i4 | 0.1 | 非法：某天资金不足 | 5.00 | 0.24 |
| i5 | 0.0 | 非法：某天资金不足 | 10.00 | 0.48 |
| i6 | 0.3 | 合法，但地块闲置 | 2.50 | 0.12 |

i4 和 i5 两道题拿走了 72% 的权重。这就是“方向”：下一个子图几乎只为这两道题而生。

**第 2 轮。** 残差报告显示，高权重的题大多是非法计划。f0 有一条规则：非法解占高权重题的多数时，把带 `repairer` 的变动排到最前。于是被评估的候选里有两个值得对比：

| 候选 | 各题单独的 g（i1 到 i6） | 单独平均 | 超过集成的题 | 加权适应度 |
| --- | --- | --- | --- | --- |
| A：通用型，处处略好 | 0.95, 0.85, 0.75, 0.15, 0.05, 0.35 | 0.52 | 全部，各 +0.05 | 0.05 |
| B：`writer → repairer` | 0.4, 0.3, 0.3, 0.6, 0.7, 0.2 | 0.42 | i4 +0.5，i5 +0.7 | 0.24×0.5 + 0.48×0.7 ≈ 0.46 |

B 的单独平均分更低，但加权适应度是 A 的 9 倍，被选中并通过留出验证。集成的逐题 g 变成 0.9, 0.8, 0.7, 0.6, 0.7, 0.3，平均从 0.47 升到 0.67。

**第 3 轮。** 权重变得平缓，最高的是 i6。新候选在留出题上的置信下界没有超过阈值，不接受；连续 `patience` 轮不接受后停止。最终集成是 \[h1, h2\]。

如果换成 GBDT 模式，区别只在第 2 轮：新子图拿到的输入不是 y0，而是 h1 改过的解，适应度直接看它在这些解上又提升了多少，不需要权重。

### 9.3 策略 f 看到什么、决定什么

```python
class Policy(Protocol):
    def rank(self, h: Subgraph, cands: list[Edit | MacroEdit], obs: Obs) -> list[Edit | MacroEdit]: ...

@dataclass
class Obs:
    subgraph: dict          # 节点数、各角色数量、深度
    residual: dict          # 权重集中度、各失败类型在高权重题里的占比、平均 g
    edit_stats: dict        # 每类变动的历史尝试次数、平均增益、正增益比例（可按失败类型细分）
    round: int
    step: int
    budget_left: float
```

f 只决定一件事：这一步哪几个候选值得花钱评估。它看不到题号、游戏名和具体分数目标，只能依赖上面这些抽象信号，这样学到的规则才可能迁移到别的难度和别的游戏。

手写的初始策略 f0 只有四条规则：

1. 高权重题里非法解占多数时，带 `repairer` 的变动优先；
2. 子图里还没有读者时，加读者的变动优先，`analyst` 排在 `critic` 前面；
3. 其余候选按历史平均增益加一个探索项排序（尝试次数少的略微靠前）；
4. 删除类变动排在最后，除非当前子图的成本惩罚已经超过它的增益。

### 9.4 f 靠什么进化：每一次评估都是一条训练数据

`grow_subgraph` 里每评估一个候选，就写下一条记录：

```python
@dataclass
class EditOutcome:
    obs: Obs                # 当时 f 看到的全部信息
    edit: dict              # 变动类型、涉及的角色、是否为宏
    rank_by_f: int          # f 把它排在第几
    delta_fit: float        # 真实评估出来的增益
    cost: float
```

这些记录回答的正是 f 要学的问题：在什么样的残差形态下，哪类变动真的有用。一个好的 f，应该把 `delta_fit` 高的候选排在前面。

### 9.5 元层：用 playbook 进化 f（M4）

Playbook 是跨运行积累的编排知识库，按失败类型组织，记录“遇到这类残差，用什么编排”。它一份数据同时充当三个角色：搜索时的提议器、搜索时的排序策略、推理时的路由器。f 的进化不再是改写代码，而是 playbook 一条一条地增加原则、更新权重、淘汰无效原则。

#### 9.5.1 子图就是编排，原则就是子图

Boosting 每接受一个子图，就等于发现了一种针对某类残差的编排方式。把它去掉题目信息、只留结构和它补上的失败类型，就是一条原则。因为子图最多 3 个节点，原则可以直接存子图本身，不需要另一套描述语言。

| 粒度 | 记录什么 | 从哪里来 | 用在哪里 |
| --- | --- | --- | --- |
| 整图原则 | 一个完整子图的结构 | 每一轮被接受的子图 | 搜索的起点；推理时的路由目标 |
| 改动原则 | 一段编辑序列，即一个宏改动 | 长子图时每一步被保留的变动 | 搜索中的候选变动 |

一个整图原则拆开就是若干改动原则：`analyst → writer → repairer` 这个子图，是从单写者出发依次保留“加分析者”和“尾部加修复者”两步得到的。两种粒度都入库，前者让好的编排整体复用，后者让其中的每一步能在别的子图上单独复用。

```python
@dataclass
class Principle:
    pid: str
    kind: str                   # "subgraph" | "edit"
    subgraph: Subgraph | None   # 整图原则：完整结构
    steps: tuple[Edit, ...]     # 改动原则：相对挂接点的编辑序列
    description: str            # 通用描述，须通过防泄漏检查
    sources: list[tuple]        # （任务族，运行号，轮次）
    status: str                 # trial | enabled | archived
    cost: float                 # 平均 token

@dataclass
class LeafStats:                # 每片叶子 × 每条原则单独统计
    s: int                      # 成功次数，不含来源运行
    f: int                      # 失败次数
    gains: list[float]          # 每次试用的连续增益
    families: set[str]          # 成功来自哪些任务族

@dataclass
class Playbook:
    tree: DiagNode              # 诊断问题 → 叶子（失败类型）
    principles: dict[str, Principle]
    stats: dict[tuple[str, str], LeafStats]    # 键为（叶子，原则）
    version: int
```

合并不需要大模型判断：两个子图的规范化哈希相同就是同一条原则，来源累加。

#### 9.5.2 分类树与诊断信号

内部节点是诊断问题，只能用推理时也拿得到的信号回答。Farm 上可用的信号有两类：

- **任务特征：** 作物种类数、地块数、天数、初始资金与最便宜种子的比值；
- **起始解的早期信号：** y0 是否合法、第一条报错的类型（资金不足、地块超限、格式错误）、地块闲置率。

第二类信号要对 y0 跑一次模拟器。Farm 上这是允许的；换到打分需要标准答案的数据集时，只能用第一类和不依赖答案的检查。每个诊断问题都标注它属于哪一类。

初始的树 v0 就是把 f0 的第一条规则写成数据：一个诊断问题“y0 是否非法”，两片叶子，“非法”叶子上挂一条“尾部加修复者”的改动原则，状态为试用。新增诊断问题和拆分叶子由开发智能体提出，只有在已有记录上回放时**路由正确率**不下降才采用。路由正确率是指：按诊断问题分到的叶子，与训练题上打分器完整诊断给出的失败类型一致的比例。

#### 9.5.3 三处接入

**搜索时：提议与排序。** 把训练题逐题路由到叶子，每片叶子的“残差质量”是落在它上面的题的权重之和。

```python
@register_proposer("playbook")
class PlaybookProposer:
    def start_search(self, seed):
        mass = leaf_residual_mass(self.pb.tree, self.items, self.weights)
        self.leaf = max(mass, key=mass.get) if self.k == 1 else sample(mass, seed)   # 森林模式按质量抽叶子
    def propose(self, h, ctx):
        from_pb = [expand(p, h) for p in self.pb.on_leaf(self.leaf, kind="edit")
                   if not p.sourced_from(ctx.run_id)]          # 不在自己的来源运行上试用
        return from_pb + legal_edits(h, ctx.limits)            # 原子编辑兜底

class PlaybookPolicy:
    def rank(self, h, cands, obs):
        pb_c = [c for c in cands if c.pid]                     # 来自 playbook 的候选
        pb_c.sort(key=lambda c: thompson(self.pb.stats[(obs.leaf, c.pid)]), reverse=True)
        return pb_c + self.fallback.rank(h, [c for c in cands if not c.pid], obs)   # 其余交给 f0
```

- 排序用汤普森采样：从每条原则在这片叶子上的 Beta(s+1, f+1) 里抽一个值来排。没试过的原则自然会得到机会，试过且有效的排得稳定靠前；
- 起点也来自 playbook：`pick_seed` 优先取这片叶子上的整图原则，没有才用单写者；
- 森林模式下 K 路搜索各抽一片叶子，多样性直接来自“各攻一类失败”，比随机限定角色更有方向；
- playbook 没覆盖的决定（何时停止、何时删节点）仍由 f0 的其余规则负责。

**每轮结束：写回。** 被接受的子图通用化为整图原则，它生长过程中被保留的每一步通用化为改动原则，都挂在它贡献边际收益最多的那片叶子上，状态为试用。同时，这一轮里所有来自 playbook 的候选，不论是否胜出，都记一次试用。

**推理时：路由。** 对一道新题：生成 y0，回答诊断问题走到叶子，只运行这片叶子上状态为“启用”的整图原则（按后验均值取前 1 到 2 个）。叶子上没有启用的原则时，退回到“全部成员都跑、取最好”，打分器不可用时直接返回 y0。配置项为 `inference.router: playbook | all`。

这样集成和 playbook 是同一件东西的两个视角：集成是某次运行里被接受的子图列表，playbook 是把它们按失败类型归档、跨运行累计证据之后的结果。

#### 9.5.4 生命周期

| 阶段 | 规则 |
| --- | --- |
| 产生 | 子图被接受时写入；通过防泄漏检查（描述里不含题号、种子、具体作物名和数值） |
| 合并 | 规范化哈希相同即合并，来源累加 |
| 试用 | 一次试用 = 一次运行里，该原则在所属叶子的题上被评估一次。这些题上（优先用袋外题）平均增益的置信下界大于 0 记成功，否则记失败。同一次运行只计一次 |
| 启用 | Beta(s+1, f+1) 的 10% 分位数大于 0.5，且成功来自至少 2 个不同的任务族 |
| 退回 | 不再满足启用条件时退回试用 |
| 归档 | 每代结束时，试用不少于 3 次、后验均值低于 0.3 且处于最低 20% 的原则归档；归档不删除 |

三条和原设计不同的地方：

- **来源运行不计数。** 原则是因为在来源运行里成功才被创建的，那次成功不是独立证据，不进 s，也不算进“2 个不同任务族”；
- **启用看分位数，不看均值。** 2 次成功、0 次失败时均值是 0.75，但 10% 分位数只有约 0.46，过不了线；要 3 次成功、0 次失败（约 0.56）才行。这是为了对付 LLM 输出的噪声；
- **统计按（叶子，原则）分开。** 同一条原则可以挂在多片叶子上，各自累计。它第一次出现在某片新叶子上时，用它在其他叶子上的合并成功率折成 2 次虚拟试用作为先验。

“任务族”指一组互不重叠的题，按（游戏，难度）划分，例如 Farm 难度 2、Farm 难度 3、Combat 难度 3。

#### 9.5.5 一代的流程

```python
def playbook_generation(pb: Playbook, families, cfg) -> Playbook:
    for fam in families:
        run = run_boosting(fam, cfg,
                           proposer=PlaybookProposer(pb),
                           policy=PlaybookPolicy(pb, fallback=f0))
        for t in run.playbook_trials:                 # 每个（叶子，原则）在本次运行里至多一条
            if pb.principles[t.pid].sourced_from(fam):
                continue                              # 来源任务族上不计数
            pb.record(t.leaf, t.pid, success=t.gain_lcb > 0, gain=t.gain, family=fam)
        for h, leaf, kept_steps in run.accepted:      # 每个被接受的子图都是一条编排
            pb.add_or_merge(as_subgraph_principle(h), leaf, source=(fam, run.id))
            for steps in kept_steps:
                pb.add_or_merge(as_edit_principle(steps), leaf, source=(fam, run.id))
    pb.update_status()                                # 启用、退回、归档
    pb.version += 1
    return pb
```

一代跑完，playbook 多了几条原则、更新了一批权重；下一代的搜索从更好的候选和更准的排序开始，它产生的子图又写回 playbook。这一圈就是元层的递归。

#### 9.5.6 怎么确认 playbook 真的让搜索变好

每个版本报告三项：

- **路由正确率**（定义见 9.5.2）；
- **命中率**：来自 playbook 的候选里，试用成功的比例，对比同一批运行里原子编辑候选的正增益比例；
- **搜索效率**：在一个没参与积累的留出任务族上，用相同预算和相同随机种子，分别以“playbook 当前版本”和“只用 f0”跑 boosting，比较留出题 g 随 token 花费的曲线下面积。

第三项最贵，每隔几个版本做一次即可。它是 playbook 整体有没有用的最终判据：前两项好看而第三项不动，说明原则只是在复述 f0 已经会做的事。

开发智能体（可以用 mini-swe-agent）在这一版里只负责两件事：给新原则写通用描述，以及提议新的诊断问题。它不直接改 `policy/` 里的代码。

## 10. 配置、缓存、日志与测试

一次运行由一个 YAML 配置完全决定，所有 LLM 调用和评估结果都落盘缓存，保证同一配置可以重放。

### 10.1 主要超参数与默认值

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `rounds` | 6 | 最多几轮 boosting |
| `patience` | 2 | 连续几轮不被接受就停止 |
| `accept_margin` | 0.01 | 留出题上 g 增量的置信下界要超过的阈值 |
| `max_members` | 4 | 集成里最多几个子图 |
| `limits.max_nodes` | 3 | 单个子图的节点上限 |
| `search.top_k` | 3 | 每一步真正评估几个候选 |
| `search.max_steps` | 3 | 一次搜索最多走几步 |
| `search.min_gain` | 0.005 | 低于这个增益就停止生长 |
| `eval.n_repeats` | 2 | 每个（子图，题）重复几次取平均 |
| `lam` | 待校准 | 成本惩罚系数，按每千 token 计 |
| `utility` / `eps` / `weight_cap` | `log` / 0.1 / 5 | AdaBoost 的权重设置 |

`lam` 需要先测量再定：跑一遍单写者和一个手写三节点子图，看 g 的提升和 token 花费的量级，让三节点子图的成本惩罚大约等于 0.02 的 g。

### 10.2 缓存与可复现

- LLM 调用缓存：键为（模型，提示词哈希，温度，重复序号）；
- 评估缓存：键见第 5.4 节；
- 所有随机性都从配置里的根种子派生，按（轮，第几路，步）取子种子；
- 每次运行把解析后的配置、题目清单、代码版本号写进 `runs/<id>/`。

### 10.3 日志

| 文件 | 每行一条 | 用途 |
| --- | --- | --- |
| `rounds.jsonl` | 每轮的残差分布、候选、是否接受、留出增益 | 画学习曲线 |
| `edit_outcomes.jsonl` | 第 9.4 节的 `EditOutcome` | f 的训练数据 |
| `library.jsonl` | 子图、逐题 g、成本、来源 | 离线组合选择 |
| `budget.jsonl` | 累计 LLM 调用数、token 数、打分次数 | 与基线对齐预算 |

### 10.4 测试

所有单元测试用 FakeLLM 和一个玩具基准，不花钱、几秒跑完。玩具基准的设计是：解是一个整数向量，分数是它与隐藏目标的接近程度；FakeLLM 的“角色”是确定性的改写规则，其中只有某些角色组合能修好某类题。这样可以断言：

- 门控之后分数不下降；
- GBDT 模式下，集成每加一个阶段，训练题平均 g 单调不降；
- AdaBoost 模式下，权重与 g 反向变化，且不超过上限；
- 专攻难题的候选在加权适应度下胜过通用候选（复现第 9.2 节的表）；
- 关闭随机森林插件时，结果与基准快照逐位一致；
- 留出题和终测题的题号从不出现在搜索阶段的日志里。

最后一条是防泄漏的硬性检查，放进持续集成。

### 10.5 在 Slurm 服务器上跑实验

当前版本不需要容器：Farm 的出题和打分是纯 Python 计算，子图的每个节点只是一次 LLM 接口调用，不执行任何生成的代码。真正要先解决的是计算节点能不能访问 LLM 接口。

**LLM 访问的三种方式**

| 方式 | 做法 | 适用情况 |
| --- | --- | --- |
| 直接出网 | 计算节点直接调外部接口 | 集群允许计算节点出网 |
| 代理 | 设置 `HTTPS_PROXY` 指向集群提供的代理 | 计算节点不能出网，但有代理 |
| 本地模型 | 在 GPU 节点上用 vLLM 起一个开源模型，主程序调它的本地地址 | 不能出网，或想避开接口限速和费用 |

`llm.py` 里的接口地址和模型名都从配置读取，三种方式只改配置，不改代码。用本地模型时，把模型服务作为一个单独的作业提交，把它的节点名和端口写到共享文件里，实验作业启动时读取。

**作业组织**

- 主循环本身很轻，只发请求和跑模拟器，申请少量 CPU 即可；
- 每一轮的 K 路并行搜索用作业数组提交，每路一个任务，各用自己的子种子；
- 所有作业共用共享文件系统上的缓存目录。缓存按键分文件写入，先写临时文件再原子改名，避免多个作业同时写坏同一个文件；
- 作业可能被抢占或超时。每轮结束时把集成、残差和随机状态写进 `runs/<id>/checkpoint.json`，重新提交时从最近一轮继续；已缓存的评估不会重复花钱；
- 并发请求数在配置里设上限，避免触发接口限速，也避免压垮自己起的模型服务。

```bash
#!/bin/bash
#SBATCH --job-name=boost-farm
#SBATCH --array=0-2                 # K = 3 路并行搜索
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=runs/%x_%A_%a.log

source activate boosttopo
export BOOSTTOPO_CACHE=/shared/$USER/boosttopo_cache
python -m boosttopo.search.grow_worker \
    --config configs/farm_adaboost.yaml \
    --round "$ROUND" --branch "$SLURM_ARRAY_TASK_ID"
```

**在这台服务器上做不了的事**

| 内容 | 原因 |
| --- | --- |
| SelfImprove-Bench 的 MC 和 Astray | 基于浏览器的游戏 |
| EdgeBench、TerminalBench | 每个任务都要启动自己的容器 |
| 基准自带的 RSI 方法对照（ACE、AHE、HyperAgents） | 可能依赖 Docker，需读仓库确认 |

BrowseComp-Plus 不需要容器，以后可以在这台服务器上接，但要预留建检索索引的内存和磁盘。

## 11. 里程碑与验收标准

五个里程碑依次推进，每个都有明确的通过条件；前一个不过就不开始下一个。

| 里程碑 | 内容 | 对应任务单 | 通过条件 |
| --- | --- | --- | --- |
| M0 基础代码 | 适配层、子图、评估器、缓存 | T1–T4 | 单写者在 40 道 Farm 题上跑通；重跑一次不产生新的 LLM 调用 |
| M1 GBDT | 串联模式全流程 | T5、T6 | 留出题平均 g 显著高于单写者，也高于同预算的“单写者多次采样取最好” |
| M2 AdaBoost | 权重、取最好、子图库、离线组合 | T7 | 留出题平均 g 不低于 M1；至少一轮选中了单独平均分不是最高的候选 |
| M3 森林插件 | 随机子空间、宏改动、K 路并行 | T8 | 开关关闭时结果不变；开启后同预算下留出 g 不下降，候选之间的相关性降低 |
| M4 Playbook | 原则库与生命周期（T9）；`PlaybookProposer` 与 `PlaybookPolicy`（T10）；推理路由（T11） | T9–T11 | 至少 2 条原则满足启用条件；留出任务族上，带 playbook 的搜索在相同预算下优于只用 f0；路由推理的留出 g 不低于“全部成员取最好”的 95%，且 token 更少 |

### 11.1 必须同时报告的基线

所有基线和我们的系统用相同的 LLM 调用预算和相同的打分次数。

| 基线 | 做法 | 用来排除什么 |
| --- | --- | --- |
| 单写者 | 起始解 y0 本身 | 起点 |
| 多次采样取最好 | 单写者采样 N 次，用打分器取最高 | 提升只是来自“多试几次再挑” |
| 手写拓扑 | `analyst → writer → repairer` | 搜索并不比人工设计好 |
| 随机策略 | 把 f 换成随机排序 | f0 没有起作用 |
| 无残差 | AdaBoost 模式下把权重固定为均匀 | 重加权没有起作用 |

“多次采样取最好”是最关键的一条。我们的门控和取最好都用到了打分器，如果不和它比，就无法说明拓扑本身有贡献。

### 11.2 最终报告的内容

- 留出题与终测题（G1、G2、G3）上的平均 g 和原始分数，带置信区间；
- 留出 g 随累计 token 的曲线，所有方法画在同一张图上；
- 每一轮被接受的子图结构和它补上的题；
- 各类变动的平均增益表，按失败类型细分。

## 12. 待确认事项与参考资料

动手之前有四件事需要先核对，其中前两件会直接影响适配层的写法。

### 12.1 待确认

1. **SelfImprove-Bench 的实际代码接口。** 本文档的适配层是按论文描述设计的，仓库里的函数名、配置字段和打分入口还没有对照过。T2 的第一步是读仓库，再把 `BenchAdapter` 落到真实接口上。
2. **打分器的使用口径。** 基准原本的 Farm 协议是智能体一次提交一个计划。我们在门控和取最好时会对同一道题多次打分，所以结果不能直接和论文表格里的数字比较。对外只和第 11.1 节里同预算的基线比，并报告每道题的打分次数。
3. **mini-swe-agent 的命令行参数。** 第 3.1 节的命令是示意，版本之间参数可能不同，以 `mini --help` 为准。
4. **模型选择。** 子图里的角色用哪个模型、开发智能体用哪个模型，还没有定。建议角色用便宜的模型，开发智能体用更强的模型，两者都在配置里固定。

另有两件事要在 Slurm 服务器上先确认：

- **计算节点能否访问 LLM 接口。** 这决定第 10.5 节三种访问方式选哪一种，要在 T1 之前测清楚；
- **SelfImprove-Bench 的出题和打分能否脱离 Docker 单独调用。** 论文描述里它们是纯计算，但仓库的运行脚本可能默认走容器。如果不能单独调用，就按论文里的规则在 `bench/` 里自己实现 Farm 的生成器和模拟器，并用官方实例数据集核对分数。

### 12.2 已知的设计取舍

- 留出题被每一轮的接受判断反复使用，所以单独保留了只用一次的终测题；
- AdaBoost 模式的推理成本随集成大小线性增长，M4 之前用 `max_members` 限制，M4 起由 playbook 路由，只运行对应叶子上的子图；
- GBDT 模式的评估记录不能离线复用，所以子图库和离线组合选择只对 AdaBoost 模式有效；
- 子图库和 playbook 的分工：子图库属于单次运行，保存逐题结果；playbook 跨运行，只保存结构和按失败类型汇总的统计，不含任何题目信息；
- playbook 的“搜索效率”对比是线上配对实验，比较贵，每隔几个版本做一次；平时靠路由正确率和命中率监控。

### 12.3 参考资料

- [SelfImprove-Bench 论文](https://arxiv.org/html/2611.02797)：游戏定义、难度参数、G0–G4 泛化阶梯、HCI 指标；
- [SelfImprove-Bench 项目页](https://theseus-labs-rsi.github.io/SelfImprove-Bench)与[实例数据集](https://huggingface.co/datasets/DreamEternal/SelfImprove-Bench-Cases)；
- [mini-swe-agent 文档索引](https://docsearch.algolia.com/mcp/docs/repo/swe-agent/mini-swe-agent)：`mini` 命令行、`mini-extra config` 配置、Python 接口。
