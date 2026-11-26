# Boosted Workflow：单智能体版本实现文档

Nov 24, 2026 · @dogecheemse

本文档说明如何把现有的 Boosted Topology 代码改成单智能体版本：被优化的对象从“几个角色组成的小团队”换成“一个智能体按步骤调用技能和工具的工作流”，boosting、留出验证和 playbook 的做法保持不变。

## 1. 目标与范围

这一版要回答一个问题：在相同花费下，按残差一段一段长出来的单智能体工作流，能不能赢过“同一个智能体多试几次、挑最好的”。代码在现有仓库（`~/projects/boostrsi_next`）上改，不另起项目。

### 1.1 为什么改成单智能体工作流

前两轮实验里有效的东西都不是“团队”，而是工作流里的模式：

| 有效的做法 | 它是什么 |
| --- | --- |
| 修复循环 | 一个带检查的循环 |
| 多写几次挑最好 | 并行分叉，再选择 |
| 后来加的循环属性 | 一种控制结构 |

分工型的角色（分析者、挑错者）基本没被搜索选中，手工设计的三角色团队在两个设置下都输给了多次采样。所以这一版把优化对象换成工作流：一个智能体，按步骤调用不同的技能和工具。

### 1.2 和第一版的关系

| 第一版 | 这一版 | 变不变 |
| --- | --- | --- |
| 节点 = 角色 | 步骤 = 一次带技能的模型调用、一次工具调用，或一次检查 | 变 |
| 有向无环图，每个节点跑一次 | 顺序、循环、分支、分叉四种结构 | 变 |
| 节点之间传消息 | 所有步骤读写同一份状态 | 变 |
| 残差、适应度、留出验证、接受准则 | 同左 | 不变 |
| 评估器、缓存、报告脚本、题目划分 | 同左 | 不变 |
| playbook（未实现） | 存的内容从子图换成工作流片段 | 设计不变 |

第一版的“角色子图”是这一版的一个特例：只用模型步骤、只用顺序结构。旧配置必须仍能逐行复现旧结果。

### 1.3 这一版不做的事

- 不做多智能体。等单智能体版本拿到一次对多次采样的胜利，再把同一套代码用回需要分工的任务；
- 不引入能直接生成计划的工具（见第 3.3 节）；
- 不换数据集和模型：仍是 Farm 难度 5，Qwen3-14B 开思考模式，单卡；
- playbook 放在最后一个里程碑。

## 2. 工作流的表示

工作流是一棵由“步骤”和“控制块”组成的树，所有步骤读写同一份状态。它等价于一张带条件边和回边的流程图，和 LangGraph 表达的是同一种东西；用树来存，是因为合法性检查和编辑都简单得多。

### 2.1 状态

```python
@dataclass
class State:
    problem: str              # 题面，只读
    plan: str | None          # 当前计划
    best_plan: str | None     # 目前分数最高的计划
    best_score: float
    diag: str | None          # 最近一次检查或工具给出的诊断
    notes: dict[str, str]     # 工具输出和中间分析，按名字存
```

第一版里节点之间传消息，谁能看到什么由边决定。这一版改成共享状态：每个步骤声明自己读哪些字段、写哪个字段。`best_plan` 由执行器维护，每次 `plan` 被改写后自动打分并更新。

### 2.2 步骤

```python
@dataclass(frozen=True)
class Step:
    kind: str                 # "llm" | "tool" | "check"
    name: str                 # 技能名 / 工具名 / 检查名，见第 3 节
    reads: tuple[str, ...]    # 从状态里读哪些字段
    writes: str               # 写回哪个字段
```

| 类型 | 做什么 | 花费 |
| --- | --- | --- |
| `llm` | 用某个技能（一段固定提示词）调一次模型 | token，慢 |
| `tool` | 调一个确定性的函数 | 几乎为零 |
| `check` | 根据状态算出一个标签，供分支和循环使用 | 几乎为零 |

### 2.3 控制块

```python
@dataclass(frozen=True)
class Block:
    kind: str                 # "seq" | "loop" | "branch" | "fanout"
    children: tuple["Step | Block", ...]
    params: dict

Workflow = Block              # 根节点是一个 seq
```

| 控制块 | 含义 | 参数 |
| --- | --- | --- |
| `seq` | 按顺序执行子节点 | 无 |
| `loop` | 重复执行子节点 | `k`：最多几次；`stop`：`legal`（合法即停）或 `best`（跑满 k 次，分数到 1 才提前停） |
| `branch` | 先做一次检查，按标签走对应的子节点；没有对应支路就直接通过 | `check`：检查名；`cases`：标签到子节点的映射 |
| `fanout` | 从同一个状态出发，把子节点独立执行 n 次，保留分数最高的那一次 | `n`：分叉数 |

第一版的两个强对照在这套表示里都能原样写出来：修复循环是 `seq[起草, loop(k=5, stop=best)[修复]]`，多写几次挑最好是 `fanout(n=5)[起草]`。这是硬性要求：搜索空间必须包含要比的对照。

### 2.4 约束

- 嵌套深度不超过 3，模型步骤总数不超过 6；
- **最坏情况的模型调用数不超过上限 `max_calls`**（循环按 k 算、分叉按 n 算、分支按最贵的支路算）。这条约束代替第一版的“节点数不超过 3”；
- 每个步骤读的字段，必须在它之前被写过；
- `branch` 用的检查只能依赖推理时拿得到的信息。

### 2.5 一个例子

&#91;embedded content: 示例工作流 · 5 个步骤、1 次检查、1 个循环\]

这张图写成代码是 `seq[crop_stats, draft, branch(status){illegal: loop(k=3, stop=legal)[validate, repair], legal_low: seq[simulate, improve]}]`。每道题只走其中一条支路：不合法的最多 4 次模型调用，合法但分数低的 2 次，分数已经够高的只有起草那 1 次。图里为了简洁，只画了合法与不合法两条支路。

## 3. 技能与工具库（Farm）

技能是固定的提示词，工具是确定性的函数。这一版新增的能力主要在工具上：它们给合法的计划提供改进方向，这正是上一轮缺的那种反馈。

### 3.1 技能（模型步骤）

| 技能 | 读 | 写 | 作用 | 来源 |
| --- | --- | --- | --- | --- |
| `draft` | 题面、notes | plan | 从头写一份完整计划 | 原 `writer` |
| `repair` | 题面、plan、diag | plan | 按报错把不合法的计划改合法 | 原 `repairer` |
| `improve` | 题面、plan、notes | plan | 计划已合法，按模拟结果提高利润 | 新增 |
| `analyze` | 题面、notes | notes | 写一段策略分析，供后续步骤参考 | 原 `analyst` |

`repair` 和 `improve` 分开，是因为上一轮发现修复者对已经合法的计划没有报错可看、不知道往哪改。旧技能的提示词保持不动，只新写 `improve`。

### 3.2 工具与检查

| 名字 | 类型 | 输入 | 输出 | 写到 |
| --- | --- | --- | --- | --- |
| `validate` | 工具 | plan | 是否合法；不合法时给出第一条报错和出错当天的状态 | diag |
| `simulate` | 工具 | plan | 逐日的现金、空闲地块数、收获；最终利润；现金和地块闲置最多的几天 | notes\["sim"\] |
| `crop_stats` | 工具 | 题面 | 每种作物的成熟天数、单块地每轮利润、日均利润、最晚可种植日 | notes\["crops"\] |
| `status` | 检查 | plan | `illegal` / `legal_low` / `legal_high`（按当前分数是否达到阈值） | 分支标签 |
| `error_type` | 检查 | diag | `funds` / `not_mature` / `farmland` / `other` | 分支标签 |

`validate` 就是现有的打分器诊断。`simulate` 和 `crop_stats` 要新写，都基于基准自带的游戏规则。

### 3.3 工具的边界

工具可以计算“关于题目的事实”和“关于某一份已有计划的事实”，不能生成或搜索计划。

- 允许：算每种作物的收益、试跑一份计划、指出哪几天有闲钱或闲地；
- 不允许：贪心求解器、对计划做局部搜索、任何读取参考解的东西。

理由有两个。基准的参考解本身就是一个贪心搜索，给模型一个求解器等于直接把题解了，没有流程可优化。另外，这条边界要写进单元测试：工具模块不得导入参考解的计算代码。

### 3.4 工具输出怎么进提示词

每个工具的输出渲染成一段不超过约 400 字的文本，拼在技能提示词的固定位置。没有跑过的工具，对应位置留空。这样同一个技能在有工具和没工具时用的是同一段提示词，对比才公平。

## 4. 执行器、成本记账与缓存

执行器是一个递归解释器：按块的类型走子节点，每次计划被改写就打分并更新“最好的一版”。最终输出永远是 `best_plan`，这就是第一版门控的等价物。

```python
def run(node, st: State, ctx: Ctx) -> State:
    if isinstance(node, Step):
        return run_step(node, st, ctx)
    if node.kind == "seq":
        for child in node.children:
            st = run(child, st, ctx)
            if st.best_score >= 1.0: break            # 已达参考解，提前结束
        return st
    if node.kind == "loop":
        for i in range(node.params["k"]):
            st = run(seq(node.children), st, ctx.at(i))
            if st.best_score >= 1.0: break
            if node.params["stop"] == "legal" and is_legal(st): break
        return st
    if node.kind == "branch":
        label = CHECKS[node.params["check"]](st)
        child = node.params["cases"].get(label)
        return run(child, st, ctx) if child else st   # 没有对应支路就直接通过
    if node.kind == "fanout":
        outs = [run(seq(node.children), copy(st), ctx.at(j)) for j in range(node.params["n"])]
        return max(outs, key=lambda s: s.best_score)   # 各路可以并行发请求

def run_step(step, st, ctx):
    if step.kind == "llm":
        out, usage = ctx.llm.call(render(SKILLS[step.name], st), seed=ctx.seed(step))
        ctx.cost.add_llm(usage)
    else:
        out = TOOLS[step.name](st)                     # 工具和检查：确定性，不计 token
        ctx.cost.add_tool(step.name)
    st = write(st, step.writes, out)
    if step.writes == "plan":
        res = ctx.bench.score(ctx.item, st.plan); ctx.cost.add_scorer()
        if res.score > st.best_score:
            st = replace(st, best_plan=st.plan, best_score=res.score)
    return st
```

### 4.1 三条语义，针对上一轮查出的问题

- **进展不丢。** 循环里的下一次修复接着 `plan`（最新一版）改，而不是退回 `best_plan`。这样“修好了第 5 天、第 9 天又出错”的部分进展能累积。需要从最好的一版重新出发时，用一个显式的工具步骤 `reset_to_best`；
- **每次执行都是新采样。** 随机种子由（运行种子、步骤在树里的路径、循环或分叉的序号、重复序号）派生，缓存键里也包含这些。同一个步骤在不同位置、不同轮次不会取回相同的输出；
- **成本按实际发生记。** 模型 token 含思考部分；打分次数和工具调用次数单列。分支没走到的支路不计费，提前停止的循环只计已执行的次数。

### 4.2 起始解

工作流包含起草这一步，所以不再需要单独生成的 y0 作为输入。改进幅度 g 的起点仍然用已缓存的 y0 分数，这样和前几轮的数字可比；但所有方法的 token 都从第一次起草算起，不再把 y0 的花费加进去。

### 4.3 并行

单卡上实测，必须顺序执行的流程约每小时 57 次调用，能并行的约 76 次。执行器对 `fanout` 的各路和不同题目之间并发发请求，并发数由客户端统一限制。

## 5. 编辑集合

搜索每一步对工作流做一次改动。改动分两类：原子编辑只动一处，片段一次插入一小段现成的流程。每个改动都必须把一个合法的工作流变成另一个合法的工作流。

### 5.1 原子编辑

位置用树上的路径表示，比如 `(2, 0)` 是根节点第 3 个子节点的第 1 个子节点。

| 编辑 | 含义 | 合法性条件 |
| --- | --- | --- |
| `insert_step(path, step)` | 在某个位置插入一个步骤 | 它读的字段在此之前已被写过；不超步骤数和调用数上限 |
| `remove(path)` | 删除一个步骤或一整个块 | 删除后仍至少有一个写 plan 的步骤；后续步骤读的字段仍有来源 |
| `replace_skill(path, name)` | 把一个模型步骤换成另一个技能 | 新技能读的字段有来源 |
| `wrap_loop(path, k, stop)` | 把一个节点包进循环 | 不超调用数上限；循环体里有写 plan 的步骤 |
| `wrap_fanout(path, n)` | 把一个节点包进分叉 | 同上 |
| `add_branch(path, check, label, child)` | 在某个位置加一个分支，或给已有分支加一条支路 | 检查需要的字段有来源；该标签还没有支路 |
| `set_param(path, key, value)` | 改循环次数、停止规则或分叉数 | 取值来自固定的小集合：k ∈ {2, 3, 5}，n ∈ {2, 3, 5} |
| `unwrap(path)` | 去掉一层循环或分叉，保留里面的内容 | 无 |

```python
def legal_edits(wf: Workflow, limits: Limits) -> list[Edit]: ...
def apply_edit(wf: Workflow, e: Edit | Fragment) -> Workflow: ...
def worst_case_calls(wf: Workflow) -> int: ...      # 用于调用数上限
```

### 5.2 片段

片段是一小棵预先写好的子树，作为一步插入。它对应第一版的宏改动，也是以后 playbook 里存的东西。

| 片段 | 内容 | 针对的问题 |
| --- | --- | --- |
| `repair_loop` | `loop(k=3, stop=legal)[validate, repair]` | 计划不合法 |
| `sim_improve` | `seq[simulate, improve]` | 计划合法但利润低 |
| `stats_first` | 在起草之前插入 `crop_stats` | 起草时作物选择差 |
| `multi_draft` | 把起草包进 `fanout(n=3)` | 单次起草质量波动大 |
| `split_by_status` | `branch(status){illegal: repair_loop, legal_low: sim_improve}` | 两类问题同时存在 |

### 5.3 每一步评估哪些候选

思考模式下每次评估都很贵，每一步只评估 3 个候选，名额固定分配：

- 1 个片段（由策略按当前残差挑）；
- 1 个控制结构类的原子编辑（`wrap_loop`、`wrap_fanout`、`add_branch`、`set_param`）；
- 1 个其余的原子编辑。

手写策略 f0 的规则改成按残差的来源排序：高权重题里不合法的占多数时，修复类的改动优先；合法但低分的占多数时，`simulate` 和 `improve` 类的改动优先；其余按历史平均增益排序。

## 6. Boosting 怎么搬到工作流上

两种 boosting 都保留，变化在于它们现在长的是同一张流程图：串联式往末尾接一段，分支式给某类题加一条支路。每一轮的产物都是一个完整的工作流，不再是“一组小团队”。

|  | 串联式（GBDT） | 分支式（AdaBoost） |
| --- | --- | --- |
| 每轮加什么 | 在根节点末尾接一段新流程 | 在末尾的分支块里，给一个标签加一条支路 |
| 新的一段拿到什么 | 前面所有阶段跑完后的状态 | 同左，但只有标签匹配的题会进来 |
| 残差体现在哪 | 输入的状态本身 | 题目权重，按标签汇总 |
| 推理成本 | 每道题跑完所有阶段 | 每道题只跑自己那条支路 |

分支式解决了第一版 AdaBoost 的两个问题：推理时不用把每个成员都跑一遍，路由也不再需要另外学，它就是分支用的那个检查。

### 6.1 长一段流程（两种模式共用）

沿用现有的 `grow` 循环，只是对象从子图换成片段：

```python
def grow_fragment(seed: Block, items, base_wf, attach, fitness, proposer, policy, cfg) -> Block:
    frag, fit = seed, fitness(attach(base_wf, seed), items)
    for step in range(cfg.max_steps):
        cands = proposer.propose(frag, ctx)              # 第 5.3 节：3 个名额
        ranked = policy.rank(frag, cands, obs)[: cfg.top_k]
        scored = [(fitness(attach(base_wf, apply_edit(frag, e)), items), e) for e in ranked]
        log_edit_outcomes(frag, scored)
        best_fit, best_e = max(scored, key=lambda t: t[0])
        if best_fit - fit <= cfg.min_gain: break
        frag, fit = apply_edit(frag, best_e), best_fit
    return frag
```

`attach` 决定新片段接在哪：串联式接在末尾，分支式接在某个标签下面。前面已接受的部分冻结，它们的输出状态从缓存读取，所以评估一个候选只花新片段的钱。

### 6.2 串联式

```python
def fit_serial(train, heldout, cfg) -> Workflow:
    wf = seq(draft_step())                                  # 第 0 阶段：只起草
    for m in range(cfg.rounds):
        active = [i for i in train if g(wf, i) < 1.0]
        frag = grow_fragment(empty_seq(), active, wf, attach_tail, fit_gain, proposer, policy, cfg)
        cand = attach_tail(wf, frag)
        if lower_ci(paired_gain(cand, wf, heldout)) <= cfg.accept_margin:
            if (stall := stall + 1) >= cfg.patience: break
            continue
        wf, stall = cand, 0
    return wf
```

适应度是新片段带来的平均 g 增量减去成本惩罚，和第一版相同。

### 6.3 分支式

```python
def fit_branch(train, heldout, cfg) -> Workflow:
    wf = seq(draft_step(), branch(check=cfg.check, cases={}))   # 起草后接一个空分支
    for m in range(cfg.rounds):
        r = residual_weights(g_all(wf, train), cfg)              # 做得差的题权重高
        mass = group_sum(r, by=lambda i: label(wf, i))           # 每个标签下的权重总和
        target = max(mass, key=mass.get)                         # 这一轮攻残差最集中的那一类
        items = [i for i in train if label(wf, i) == target]
        seed = wf.cases.get(target, empty_seq())                 # 已有支路就在它上面继续长
        frag = grow_fragment(seed, items, wf, attach_case(target), fit_weighted(r), proposer, policy, cfg)
        cand = attach_case(target)(wf, frag)
        ... # 留出验证与接受，同串联式
    return wf
```

- 第一版只用 `status` 这一个检查（不合法、合法但低分、合法且高分）。用 `error_type` 继续细分，作为一种编辑留给搜索；
- 权重的算法不变：`1 / (ε + g)`，封顶为均值的 5 倍；
- 一类题的权重少于总量的 10% 时不为它单独长支路，防止在几道题上过拟合。

### 6.4 接受准则

和第一版相同：候选工作流和当前工作流在留出题上逐题配对，g 增量的 90% 置信下界超过阈值才接受。报告用的留出分数另换一组随机种子重新跑，避免“碰巧过线”带来的偏高。

## 7. 预检实验：先确认有空间，再上搜索

在写任何搜索代码之前，先用三个手写工作流和“多次采样挑最好”比一次。手写流程都赢不了，搜索大概率也赢不了，就应该停下来改工具或换任务。第一版没做这一步，白花了两轮。

### 7.1 三个手写工作流

| 名字 | 结构 | 最多几次模型调用 | 检验什么 |
| --- | --- | --- | --- |
| F1 模拟后改进 | `draft → simulate → improve` | 2 | 模拟工具的反馈有没有用 |
| F2 先算再写 | `crop_stats → draft → branch(status){illegal: loop(k=2)[validate, repair], legal_low: [simulate, improve]}` | 3 | 计算工具加分流处理 |
| F3 分叉后改进 | `fanout(n=3)[draft] → simulate → improve` | 4 | 采样和改进能不能叠加 |

### 7.2 对照

| 对照 | 做法 |
| --- | --- |
| B1 多次起草挑最好 | `fanout(n)[draft]`，n 取 2、3、4，分别对齐三个流程的调用数 |
| B2 带工具信息的多次起草 | 同 B1，但起草前先跑 `crop_stats` |
| B3 修复循环 | `draft → loop(k, stop=best)[validate, repair]`，调用数对齐 |

B2 是为了分清功劳：如果 F2 赢了 B1 却赢不了 B2，说明起作用的只是那个工具，流程结构没有贡献。

### 7.3 规模与闸门

- 题目：难度 5 的 40 道校准题，和上一轮第一段相同，这样 B1 和 B3 的大部分样本可以从缓存复用；
- 每题跑 1 次，种子 0；
- 我的粗略估算是 500 到 700 次新调用，按每小时约 64 次算需要 8 到 11 小时。启动前要按实测吞吐重算。

**继续的条件：** 至少一个手写流程，相对同调用数的 B1 和 B2 中较强的那个，逐题配对差值的 90% 置信下界大于 0。

**不满足时：** 停下来报告。按结果选一条路：

| 看到的结果 | 说明 | 下一步 |
| --- | --- | --- |
| F1 比单次起草好，但不如同花费的 B1 | 改进一次不如重写一次 | 模拟工具的输出可能不够具体，改工具后重测 |
| 所有流程都和 B1 打平 | 这个任务上流程没有空间 | 换任务，代码不动 |
| B2 明显好于 B1，流程不比 B2 好 | 工具有用，结构没用 | 把 `crop_stats` 当作默认配置，重新找结构上的空间 |

## 8. 基线、指标与消融

所有结论都以“相同花费下相对多次采样的配对差值”来表述。赢不了它，其余数字都不构成证据。

### 8.1 基线

| 基线 | 做法 | 花费对齐方式 |
| --- | --- | --- |
| 单次起草 | `draft` | — |
| 多次起草挑最好 | `fanout(n)[draft]` | n 向下取整到不超过我们的实际 token |
| 带工具信息的多次起草 | `crop_stats → fanout(n)[draft]` | 同上 |
| 修复循环 | `draft → loop(k, stop=best)[validate, repair]` | k 向下取整 |
| 最好的手写流程 | 预检里得分最高的那个 | 按实际 token 报告 |

### 8.2 指标

- **主指标：** 测试题上的平均 g，以及相对每个基线的逐题配对差值和 90% 区间；
- **拆解：** 合法率；合法计划的平均分（不截断上限）；增益里有多少来自“不合法变合法”，多少来自“合法的分数变高”；
- **花费：** 每题的模型 token（含思考）、模型调用数、打分次数、工具调用数；
- **流程本身：** 最终工作流的结构，每条分支实际被多少题走到，每个循环实际跑了几次。

### 8.3 消融

这三项对应要论证的三件事，按顺序做，前一项有正结果才做下一项。

| 消融 | 对比 | 回答的问题 |
| --- | --- | --- |
| 去掉残差 | 分支式里把权重固定为均匀，目标标签随机选 | 按残差决定补哪里，有没有用 |
| 去掉分阶段 | 同样的编辑集合和评估预算，不冻结、不分轮，直接对整个工作流做贪心搜索 | 一段一段地长，是否比整体搜索好 |
| 去掉工具 | 编辑集合里去掉 `simulate` 和 `crop_stats` | 提升来自结构还是来自工具 |

### 8.4 题目划分

| 集合 | 数量 | 用途 |
| --- | --- | --- |
| 训练 | 12（算力允许时 24） | 算残差、评估候选 |
| 留出 | 30 | 每轮决定是否接受 |
| 测试 | 40 | 报告结果，就是预检用的那 40 道 |
| 终测 | 已生成，未启用 | 整个项目只跑一次 |

测试题在预检里已经被看过。所以预检之后不能再根据测试题上的数字改技能提示词、工具输出或任何超参数；要改就在训练题上看。

## 9. 代码改动与任务单

新代码放在一个新的 `workflow/` 包里，评估、缓存、报告和 boosting 的外层循环尽量复用。旧的角色子图代码不删，旧配置必须继续逐行复现已存结果。

### 9.1 模块对应

| 模块 | 改动 | 说明 |
| --- | --- | --- |
| `workflow/model.py` | 新增 | `State`、`Step`、`Block`、合法性检查、最坏调用数 |
| `workflow/exec.py` | 新增 | 第 4 节的解释器 |
| `workflow/edits.py` | 新增 | 原子编辑、片段、`legal_edits` |
| `workflow/tools.py` | 新增 | `simulate`、`crop_stats`、两个检查；`validate` 包装现有打分器 |
| `roles/` | 小改 | 旧提示词注册为技能；新增 `improve` |
| `evalx/` | 小改 | 评估入口接受工作流；缓存键加入步骤路径和循环、分叉序号；成本加工具调用数 |
| `search/grow.py` | 小改 | 对象泛化为“可编辑的东西”，子图和片段都满足同一个接口 |
| `search/proposers.py` | 新增一个 | `WorkflowProposer`，按第 5.3 节分配三个名额 |
| `policy/` | 新增一个 | 按残差来源排序的 f0 |
| `boost/` | 新增两个 | `serial.py` 和 `branch.py`，第 6 节 |
| `baselines.py` | 改写 | 所有基线改成用工作流表示来跑，保证和我们的方法走同一个执行器 |
| `scripts/report.py` | 小改 | 加第 8.2 节的拆解和流程结构输出 |

基线和我们的方法走同一个执行器，这一点很重要。前几轮的好几个问题（提示词不一致、预算口径不一致、缓存空转）都出在基线和方法用了两套代码路径上。

### 9.2 任务单

| 任务 | 内容 | 验收 |
| --- | --- | --- |
| W1 | `model.py` 与 `exec.py` | 四种控制块的执行顺序和提前停止；成本只计实际执行的步骤；用工作流写出的修复循环和多次采样，从缓存重放时与已存的基线结果逐题一致 |
| W2 | `tools.py` 与 `improve` 技能 | `simulate` 算出的最终利润与打分器在 40 道题的已缓存计划上完全一致；`crop_stats` 有手算样例；工具模块不导入参考解代码 |
| W3 | 所有基线改用工作流表示；预检脚本 | 三个手写流程和三个对照能在假模型上跑通；花费对齐向下取整 |
| W4 | `edits.py`、`WorkflowProposer`、新 f0 | 每种编辑的合法性测试；任何编辑序列之后工作流仍合法（随机测试）；调用数上限从不被突破 |
| W5 | `boost/serial.py` | 玩具题上每接受一段，训练 g 单调不降；冻结部分不重复计费 |
| W6 | `boost/branch.py` | 权重按标签汇总正确；推理时每题只走一条支路；权重不足 10% 的标签不长支路 |
| W7 | 三项消融的开关 | 关闭开关时结果与 W5、W6 一致 |

W1 到 W3 做完就能跑预检。W4 之后的任务可以在预检跑的时候写，但预检过关之前不启动搜索实验。

### 9.3 沿用的约束

- 模型任务一次只跑一个，并发由客户端限制；
- 不查找、不读取任何凭据；
- 不跑终测，不按测试题调参；
- 每处偏离文档的实现都记入 `docs/IMPLEMENTATION_NOTES.md`；
- 汇报时每条结论注明是实际跑出来的还是推断。

## 10. 里程碑、验收标准与待确认事项

六个里程碑依次推进。S1 是闸门：预检不过，后面的搜索实验不启动。

| 里程碑 | 内容 | 任务单 | 通过条件 |
| --- | --- | --- | --- |
| S0 表示与工具 | 工作流的数据结构、执行器、工具、基线改写 | W1–W3 | 用工作流写出的修复循环和多次采样，从缓存重放与已存结果一致；旧配置仍能复现 |
| S1 预检 | 三个手写流程对三个对照，40 题 | — | 第 7.3 节的继续条件 |
| S2 串联式搜索 | 编辑、提议器、串联式 boosting | W4、W5 | 测试题上不低于最好的手写流程，并在相同花费下赢过多次采样 |
| S3 分支式搜索 | 分支式 boosting | W6 | 相同花费下不低于串联式；推理花费更低或分数更高，至少占一样 |
| S4 消融 | 去掉残差、去掉分阶段、去掉工具 | W7 | 三项各给出带区间的差值，结果如实报告 |
| S5 playbook | 把被接受的片段按问题类型存库，跨难度复用 | 另立 | 在难度 4 上学到的片段库，用到难度 5 时搜索更快达到同样的留出分数 |

### 10.1 算力预估

下面是我的粗略估算，按单卡、每小时约 64 次调用计。每个阶段启动前要按实测吞吐重算。

| 阶段 | 调用数 | 时间 |
| --- | --- | --- |
| S1 预检 | 500–700 | 8–11 小时 |
| S2 一个种子 | 1200–2200 | 20–35 小时 |
| S3 一个种子 | 与 S2 相近 | 20–35 小时 |
| S4 三项消融 | 每项与 S2 相近 | 每项 20–35 小时 |

S2 的数字参考了上一轮第二段启动前的估算。要跑 3 个种子或做完全部消融，单卡需要两周以上；多一张卡可以按种子并行。

### 10.2 待确认

1. **上一轮第二段的结果还没看到。** 它是第一版（角色子图加循环和宏改动）在新设置下的唯一结果。如果它已经赢过多次采样，这份文档的优先级要重新考虑。
2. **`improve` 技能的提示词还没写。** 它是这一版成败的关键之一。写完后先在几道训练题上人工看输出，再跑预检。
3. **`simulate` 输出什么最有用还不确定。** 第 3.2 节列的是初版。预检如果显示 F1 不如重写一次，首先要改的就是它。
4. **分支用的分数阈值。** `status` 检查里“低分”和“高分”的分界线先取 0.8，在训练题上看分布后再定。
5. **思考模式下每次调用 10 到 20 分钟。** 如果能拿到一个 API 模型的密钥，算力约束基本消失，但需要重新校准难度。

### 10.3 这一版要证明的事，按顺序

1. 一次胜利：相同花费下，搜索出的工作流赢过多次采样（S2）；
2. 胜利来自 boosting：按残差、分阶段地长，比不按残差、整体搜索更好（S4）；
3. 学到的东西能复用：片段库跨难度迁移（S5）。

第 1 条拿不到，后两条无从谈起。拿到之后，再把同一套代码用到需要分工的任务上，论证单智能体和多智能体都适用。
