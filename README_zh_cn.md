# nnc-joint-solver

Neural Network Compiler 的联合求解器，用于解决神经网络推理在硬件加速器（NPU）上的 **tiling 选择、调度排布与 SRAM 放置** 联合优化问题。

## 问题定义

给定一个 `JointProblem`，求解器需要同时完成以下决策：

1. **Recipe 选择** — 每个计算区域（region）有多个候选 tiling 方案（recipe），求解器为每个 region 恰好选择一个 recipe，且相邻 region 的 recipe 必须满足边界兼容性约束（`boundary_constraints`）。
2. **Action 调度** — 所有 action（compute、DMA in/out、spill、reload）必须分配起始时间，满足依赖边约束和资源互斥约束（同一资源上的 action 不可在时间上重叠）。
3. **SRAM 驻留窗口** — 为每个需要驻留在 SRAM 中的 value 确定驻留时间区间，保证读取时 value 已在 SRAM 中。
4. **SRAM 偏移分配** — 为所有 SRAM item（temp buffer、transfer buffer、resident window）分配内存偏移，确保同一时刻存活的 item 不发生空间重叠，且总用量不超过 `sram_capacity_bytes`。

**优化目标：最小化 makespan**（最后一个 action 的结束时间）。

## 安装与使用

### 作为 Python 库使用

```python
from nnc_joint_solver import (
    V0JointScheduleSolver,
    V1JointScheduleSolver,
    LatestJointScheduleSolver,
    JointScheduleSolver,
    JointProblem,
    JointSolution,
    JointFailure,
)

problem = JointProblem.from_json(payload)
solver = LatestJointScheduleSolver()  # 当前指向 V1
result = solver.solve(problem)

if isinstance(result, JointSolution):
    print(f"makespan = {result.objective_value}")
else:
    print(f"failed: {result.status} / {result.error_category}")
```

### 作为 CLI 工具使用

CLI 通过 stdin 接收 `JointProblem` JSON，通过 stdout 输出 `JointSolution` 或 `JointFailure` JSON：

```bash
# 使用默认最新版本（V1）
echo '{"schema_version": "joint_tiling_schedule_problem_v1", ...}' | nnc-joint-solver

# 指定 V0
echo '...' | nnc-joint-solver --solver-version v0
```

### 子进程调用

通过 `CliJointScheduleSolver` 以子进程方式调用：

```python
from nnc_joint_solver import CliJointScheduleSolver

transport = CliJointScheduleSolver(
    command=["nnc-joint-solver"],
    timeout_seconds=10.0,
)
result = transport.solve(problem)  # JointSolution | JointFailure
```

## API 接口

### 抽象接口

所有求解器实现 `JointScheduleSolver` 抽象基类：

```python
class JointScheduleSolver(ABC):
    @abstractmethod
    def solve(self, problem: JointProblem) -> JointSolution | JointFailure:
        ...
```

**输入** `JointProblem` — 联合优化问题的完整描述。
**输出** discriminated union：成功返回 `JointSolution`，失败返回 `JointFailure`。

### 核心数据模型

所有数据模型定义在 `ir/joint_tiling_schedule.py`，均为 frozen dataclass，支持 `to_json()` / `from_json()` 双向 JSON 序列化。

#### 枚举类型

| 枚举 | 值 | 说明 |
|------|----|------|
| `JointRegionKind` | `single_op`, `fused_group` | 计算区域类型 |
| `JointValueTier` | `unmaterialized`, `input`, `const`, `slow`, `sram` | value 的存储层级 |
| `JointSramItemKind` | `temp_interval`, `transfer_buffer`, `resident_window` | SRAM item 类型 |
| `JointActionKind` | `compute`, `dma_in`, `dma_out`, `spill`, `reload` | action 类型 |
| `JointDependencyEdgeKind` | `data`, `order` | 依赖边类型 |
| `JointResourceKind` | `DMA`, `MATMUL`, `SHAPE`, `OTHER` | 硬件资源类型 |
| `JointFailureStatus` | `infeasible`, `timeout`, `invalid_problem`, `error` | 失败状态码 |
| `JointFailureCategory` | 8 种（`dependency_violation`, `resource_overlap`, `sram_capacity_exceeded` 等） | 详细错误分类 |

#### Problem 侧

```python
@dataclass(frozen=True)
class JointProblem:
    schema_version: str              # "joint_tiling_schedule_problem_v1"
    regions: tuple[JointRegion, ...]
    recipes: tuple[JointRecipe, ...]
    values: tuple[JointValue, ...]
    actions: tuple[JointAction, ...]
    boundary_constraints: tuple[JointBoundaryConstraint, ...]
    dependency_edges: tuple[JointDependencyEdge, ...]
    resources: tuple[JointResource, ...]        # 必须包含 DMA/MATMUL/SHAPE/OTHER
    sram_capacity_bytes: int                    # SRAM 容量上限
    sram_items: tuple[JointSramItem, ...]       # 固定 SRAM 分配需求
    default_alignment_bytes: int                # 默认对齐字节数
    objective: str                              # "minimize_makespan"
```

- `JointRegion` — 计算区域，包含 input/output value ID、前驱/后继 region。
- `JointRecipe` — region 的 tiling 方案，包含 tile_spec、layout_spec、激活的 action ID、value footprint、cost 参数。
- `JointValue` — 张量/值，包含 size、初始/最终 tier、producer/consumer、是否可 spill、SRAM 驻留约束。
- `JointAction` — 操作（compute/DMA/spill/reload），包含 resource kind、duration、launch overhead、读写 value ID、temp bytes。
- `JointBoundaryConstraint` — 相邻 region 间的边界约束：哪些 recipe 对是兼容的。
- `JointDependencyEdge` — action 间的有向依赖边。

#### Solution 侧

```python
@dataclass(frozen=True)
class JointSolution:
    schema_version: str                          # "joint_tiling_schedule_solution_v1"
    selected_recipes: tuple[JointSelectedRecipe, ...]    # 每个 region 选择的 recipe
    scheduled_actions: tuple[JointScheduledAction, ...]  # 每个 action 的起始时间
    residency_windows: tuple[JointResidencyWindow, ...]  # value 的 SRAM 驻留窗口
    objective_value: int                         # makespan
    generated_sram_items: tuple[JointSramItem, ...]      # 由驻留窗口生成的 SRAM item
    sram_allocations: tuple[JointSramAllocation, ...]    # 每个 SRAM item 的偏移
    diagnostics: object                          # 可选诊断信息
```

#### Failure 侧

```python
@dataclass(frozen=True)
class JointFailure:
    schema_version: str                          # "joint_tiling_schedule_failure_v1"
    status: JointFailureStatus                   # 失败状态
    error_category: JointFailureCategory         # 错误分类
    diagnostics: object                          # 诊断详情
```

### CLI 传输层

`CliJointScheduleSolver` 封装了子进程 JSON 通信协议：

- 将 `JointProblem` 序列化为 JSON 写入子进程 stdin
- 从子进程 stdout 读取 JSON，根据 `schema_version` 反序列化为 `JointSolution` 或 `JointFailure`
- 处理超时（`DEFAULT_SOLVER_TIMEOUT_SECONDS = 5.0`）、进程错误、格式错误

### Schema 版本

Wire protocol 使用显式版本字符串：

| 场景 | schema_version |
|------|---------------|
| 输入 | `joint_tiling_schedule_problem_v1` |
| 成功输出 | `joint_tiling_schedule_solution_v1` |
| 失败输出 | `joint_tiling_schedule_failure_v1` |

### 求解器版本

| 类名 | 说明 |
|------|------|
| `V0JointScheduleSolver` | 基线求解器，每个 region 选第一个 recipe |
| `V1JointScheduleSolver` | 启发式求解器，beam search + 局部搜索 |
| `LatestJointScheduleSolver` | 指向当前最新版本（V1） |
| `BaselineJointScheduleSolver` | `V0JointScheduleSolver` 的别名 |

## 开发新算法

### 1. 创建求解器实现

在 `src/nnc_joint_solver/` 下新建版本目录，实现 `JointScheduleSolver` 接口：

```
src/nnc_joint_solver/v2/
  __init__.py      # 导出 V2JointScheduleSolver
  solver.py        # 实现文件
```

`v2/solver.py` 的最小骨架：

```python
from nnc_joint_solver.base import JointScheduleSolver
from nnc_joint_solver.ir.joint_tiling_schedule import (
    JointProblem,
    JointSolution,
    JointFailure,
)


class V2JointScheduleSolver(JointScheduleSolver):
    def solve(self, problem: JointProblem) -> JointSolution | JointFailure:
        # 1. (可选) 校验问题
        # 2. 选择 recipe
        # 3. 调度 action
        # 4. 计算 SRAM 驻留窗口和分配
        # 5. 构造并返回 JointSolution 或 JointFailure
        ...
```

### 2. 利用共享工具函数

`solve_utils.py` 提供了可复用的调度基础设施：

| 函数 | 用途 |
|------|------|
| `solve_recipe_selection()` | 完整流程：构建选中 recipe → 确定 active action → 拓扑排序 → 调度 → 驻留窗口 → SRAM 分配 → 校验 → 返回解 |
| `topological_action_order()` | 基于依赖边的 Kahn 拓扑排序，支持自定义优先级打破平局 |
| `schedule_ordered_actions()` | 按拓扑序分配起始时间，处理资源互斥和 JIT DMA 延迟启发式 |
| `_minimal_residency_windows()` | 计算最小的 SRAM 驻留时间窗口 |
| `_pack_sram_allocations()` | First-fit decreasing 区间分配算法 |

大多数新算法只需关注 **recipe 选择策略**，然后调用 `solve_recipe_selection()` 即可获得完整解。例如：

```python
from nnc_joint_solver.solve_utils import solve_recipe_selection

class V2JointScheduleSolver(JointScheduleSolver):
    def solve(self, problem: JointProblem) -> JointSolution | JointFailure:
        # 自定义 recipe 选择逻辑
        selected = {...}  # region_id -> recipe_id
        return solve_recipe_selection(problem, selected)
```

### 3. 注册求解器

更新以下文件使新求解器可用：

**`src/nnc_joint_solver/v2/__init__.py`：**

```python
from nnc_joint_solver.v2.solver import V2JointScheduleSolver

__all__ = ["V2JointScheduleSolver"]
```

**`src/nnc_joint_solver/solver.py` — 添加导出和更新 latest：**

```python
from nnc_joint_solver.v2.solver import V2JointScheduleSolver

LatestJointScheduleSolver = V2JointScheduleSolver  # 更新指向
```

**`src/nnc_joint_solver/cli.py` — 添加 CLI 版本选项：**

```python
parser.add_argument("--solver-version", choices=("v0", "v1", "v2"), default="v2")
```

### 4. 校验

`validation.py` 提供完整的约束检查：

- `validate_joint_problem()` — 校验问题结构完整性
- `validate_joint_solution()` — 校验解满足所有约束（recipe 覆盖、action 调度合法性、资源无重叠、依赖边满足、边界兼容性、SRAM 容量和重叠检查）

建议在求解器返回前调用校验，确保解合法。
