# cuEquivariance 核心架构梳理与可执行验证笔记 (`learning` branch)

本目录面向分子模拟（MD 力场、极化多极矩力场 PFF、UMA/eSCN 大模型推理与 BF16/FP8 混合精度训练）梳理 `cuEquivariance` 的三层核心架构，并提供三个**零外部依赖开销、可直接运行并通过数值断言**的拆解脚本。

## 三层核心架构速览

```text
Layer 3: Group Theory & Equivariant Descriptors (cuequivariance/group_theory/)
  ├── Lie 代数与群表示: Rep, Irrep (SU2 -> SO3 -> O3), Clebsch-Gordan (sqrtQarray 精确代数)
  ├── 布局与张量基: Irreps, IrrepsAndLayout (ir_mul vs mul_ir), reduced_tensor_product_basis
  └── 等变描述符工厂: spherical_harmonics, yxy/yx/xy_rotation, escn_tp, channelwise_tp, symmetric_contraction
          │  (.polynomial 剥离群论语义，降维为纯分块多线性代数 IR)
          ▼
Layer 2: Segmented Polynomial Compiler IR (cuequivariance/segmented_polynomials/)
  ├── SegmentedOperand: 分段形状列表 [(d1, d2), ...] 定义扁平张量的分段视图
  ├── SegmentedTensorProduct (STP): 稀疏路径 Paths(indices, coefficients) + Einstein 下标
  ├── 代数重写: flatten_coefficient_modes, squeeze_modes, split_mode, symmetrize_operands, dot
  └── 符号自动微分闭包: jvp (前向切向), transpose (线性伴随), backward (任意高阶导数闭合于同一 IR!)
          │
          ▼
Layer 1: Runtime Backends (cuequivariance_torch/primitives/)
  ├── uniform_1d: 单一均匀 1D 模式 (如 ir_mul 下所有分段均为 (u,)) -> JIT CUDA Warp Kernel + 融合 Gather/Scatter
  ├── fused_tp: 3/4 操作数通用张量积 CUDA Kernel
  ├── indexed_linear: 按元素种类排序索引的稀疏权重线性层 Kernel
  └── naive: 基于 torch.fx + einsum 的全平台参考后端（CPU / 任意分段形状回退路径）
```

## 脚本清单与运行方式

在仓库根目录使用带有 `torch`、`numpy`、`scipy`、`sympy` 的 Python 环境运行：

```powershell
$py = "C:\Users\yanig\miniconda3\envs\ase\python.exe"
$env:PYTHONPATH = "$PWD\cuequivariance;$PWD\cuequivariance_torch"

& $py learning_notes/01_segmented_polynomials_ir.py
& $py learning_notes/02_spherical_harmonics_audit.py
& $py learning_notes/03_wigner_and_escn_audit.py
```

| 脚本 | 核心拆解内容 | 科研视角关键发现 |
| :--- | :--- | :--- |
| `01_segmented_polynomials_ir.py` | `STP` 稀疏路径、`ir_mul` vs `mul_ir` 内存连续性、`split_mode` 均匀化、一阶/二阶符号微分闭包 | 异构通道 `"128x0e+64x1o+32x2e"` 经 `split_mode("u", 32)` 可在不改内存布局的前提下满足 `uniform_1d` 约束；`backward` 对任意阶导数严格闭合 |
| `02_spherical_harmonics_audit.py` | `sympy` 递归 CG 构造、对称齐次多项式压缩、$O(3)$ 旋转+反演等变门禁、零向量一阶/二阶导数审计 | 标准 `F.normalize` 在 $\mathbf{r}=\mathbf{0}$（如 Verlet padding 槽位）处一阶梯度达 $1.73\times 10^{12}$（BF16 溢出），二阶梯度直接产生 `NaN`；双重 `torch.where` 安全归一化可彻底消除 |
| `03_wigner_and_escn_audit.py` | 复数 $SU(2) \to$ 实数 $SO(3)$ 基变换、Euler `yxy` vs 边对齐 `yx`/`xy` 路径复杂度、`escn_tp` vs `escn_tp_compact`、BF16 极点梯度审计 | 2 角 `yx_rotation` 比 3 角 `yxy_rotation` 减少 **39%~45%** FLOPs；`acos(y)` 在 BF16 下距 $\pm\hat{\mathbf{y}}$ 约 $7.2^\circ$ 锥区内触发 `NaN` 梯度，改用 `atan2(r_xz, y)` 可根治 |
