# cuEquivariance 核心架构梳理与可执行验证笔记 (`learning` branch)

本目录面向分子模拟（凝聚相电解液 MD 力场、极化多极矩力场 PFF、MACE/NequIP/UMA/eSCN 多体势函数，以及 Evoformer/Pairformer 生物分子结构预测与 A100/H100 混合精度加速），系统拆解 `cuEquivariance` 的三层核心架构与三角注意力原语，并提供 **7 个零外部数据依赖、纯本地可直接运行并通过严格数值断言（零警告）** 的审计脚本。

## 三层核心架构与原语全景

```text
Layer 3: Group Theory, Equivariant Descriptors & Model Layers
  ├── Lie 代数与群表示: Rep, Irrep (SU2 -> SO3 -> O3), Clebsch-Gordan (sqrtQarray 精确代数)
  ├── 布局与对称张量基: Irreps, IrrepsAndLayout (ir_mul vs mul_ir), reduced_symmetric_tensor_product_basis
  ├── 等变描述符工厂: spherical_harmonics, yxy/yx/xy_rotation, escn_tp(_compact), channelwise_tp, symmetric_contraction
  └── 领域模型层: MACE SymmetricContraction (P 投影压缩), NequIP FullyConnectedTensorProductConv (预聚合拆分 MLP)
          │  (.polynomial 剥离群论语义，降维为纯分块多线性代数 IR)
          ▼
Layer 2: Segmented Polynomial Compiler IR (cuequivariance/segmented_polynomials/)
  ├── SegmentedOperand & STP: 分段视图 [(d1, d2), ...] + 稀疏路径 Paths(indices, coefficients) + Einstein 下标
  ├── 代数重写与多体堆叠: flatten_coefficient_modes, squeeze_modes, split_mode("u", 32), stack (MACE nu=1,2,3 融合)
  ├── 静态性能建模: poly.flop(batch) 与 poly.memory(batch_sizes) 精确算术强度 (Arithmetic Intensity) 建模
  └── 符号自动微分闭包: jvp (前向切向), transpose (线性伴随), backward (任意高阶导数严格闭合于同一 IR)
          │
          ▼
Layer 1: Runtime Backends & Biomolecular Primitives (cuequivariance_torch/primitives/)
  ├── uniform_1d: 单一均匀 1D 模式 (ir_mul 连续通道 u % 32 == 0) -> JIT CUDA Warp Kernel + 融合 Gather/Scatter
  ├── indexed_linear: SUBDICT 12 种规范下标 + 物种排序 bincount 分组 GEMM (消除 W[species] 显存爆炸)
  ├── fused_tp: 3/4 操作数通用张量积 CUDA Kernel
  ├── triangle: triangle_multiplicative_update, triangle_attention (kv_lengths 快路径), attention_pair_bias (z_proj 缓存)
  └── naive: 基于 torch.fx + opt_einsum 的全平台参考后端（CPU / 任意分段形状精确回退路径）
```

## 脚本清单与运行方式

在仓库根目录使用带有 `torch`、`numpy`、`scipy`、`sympy` 的 Python 环境运行：

```powershell
$py = "C:\Users\yanig\miniconda3\envs\ase\python.exe"
$env:PYTHONPATH = "$PWD\cuequivariance;$PWD\cuequivariance_torch"

& $py learning_notes/01_segmented_polynomials_ir.py
& $py learning_notes/02_spherical_harmonics_audit.py
& $py learning_notes/03_wigner_and_escn_audit.py
& $py learning_notes/04_mace_architecture_audit.py
& $py learning_notes/05_nequip_conv_and_indexed_linear.py
& $py learning_notes/06_triangle_attention_and_pair_bias.py
& $py learning_notes/07_gpu_roofline_a100_h100_audit.py
```

| 脚本 | 核心拆解内容 | 科研与硬件审计关键发现 |
| :--- | :--- | :--- |
| `01_segmented_polynomials_ir.py` | `STP` 稀疏路径、`ir_mul` vs `mul_ir` 内存连续性、`split_mode` 均匀化、一阶/二阶符号微分闭包 | 异构通道 `"128x0e+64x1o+32x2e"` 经 `split_mode("u", 32)` 零转置满足 `uniform_1d` 约束；`backward` 对 MD 力与二阶导严格闭合 |
| `02_spherical_harmonics_audit.py` | `sympy` 递归 CG 构造、对称齐次多项式压缩、$O(3)$ 旋转+反演等变门禁、零向量一阶/二阶导数审计 | 标准 `F.normalize` 在 $\mathbf{r}=\mathbf{0}$（Verlet padding 槽位）处一阶梯度达 $1.73\times 10^{12}$（BF16 溢出），二阶梯度产生 `NaN`；双重 `torch.where` 可彻底消除 |
| `03_wigner_and_escn_audit.py` | 复数 $SU(2) \to$ 实数 $SO(3)$ 基变换、Euler `yxy` vs 边对齐 `yx`/`xy` 复杂度、`escn_tp_compact`、BF16 极点梯度审计 | 2 角 `yx_rotation` 比 3 角 `yxy_rotation` 减少 **36.7%~47.8%** FLOPs；`acos(y)` 在 BF16 极点附近触发 `NaN` 梯度，改用 `atan2(r_xz, y)` 根治 |
| `04_mace_architecture_audit.py` | MACE 两阶段管线（Stage A 融合 Gather/Scatter 构建 $A_{i,c\ell m}$ + Stage B $\nu=1,2,3$ 四体 `SymmetricContraction`）与 $P$ 矩阵压缩 | 原始 MACE 递归 CG `U_matrix_real` 存在对称幂冗余：`SO(3)` 权重从 `62 -> 18`（**3.44x 压缩**），`O(3)` 从 `41 -> 17`（**2.41x 压缩**）；离线折叠 $W_{\text{cue}} = P^T W_{\text{mace}}$ 零误差等价且全管线满足 $O(3)$ 旋转+反演等变（相对误差 $3.35\times 10^{-15}$） |
| `05_nequip_conv_and_indexed_linear.py` | NequIP 边 MLP 预聚合权重拆分 `(w_edge, w_src, w_dst)`；`SegmentedPolynomialIndexedLinear` (`SUBDICT`) 与物种排序 `bincount` 分组 GEMM | 在配位数 $z=30$ 的电解液 MD 图（$N=8192$）上，首层边 MLP 拆分为 $O(N)$ 节点 GEMM 带来 **10.47x FLOP 加速（节省 90.4%）** 并省去 `503.3 MB` 标量拼接显存；`IndexedLinear` 避免 `W[species]` 展开带来的 **2731x 显存膨胀**（`1.12 MB` L2 常驻表 vs `3072 MB`） |
| `06_triangle_attention_and_pair_bias.py` | 纯 PyTorch 精确参考实现：`triangle_multiplicative_update`（双向几何闭包与 $S_N$ 置换等变）、`triangle_attention`（`kv_lengths` 快路径）、`attention_pair_bias`（`z_proj` 缓存） | 证明三角更新严格满足 $P\cdot\text{TriMul}(Z)\cdot P^T = \text{TriMul}(PZP^T)$ 并通过二跳桥接精确恢复缺失边三维位移；揭示全掩码空行 `kv_lengths=0` 精确归零而朴素 `-1e9` softmax 泄漏均值；48 层 `M=16` 扩散采样复用 `is_cached_z_proj` 减少 **107.8x** Pair HBM 读取流量（`115.96 GB -> 1.08 GB`） |
| `07_gpu_roofline_a100_h100_audit.py` | **A100-SXM4 vs H100-SXM5** 分析型 Roofline 模型（6 类典型算子）、`u % 32 == 0` Warp/Cache 对齐审计、`float8_e4m3fn` $SO(3)$ 等变性审计 | 球谐、CG 张量积、边旋转与 MACE 收缩算术强度（`0.8~86.5 FLOP/B`）严格低于 A100/H100 BF16 脊点（`153.0 / 295.2 FLOP/B`），属**显存带宽受限**（融合 Gather/Scatter 提升 **4.93x** AI）；高通道 `Linear` 与 `TriangleMulUpdate`（`329.1 FLOP/B`）跨入 **Tensor Core 算力受限区**（H100 FP8 达 **6.34x** 峰值加速）；逐分量 FP8 裁剪破坏 $SO(3)$ 对称性（误差 `7.05e-2`），范数分块缩放恢复精确包络对称性（`3.22e-16`），但 E4M3 3-bit 尾数底噪（`3.41e-2` vs BF16 `2.36e-3`）决定 MD 保守力场 $\ell\ge 1$ 特征宜保持 FP32/TF32/BF16 |

## A100 vs H100 硬件设计准则速查

1. **区分 SIMT 带宽受限算子与 Tensor Core 算力受限算子**：
   - 从 A100 到 H100，HBM 带宽仅提升 **1.64x**（`2.039 -> 3.35 TB/s`），而 BF16 Tensor Core 算力提升 **3.17x**（`312 -> 989 TF`），导致脊点（Ridge Point）右移近一倍（`153.0 -> 295.2 FLOP/B`）。
   - 稀疏 CG 张量积（`ChannelWiseTP`）与 `SphericalHarmonics` 算术强度远低于脊点，优化核心在于 **`uniform_1d` 融合邻居 Gather/Scatter**（消除 `O(E)` 边消息显存往返）与 **`ir_mul` 布局 + `u % 32 == 0` 对齐**（100% Warp 车道利用率 + 128B 缓存扇区对齐）。
   - 高通道块对角算子（`Linear` / `escn_tp_compact` / `indexed_linear`）及 $O(DN^3)$ `TriangleMultiplicativeUpdate` 跨越脊点，可充分释放 H100 TF32/BF16/FP8 Tensor Core 吞吐。
2. **混合精度与旋转等变守恒性**：
   - 对 $\ell \ge 1$ 不可约表示，严禁按独立磁量子数分量 $m$ 做非各向同性缩放或截断；若需动态缩放，必须以 $O(3)$ 不变欧氏范数 $\|x_{b,\ell,c}\|_2$ 为单位共享标量因子，且在需要能量-力严格守恒的 NVE/NVT 分子动力学中优先采用 **FP32 / TF32 / BF16**（将 FP8 限定于 $\ell=0$ 标量径向/读出 MLP 或非保守预筛选网络）。
