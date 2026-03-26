# SHAP 可解释性分析文档

本文档详细介绍了 `exp/pipeline/shap_explain.py` 中使用 SHAP（SHapley Additive exPlanations）进行模型可解释性分析的原理、功能与实现细节。

---

## 目录

1. [SHAP 原理介绍](#1-shap-原理介绍)
2. [功能介绍](#2-功能介绍)
3. [代码解读](#3-代码解读)
4. [使用示例](#4-使用示例)
5. [输出文件说明](#5-输出文件说明)

---

## 1. SHAP 原理介绍

### 1.1 SHAP 值的基本概念

SHAP（SHapley Additive exPlanations）是一种基于博弈论中 Shapley 值的模型解释方法。SHAP 值的核心思想是：**每个特征对模型预测的贡献可以用一个公平分配的方式来计算**。

对于给定的样本 $x$ 和模型 $f$，SHAP 值满足以下性质：

$$
f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i
$$

其中：
- $\phi_0$ 是基准值（expected value），表示模型在所有样本上的平均预测值
- $\phi_i$ 是特征 $i$ 的 SHAP 值，表示特征 $i$ 对当前预测的贡献
- $M$ 是特征数量

### 1.2 Shapley 值的定义

Shapley 值来自合作博弈论，用于公平分配合作收益。在机器学习中，将特征视为"玩家"，预测值视为"收益"，SHAP 值计算公式为：

$$
\phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} [f(S \cup \{i\}) - f(S)]
$$

其中：
- $F$ 是所有特征的集合
- $S$ 是不包含特征 $i$ 的特征子集
- $f(S)$ 表示仅使用特征子集 $S$ 时的模型预测值

### 1.3 TreeExplainer 的工作原理

对于树模型（如 LightGBM），SHAP 库提供了高效的 `TreeExplainer`，它利用树的递归结构快速计算 SHAP 值：

1. **路径遍历**：从根节点到叶节点，记录每条路径上的特征分裂
2. **条件期望**：利用训练数据在子节点中的分布，快速估计条件期望
3. **组合贡献**：将所有路径上的特征贡献相加，得到最终的 SHAP 值

**时间复杂度**：TreeExplainer 的时间复杂度为 $O(TLD)$，其中 $T$ 是树的数量，$L$ 是叶子节点数量，$D$ 是树的深度。这比标准的 SHAP 计算（需要遍历所有特征子集）快得多。

### 1.4 SHAP 值的可解释性

SHAP 值提供三个层面的解释：

1. **全局解释**：哪些特征最重要（通过平均绝对 SHAP 值）
2. **局部解释**：对于单个样本，哪些特征影响了预测结果
3. **交互解释**：特征之间的交互作用如何影响预测

---

## 2. 功能介绍

`run_lgbm_shap_explain` 函数实现了完整的 SHAP 可解释性分析流程，包含以下功能模块：

### 2.1 核心功能

#### 2.1.1 SHAP 值计算
- 使用 `TreeExplainer` 计算 SHAP 值
- 自动处理多分类情况（回归任务取第一个元素）
- 支持数据采样以降低计算成本

#### 2.1.2 特征重要性分析
- 计算平均绝对 SHAP 值（`mean_abs_shap`）：衡量特征的整体重要性
- 计算平均 SHAP 值（`mean_shap`）：衡量特征对预测的平均方向性影响
- 生成特征重要性排序 CSV 文件

#### 2.1.3 基准值（Expected Value）
- 保存模型的基准值，表示模型在训练集上的平均预测值
- 所有样本的 SHAP 值总和加上基准值应等于模型预测值

### 2.2 可视化功能

#### 2.2.1 Summary Plot（汇总图）

**Beeswarm Plot（蜂群图）**
- 显示每个特征的 SHAP 值分布
- 点颜色表示特征值的大小（红色=高值，蓝色=低值）
- 点的水平位置表示 SHAP 值（向右=正向贡献，向左=负向贡献）
- **用途**：快速识别哪些特征最重要，以及它们如何影响预测

**Bar Plot（条形图）**
- 显示特征的平均绝对 SHAP 值
- 按重要性降序排列
- **用途**：清晰展示特征重要性的相对排序

#### 2.2.2 Dependence Plot（依赖图）
- 显示单个特征对模型输出的影响
- X 轴：特征值
- Y 轴：该特征对应的 SHAP 值
- 颜色：自动检测与该特征交互最强的另一个特征
- **用途**：理解特征的非线性关系、阈值效应和特征交互

#### 2.2.3 Waterfall Plot（瀑布图）
- 展示单个样本从基准值到最终预测值的"路径"
- 每个条块表示一个特征的贡献
- 条块从下往上累积，最终达到预测值
- **用途**：详细解释单个预测结果，适合向非技术用户展示

#### 2.2.4 Decision Plot（决策图）
- 展示多个样本的决策路径
- 每条线代表一个样本，从基准值开始，随着特征逐个加入而累积变化
- 特征按重要性排序
- **用途**：比较不同样本的预测路径，识别关键决策特征

#### 2.2.5 Heatmap（热力图）
- 展示样本-特征矩阵的 SHAP 值
- 每一行是一个样本，每一列是一个特征
- 颜色深浅表示 SHAP 值的大小（红色=正向，蓝色=负向）
- 样本和特征可以按各种方式排序
- **用途**：发现样本簇、模式识别、时间序列分析

#### 2.2.6 Interaction Values（交互值）
- 计算特征对之间的交互 SHAP 值
- 生成交互矩阵热力图
- 识别最强的特征交互
- **注意**：计算成本较高，需要显式启用（`enable_interaction=True`）
- **用途**：发现特征间的协同或对抗关系

### 2.3 辅助功能

#### 2.3.1 智能采样
- 支持对大规模数据进行采样以降低计算成本
- 默认不采样（`max_samples=None`），当 `max_samples` 指定时进行采样
- 使用随机种子确保可重复性

#### 2.3.2 多样化样本选择
`_select_diverse_samples` 函数选择具有代表性的样本：
- 高影响样本（SHAP 值总和大的样本）
- 中等影响样本（中位数附近）
- 低影响样本（SHAP 值总和小的样本）
- **目的**：确保可视化的样本覆盖不同的预测模式

#### 2.3.3 错误处理
- 每个可视化功能独立处理，单个失败不影响其他功能
- 记录错误信息到文本文件，便于调试
- 兼容不同版本的 SHAP API

---

## 3. 代码解读

### 3.1 函数签名与参数

```python
def run_lgbm_shap_explain(
    *,
    model: Any,  # LightGBM 模型
    X: np.ndarray,  # 输入特征矩阵
    feature_names: Sequence[str],  # 特征名称列表
    save_dir: str | Path,  # 输出目录
    split: str,  # 数据集划分标识（如 "test"）
    tag: str,  # 实验标签
    max_samples: Optional[int] = None,  # 最大采样数（None=不采样）
    seed: int = 2021,  # 随机种子
    topk: int = 50,  # 显示的特征数量
    n_dependence_plots: int = 10,  # 依赖图数量
    n_waterfall_samples: int = 3,  # 瀑布图样本数
    n_decision_samples: int = 20,  # 决策图样本数
    enable_interaction: bool = False,  # 是否启用交互值
    enable_heatmap: bool = True,  # 是否生成热力图
) -> Optional[dict]:
```

### 3.2 核心代码流程

#### 3.2.1 数据准备与验证

```python
# 验证输入
if X is None or len(X) == 0:
    return None
if len(feature_names) != int(X.shape[1]):
    return None

# 数据采样（如果需要）
if max_samples is not None and max_samples > 0:
    take = min(int(max_samples), n)
    rng = np.random.default_rng(int(seed))
    if take < n:
        idx = rng.choice(n, size=take, replace=False)
        X_use = X[idx]
```

**关键点**：
- 采样使用无放回随机抽样，保证样本独立性
- 采样索引会被排序，便于后续跟踪

#### 3.2.2 SHAP 值计算

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_df)
expected_value = getattr(explainer, "expected_value", None)
```

**关键点**：
- `TreeExplainer` 是针对树模型优化的解释器
- 如果 DataFrame 方式失败，会回退到 NumPy 数组方式
- `expected_value` 是模型在训练集上的平均预测值

#### 3.2.3 特征重要性计算

```python
mean_abs = np.mean(np.abs(shap_values_arr), axis=0)  # 平均绝对 SHAP 值
mean_signed = np.mean(shap_values_arr, axis=0)  # 平均 SHAP 值（带符号）
df_imp = pd.DataFrame({
    "feature": list(feature_names),
    "mean_abs_shap": mean_abs,
    "mean_shap": mean_signed,
}).sort_values("mean_abs_shap", ascending=False)
```

**解释**：
- `mean_abs_shap`：特征的全局重要性，值越大越重要
- `mean_shap`：特征的平均影响方向（正数=平均正向影响，负数=平均负向影响）

#### 3.2.4 Summary Plot 生成

```python
# Beeswarm plot
shap.summary_plot(
    shap_values_arr,
    X_df,
    show=False,
    max_display=int(topk),
)
plt.savefig(out_dir / f"shap_summary_{split}_beeswarm.png")

# Bar plot
shap.summary_plot(
    shap_values_arr,
    X_df,
    plot_type="bar",
    show=False,
    max_display=int(topk),
)
```

**代码细节**：
- `show=False`：不显示图形，直接保存到文件
- `max_display`：限制显示的特征数量，避免图形过于拥挤
- 使用 `plt.tight_layout()` 确保标签不被裁剪
- 高 DPI（220）确保图片清晰度

#### 3.2.5 Dependence Plot 生成

```python
for feat_name, feat_idx in zip(dep_features, dep_feature_indices):
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feat_idx,
        shap_values_arr,
        X_df,
        interaction_index="auto",  # 自动检测交互特征
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_dependence_{feat_name}_{split}.png")
    plt.close()
```

**代码细节**：
- `interaction_index="auto"`：自动选择与当前特征交互最强的特征
- 每个特征生成一个独立的图，便于单独查看
- 使用 `plt.close()` 释放内存

#### 3.2.6 Waterfall Plot 生成

```python
# 选择多样化样本
sample_indices = _select_diverse_samples(shap_values_arr, n_wf, seed)

for i, sample_idx in enumerate(sample_indices):
    explanation = shap.Explanation(
        values=shap_values_arr[sample_idx],
        base_values=base_val,
        data=X_df.iloc[sample_idx].values,
        feature_names=list(feature_names),
    )
    shap.plots.waterfall(explanation, show=False)
    plt.savefig(out_dir / f"shap_waterfall_{split}_sample{i}.png")
```

**代码细节**：
- `shap.Explanation`：封装 SHAP 值和相关元数据
- 使用 `_select_diverse_samples` 确保样本多样性
- 兼容新旧 SHAP API（`shap.plots.waterfall` vs `shap.waterfall_plot`）

#### 3.2.7 Decision Plot 生成

```python
sample_indices = _select_diverse_samples(shap_values_arr, n_dec, seed)
base_val = expected_value if np.isscalar(expected_value) else expected_value[0]

shap.decision_plot(
    base_val,
    shap_values_arr[sample_indices],
    X_df.iloc[sample_indices],
    feature_names=list(feature_names),
    feature_order="importance",  # 按重要性排序
    show=False,
)
```

**代码细节**：
- `feature_order="importance"`：特征按 SHAP 重要性排序，便于识别关键特征
- 图形高度随样本数动态调整（`max(6, n_dec * 0.3)`）

#### 3.2.8 Heatmap 生成

```python
# 限制样本数以避免内存问题
n_heatmap = min(100, X_df.shape[0])
heatmap_indices = _select_diverse_samples(shap_values_arr, n_heatmap, seed)

explanation = shap.Explanation(
    values=shap_values_arr[heatmap_indices],
    base_values=expected_value,
    data=X_df.iloc[heatmap_indices],
    feature_names=list(feature_names),
)
shap.plots.heatmap(explanation, show=False, max_display=int(topk))
```

**代码细节**：
- 限制最多 100 个样本，避免热力图过大
- `max_display`：限制显示的特征数量

#### 3.2.9 Interaction Values 计算

```python
if enable_interaction:
    n_interaction = min(50, X_df.shape[0])  # 限制样本数
    interaction_indices = _select_diverse_samples(shap_values_arr, n_interaction, seed)
    
    # 计算交互值（计算成本高）
    shap_interaction_values = explainer.shap_interaction_values(X_df.iloc[interaction_indices])
    
    # 交互值是一个 3D 数组: [n_samples, n_features, n_features]
    # 其中 shap_interaction_values[i, j, k] 表示样本 i 中特征 j 和 k 的交互
    
    # 计算平均交互强度
    mean_interaction = np.mean(np.abs(shap_interaction_arr), axis=0)
    
    # 计算每个特征的交互强度总和
    feature_interaction_strength = (
        np.sum(mean_interaction, axis=0) + 
        np.sum(mean_interaction, axis=1) - 
        np.diag(mean_interaction)
    )
```

**代码细节**：
- 交互值是一个对称矩阵（`shap_interaction_values[i, j, k] == shap_interaction_values[i, k, j]`）
- `feature_interaction_strength` 计算每个特征与其他所有特征的总交互强度
- 对角线元素（特征与自身的交互）被减去，因为这不代表特征间的交互

### 3.3 辅助函数

#### 3.3.1 `_select_diverse_samples`

```python
def _select_diverse_samples(
    shap_values: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """选择多样化的样本（高/中/低 SHAP 影响）"""
    total_impact = np.sum(np.abs(shap_values), axis=1)
    sorted_indices = np.argsort(total_impact)
    
    if n_samples <= 3:
        # 简单选择：最小值、中位数、最大值
        selected = [
            sorted_indices[0],  # 低影响
            sorted_indices[len(sorted_indices) // 2],  # 中等影响
            sorted_indices[-1],  # 高影响
        ][:n_samples]
    else:
        # 从不同分位数选择
        quantiles = np.linspace(0, len(sorted_indices) - 1, n_samples, dtype=int)
        selected = sorted_indices[quantiles].tolist()
        
        # 添加随机性
        if n_samples > 10:
            rng = np.random.default_rng(seed)
            n_random = min(n_samples // 3, 5)
            for _ in range(n_random):
                idx = rng.integers(0, n)
                if idx not in selected:
                    selected.append(idx)
    
    return np.array(sorted(set(selected))[:n_samples])
```

**设计思路**：
- 通过 SHAP 值的总和衡量样本的"影响程度"
- 选择分布在不同分位数的样本，确保覆盖各种预测模式
- 添加随机性避免样本选择过于规律

#### 3.3.2 `_jsonable`

```python
def _jsonable(x: Any) -> Any:
    """将任意 Python 对象转换为 JSON 可序列化格式"""
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    try:
        return float(x)  # 尝试转换为浮点数
    except Exception:
        return str(x)  # 最后转换为字符串
```

**用途**：将 NumPy 数组、标量等转换为 JSON 可序列化格式，便于保存到 JSON 文件。

---

## 4. 使用示例

### 4.1 基本使用

```python
from exp.pipeline.shap_explain import run_lgbm_shap_explain

result = run_lgbm_shap_explain(
    model=lgbm_model,
    X=X_test,
    feature_names=feature_names,
    save_dir="./results/my_experiment",
    split="test",
    tag="experiment_v1",
    max_samples=1000,  # 使用 1000 个样本
    topk=30,  # 显示前 30 个特征
)
```

### 4.2 完整配置

```python
result = run_lgbm_shap_explain(
    model=lgbm_model,
    X=X_test,
    feature_names=feature_names,
    save_dir="./results/my_experiment",
    split="test",
    tag="experiment_v1",
    max_samples=5000,  # 采样 5000 个样本
    seed=42,
    topk=50,
    n_dependence_plots=15,  # 为前 15 个特征生成依赖图
    n_waterfall_samples=5,  # 生成 5 个瀑布图
    n_decision_samples=30,  # 决策图包含 30 个样本
    enable_interaction=True,  # 启用交互值分析
    enable_heatmap=True,  # 启用热力图
)
```

### 4.3 在 Pipeline 中使用

在 `exp/exp_pipeline.py` 中，SHAP 分析会在模型训练和评估后自动执行：

```python
# 5) SHAP explainability
if not getattr(self.args, "disable_shap", False):
    run_lgbm_shap_explain(
        model=model,
        X=X_te,
        feature_names=meta.get("feature_names", []),
        save_dir=self.args.save_dir,
        split="test",
        tag=getattr(self.args, "tag", "pipeline"),
        max_samples=int(shap_max_samples) if shap_max_samples is not None else None,
        seed=int(getattr(self.args, "shap_seed", getattr(self.args, "seed", 2021))),
        topk=int(getattr(self.args, "shap_topk", 50)),
        # ... 其他参数
    )
```

---

## 5. 输出文件说明

### 5.1 数据文件

#### `shap_values_{split}.npy`
- **格式**：NumPy 数组
- **形状**：`(n_samples, n_features)`
- **内容**：每个样本、每个特征的 SHAP 值
- **用途**：后续分析、自定义可视化

#### `shap_expected_value_{split}.json`
- **格式**：JSON 文件
- **内容**：
  ```json
  {
    "tag": "experiment_v1",
    "split": "test",
    "expected_value": 50.3,  # 模型基准值
    "n_rows_total": 10000,
    "n_rows_used": 1000,
    "sampled": true,
    "max_samples": 1000,
    "seed": 42
  }
  ```
- **用途**：记录分析元数据和参数

#### `shap_importance_{split}.csv`
- **格式**：CSV 文件
- **列**：`feature`, `mean_abs_shap`, `mean_shap`
- **排序**：按 `mean_abs_shap` 降序
- **用途**：特征重要性排序表

#### `shap_interaction_values_{split}.npy`（如果启用）
- **格式**：NumPy 数组
- **形状**：`(n_samples, n_features, n_features)`
- **内容**：特征对之间的交互 SHAP 值
- **用途**：特征交互分析

### 5.2 可视化文件

#### 汇总图
- `shap_summary_{split}_beeswarm.png`：蜂群图
- `shap_summary_{split}_bar.png`：条形图

#### 依赖图
- `shap_dependence_{feature_name}_{split}.png`：每个重要特征的依赖图

#### 瀑布图
- `shap_waterfall_{split}_sample{0}.png`：第 1 个代表性样本
- `shap_waterfall_{split}_sample{1}.png`：第 2 个代表性样本
- ...

#### 决策图
- `shap_decision_{split}.png`：多样本决策路径对比

#### 热力图
- `shap_heatmap_{split}.png`：样本-特征 SHAP 值热力图

#### 交互值图
- `shap_interaction_summary_{split}.png`：特征交互矩阵热力图

### 5.3 错误日志文件

如果某个功能执行失败，会生成相应的错误日志文件：
- `shap_{split}_skipped.txt`：SHAP 计算被跳过
- `shap_{split}_missing_shap.txt`：SHAP 库未安装
- `shap_{split}_plot_failed.txt`：绘图失败
- `shap_dependence_{feature}_{split}_failed.txt`：特定依赖图失败
- `shap_waterfall_{split}_sample{i}_failed.txt`：特定瀑布图失败
- ...

---

## 6. 最佳实践

### 6.1 性能优化

1. **数据采样**：对于大型数据集，使用 `max_samples` 限制计算样本数
   ```python
   max_samples=5000  # 对于 10 万样本，采样 5000 个足够代表整体
   ```

2. **交互值计算**：仅在需要时启用（计算成本高）
   ```python
   enable_interaction=True  # 仅当需要分析特征交互时启用
   ```

3. **特征数量**：使用 `topk` 限制显示的特征数量，避免图形过于拥挤
   ```python
   topk=30  # 只显示前 30 个重要特征
   ```

### 6.2 可解释性建议

1. **理解 Summary Plot**：
   - Beeswarm plot 适合探索性分析
   - Bar plot 适合展示给非技术人员

2. **使用 Waterfall Plot**：
   - 解释异常预测
   - 向业务人员展示单个预测结果

3. **Dependence Plot 的重要性**：
   - 揭示特征的非线性关系
   - 发现阈值效应（如"当温度 > 30°C 时，影响反转"）

4. **Heatmap 的时间序列应用**：
   - 如果样本有时间顺序，可以通过排序发现时间模式
   - 适合发现周期性或趋势性变化

### 6.3 调试技巧

1. **检查错误日志**：如果某些图未生成，检查对应的 `*_failed.txt` 文件

2. **验证 SHAP 值**：
   ```python
   # 对于回归任务，应该满足：
   # model.predict(X[i]) ≈ expected_value + sum(SHAP[i, :])
   prediction = model.predict(X[0])
   shap_sum = shap_values[0].sum()
   print(f"预测值: {prediction}, 基准值+SHAP: {expected_value + shap_sum}")
   ```

3. **内存管理**：对于超大数据集，逐步增加 `max_samples` 测试内存占用

---

## 7. 参考文献

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in neural information processing systems*, 30.

2. Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature machine intelligence*, 2(1), 56-67.

3. SHAP Documentation: https://shap.readthedocs.io/

---

## 附录：SHAP 值的数学性质

SHAP 值满足以下四个公理（来自 Shapley 值理论）：

1. **效率性（Efficiency）**：
   $$
   \sum_{i=1}^{M} \phi_i = f(x) - E[f(X)]
   $$

2. **对称性（Symmetry）**：如果两个特征对模型的影响相同，它们的 SHAP 值相等

3. **虚拟性（Dummy）**：如果特征对模型没有影响，其 SHAP 值为 0

4. **可加性（Additivity）**：如果模型是两个模型的组合，SHAP 值也是可加的

这些性质确保了 SHAP 值是一种公平且一致的特征贡献分配方法。

---

*文档版本：1.0*  
*最后更新：2025-01-XX*
