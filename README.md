# FutureBoosting

FutureBoosting 是一个两阶段时序预测实验框架：

1. 用 TSFM 做 rollout 预测
2. 把 TSFM 预测和原始协变量拼接后，交给第二阶段回归器做融合

项目支持两类数据：
- Shanxi 省电价数据
- 标准滑窗数据，使用 `--is_std` 控制

## 支持功能

- TSFM rollout、缓存、zero-shot 评估
- 特征构造与训练/验证/测试切分
- 第二阶段回归器：
  - `lgbm`
  - `linear`
  - `lgbm+linear`
- `linear` 支持：
  - `ridge`
  - `lasso`
  - `elasticnet`
- 统一评估、绘图、指标汇总
- SHAP 全局解释和 casebook

当前支持的 TSFM：
- `sundial`
- `timerxl`
- `chronos2`
- `tirex`
- `moirai2`
- `timesfm`
- `tabpfn`

## 目录结构

```text
FutureBoosting/
├── run_pipeline.py
├── README.md
├── pyproject.toml
├── configs/
├── data_provider/
│   └── data_loader.py
├── exp/
│   ├── exp_pipeline.py
│   └── pipeline/
│       ├── tsfm_infer.py
│       ├── feature_select.py
│       ├── regressor.py
│       ├── evaluator.py
│       ├── shap_explain.py
│       ├── shap_case.py
│       └── eff_profile.py
├── ts_models/
│   ├── base.py
│   ├── factory.py
│   └── adapters/
├── scripts/
│   ├── shanxi/
│   └── realE/
└── results/
```

## 环境

建议：
- Python 3.12
- CUDA 可用
- 已准备好各 TSFM checkpoint

安装：

```bash
uv sync
```

## 运行

在项目根目录下运行脚本即可。

Shanxi 脚本目录：
- `scripts/shanxi/dayahead/`
- `scripts/shanxi/realtime/`

REALE 脚本目录：
- `scripts/realE/DE/`
- `scripts/realE/FR/`

示例：

```bash
bash scripts/shanxi/dayahead/pipeline_ic94_tsfm_ic27_lgbm_chronos2.sh
```

```bash
bash scripts/realE/FR/FR_ic16_ic15_chronos2.sh
```

## 脚本说明

脚本顶部一般包含：
- 数据路径
- 列配置路径
- 切分配置路径
- TSFM 模型名
- 回归器配置

当前脚本默认已经支持：
- `regression_model="lgbm+linear"`
- `linear_method="ridge"`

如果要改线性模型，只需要改脚本顶部变量，例如：

```bash
regression_model="linear"
linear_method="elasticnet"
linear_alpha=0.001
linear_l1_ratio=0.3
```

或只跑 LightGBM：

```bash
regression_model="lgbm"
```

## TSFM checkpoint 配置

脚本里的 `tsfm_model_paths` 需要按你自己的环境配置，例如：

```bash
tsfm_model_paths="$(cat <<'JSON'
{
  "sundial":  "/path/to/Sundial",
  "timerxl":  "/path/to/TimerXL",
  "chronos2": "/path/to/Chronos2",
  "tirex":    "/path/to/TiRex",
  "moirai2":  "/path/to/Moirai2",
  "timesfm":  "/path/to/TimesFM",
  "tabpfn":   "/path/to/TabPFN"
}
JSON
)"
```

## 输出

常见输出包括：
- `metrics_test.json`
- `metrics_all.csv`
- `plots/`
- `shap/`
- `tsfm_cache/*.parquet`
- `metrics_efficiency_cache.csv`
