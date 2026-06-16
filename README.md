# IKD

IKD is a PyTorch/timm project for ImageNet adversarial attack experiments. It compares baseline attack variants with IKD-regularized variants and records benign accuracy, adversarial accuracy, and black-box attack success rate (ASR) across target models.

IKD 是一个基于 PyTorch 和 timm 的 ImageNet 对抗攻击实验项目。项目用于比较 baseline attack 与 IKD regularized variants，并在不同目标模型上记录 benign accuracy、adversarial accuracy 和 black-box attack success rate (ASR)。

## Repository Structure / 仓库结构

```text
attack/                      Attack implementations and IKD variants
attack_main.py               In-memory experiment entry point
data/dev_data/               Default ImageNet-style evaluation data location
results/                     Experiment logs grouped by attack/model/regularization
figures/parameter_analysis/  Generated parameter-analysis CSV, PDF, and PNG files
tests/                       Unit tests for plotting and result parsing utilities
run.sh                       Batch commands for repeated experiments
```

## Environment / 环境

The project was developed with Python 3.8 and CUDA-enabled PyTorch. Install the direct Python dependencies with:

本项目推荐使用 Python 3.8 和支持 CUDA 的 PyTorch 环境。安装依赖：

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` pins PyTorch 1.12.1 with CUDA 11.3 wheels. If your machine uses a different CUDA version, CPU-only PyTorch, or a newer GPU stack, replace the `torch` and `torchvision` lines with the install command from the official PyTorch selector before installing the rest of the dependencies.

`requirements.txt` 当前固定为 PyTorch 1.12.1 + CUDA 11.3 wheel。如果你的机器使用不同 CUDA 版本、CPU-only PyTorch 或更新 GPU 环境，请先按 PyTorch 官方安装选择器替换 `torch` 和 `torchvision` 两行，再安装其余依赖。

## Data Preparation / 数据准备

The attack scripts read data from:

攻击脚本默认读取以下路径：

```text
data/dev_data/val_rs.csv
data/dev_data/val_rs/
```

`val_rs.csv` should contain image names and labels. `val_rs/` should contain the corresponding image files. If you need to download images from a dataset CSV, use:

`val_rs.csv` 需要包含图片文件名和标签，`val_rs/` 需要包含对应图片。若需要从数据集 CSV 下载图片，可使用：

```bash
mkdir -p data/dev_data/val_rs
python data/download_images.py \
  --input_file data/dev_dataset.csv \
  --output_dir data/dev_data/val_rs
```

The downloader expects columns such as `ImageId`, `URL`, `x1`, `y1`, `x2`, and `y2`.

下载脚本需要输入 CSV 包含 `ImageId`、`URL`、`x1`、`y1`、`x2`、`y2` 等列。

## Running Attacks / 运行攻击

Run a single attack experiment with:

运行单个攻击实验：

```bash
python attack_main.py \
  --attack difgsm \
  --model resnet50 \
  --regularization KL \
  --weight 0.01 \
  --seed 1111 \
  --batch 40
```

Results are written to:

结果会写入：

```text
results/<attack>/<model>/<regularization>/<seed>/<weight>/log (acc, asr).csv
```

For repeated experiments, edit or run the commands in:

批量实验可使用或修改：

```bash
bash run.sh
```

### Attack entry point / 攻击入口

`attack_main.py` is the supported experiment entry point. It generates adversarial examples once, keeps cached batches in memory, and evaluates the full target model list configured in `MODEL_CONFIG`. Its default regularization is `KL`.

`attack_main.py` 是当前支持的实验入口。它一次生成对抗样本，将 batch 缓存在内存中，并按 `MODEL_CONFIG` 中的完整目标模型列表评估。默认 `regularization=KL`。

## Plotting / 绘图

Generate weight analysis figures from existing logs:

基于已有日志生成权重分析图：

```bash
python plot_weight_analysis.py \
  --results-dir results \
  --source-model resnet50 \
  --regularization KL \
  --seed 1111 \
  --baseline-weight 0.01 \
  --output-dir figures/parameter_analysis
```

Generate IKD ablation figures:

生成 IKD ablation 图：

```bash
python plot_fd_ablation.py \
  --results-dir results \
  --source-model resnet50 \
  --seed 1111 \
  --weight 0.01 \
  --output-dir figures/parameter_analysis
```

The script filename is kept for compatibility; its default outputs use the `ikd_ablation_blackbox_asr` prefix.

脚本文件名为兼容旧入口保留；默认输出已使用 `ikd_ablation_blackbox_asr` 前缀。

Both plotting scripts write summary CSV files plus PDF and PNG figures.

两个绘图脚本都会输出 summary CSV，以及 PDF/PNG 图像。

## Testing / 测试

Run the unit tests with:

运行单元测试：

```bash
python -m pytest tests
```

The tests cover parsing, aggregation, plotting metadata, output naming, and summary CSV generation for the parameter-analysis scripts.

测试覆盖参数分析脚本中的日志解析、聚合、绘图元数据、输出命名和 summary CSV 生成。
