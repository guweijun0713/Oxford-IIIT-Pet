# Oxford-IIIT Pet 37 类细粒度图像分类

这是一个可复现的 PyTorch 细粒度分类项目：使用 ImageNet 预训练 ResNet-18，在 Oxford-IIIT Pet 37 类数据上比较 Baseline、RandAugment 和 Label Smoothing 三组固定种子实验，并提供独立测试评估、Top-5、Macro-F1、混淆矩阵和 Grad-CAM。

作者：王澎宇（W124302227）  
邮箱：w124302227@stu.ahu.edu.cn  
仓库：[github.com/guweijun0713/Oxford-IIIT-Pet](https://github.com/guweijun0713/Oxford-IIIT-Pet)

## 正式实验结果

固定 `seed=42`。模型只按验证集 Top-1 选择；测试集不参与模型选择。

| 实验 | 唯一变化 | Best Val Top-1 | Test Top-1 | Test Top-5 | Test Macro-F1 | 训练耗时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 无（Crop + Flip） | 92.74% | 90.39% | 99.00% | 90.33% | 日志跨度约 214 s |
| RandAugment | 训练增强增加 `RandAugment(2, 9)` | 92.29% | 91.66% | 99.27% | 91.66% | 553.98 s |
| Label Smoothing | 训练损失 `label_smoothing=0.1` | **93.65%** | **92.38%** | 98.91% | **92.36%** | 550.82 s |

RandAugment 的验证集准确率略低于 Baseline（92.29% vs 92.74%），但测试集准确率更高（91.66% vs 90.39%）。这说明单次固定划分上的验证/测试排序存在抽样波动，不能据此声称统计显著；本项目仍按预先约定的验证集指标选择模型。最终混淆矩阵和 Grad-CAM 均来自验证集表现最高的 Label Smoothing checkpoint。

## 项目结构

```text
.
├── data/
│   ├── dataset.py              # 数据下载、分层划分与 Transform
│   └── oxford-iiit-pet/        # 本地数据（Git 忽略）
├── models/model.py             # ResNet-18 与安全 checkpoint 加载
├── utils/
│   ├── metrics.py              # Top-1/Top-5/F1、曲线与混淆矩阵
│   └── gradcam.py              # Grad-CAM 与样本选择
├── train.py                    # 训练入口：不访问测试集
├── evaluate.py                 # 独立测试评估与可视化
├── baseline.py                 # Day 1 兼容入口
├── day2_experiments.py         # Day 2 兼容入口
├── day2_visualize.py           # Day 2 可视化兼容入口
├── tests/                      # 单元与合成集成测试
└── artifacts/                  # 指标、预测明细、图表和表格
```

## 数据与实验协议

程序合并 torchvision 官方 `trainval` 和 `test`，再以标签分层抽样重新划分。总计 7,349 张图片、37 类，训练/验证/测试为 `5144 / 1102 / 1103`；三组实验使用完全相同的索引。

- 训练：`Resize(256) → RandomCrop(224) → RandomHorizontalFlip → ToTensor → ImageNet Normalize`
- 验证/测试：`Resize(256) → CenterCrop(224) → ToTensor → ImageNet Normalize`
- 模型：ImageNet 预训练 ResNet-18，全参数微调，37 类分类头
- 优化：AdamW，`lr=1e-4`，`weight_decay=1e-4`，Batch Size 32，15 Epoch
- Baseline：基础增强 + 标准 CrossEntropy
- RandAugment：只在基础训练增强后增加 `RandAugment(num_ops=2, magnitude=9)`
- Label Smoothing：只把训练损失改为 `CrossEntropyLoss(label_smoothing=0.1)`；验证/测试仍为标准 CrossEntropy

## 环境安装

### Windows + Anaconda

```powershell
git clone https://github.com/guweijun0713/Oxford-IIIT-Pet.git
Set-Location -LiteralPath ".\Oxford-IIIT-Pet"

conda create -n pet37 python=3.11 -y
conda activate pet37

python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip check

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unavailable')"
```

NVIDIA 驱动支持对应 CUDA runtime 即可，不需要另外安装 CUDA Toolkit 或 cuDNN。PowerShell 使用 `Set-Location`，不要使用 CMD 专属的 `cd /d`。

### Linux + Conda

```bash
git clone https://github.com/guweijun0713/Oxford-IIIT-Pet.git
cd Oxford-IIIT-Pet

conda create -n pet37 python=3.11 -y
conda activate pet37

python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip check

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unavailable')"
```

### Google Colab

先在“运行时 → 更改运行时类型”中选择 GPU，然后执行：

```python
!git clone https://github.com/guweijun0713/Oxford-IIIT-Pet.git
%cd Oxford-IIIT-Pet

!python -m pip install --upgrade pip
!python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
!python -m pip install -r requirements.txt
!python -m pip check

import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unavailable")
```

## 一键复现

完整重训三组实验约 45 Epoch。`train.py` 只使用训练集和验证集，测试集仅由 `evaluate.py` 访问一次：

```bash
python train.py
python evaluate.py
```

首次运行会自动下载数据和 ImageNet 预训练权重。普通源码提交不直接包含约 128 MB/个的 checkpoint；可以从 GitHub Release 下载正式权重后直接评估，也可以先运行 `train.py` 从零训练、再运行 `evaluate.py`。
如果出现数据下载失败情况，可在GitHub release中下载数据集放入到data文件目录下
只训练某一实验：

```bash
python train.py --experiments label_smoothing
```

已有本地三份 checkpoint、history 和 `training_summary.csv` 时，仅评估：

```bash
python evaluate.py --num-workers 0
```

一轮真实数据冒烟测试：

```bash
python train.py --experiments baseline --epochs 1 --num-workers 0 --artifact-dir artifacts/smoke --log-dir runs/smoke
python evaluate.py --experiments baseline --num-workers 0 --artifact-dir artifacts/smoke
```

Windows 若遇到 `WinError 1455` 或 DataLoader 卡住，只降低 worker 数，不改变实验超参数：

```powershell
python train.py --num-workers 0
```

若显存不足，只把 Batch Size 降为 16：

```powershell
python train.py --batch-size 16
```

查看 TensorBoard：

```powershell
tensorboard --logdir .\runs --port 6006
```

浏览器访问 `http://localhost:6006`。

`train.py` 会在本地 `runs/{experiment}/` 中生成 TensorBoard 原始日志。

## 测试与复现产物

```powershell
python -m py_compile train.py evaluate.py data\dataset.py models\model.py utils\metrics.py utils\gradcam.py
python -m unittest discover -s tests -v
```

主要正式产物：

- `artifacts/results.csv`：三组 Val、Test Top-1/Top-5、Macro-F1
- `artifacts/predictions/*.npz`：1,103 条标签、预测、Top-5 与置信度
- `artifacts/figures/`：收敛曲线、消融柱状图、混淆矩阵和 Grad-CAM
- `artifacts/tables/`：混淆矩阵与 Top-3 易混淆品种对

## 复现边界

本结果来自单一 `seed=42` 和一次固定分层划分，没有多种子均值、标准差或显著性检验。结论只描述本次受控实验，不外推为统计显著规律。Oxford-IIIT Pet 原始数据由 torchvision 自动下载，不纳入仓库；三个正式 checkpoint 通过 GitHub Release 单独发布；CSV、NPZ、PNG 和 JSON 随源码仓库提交。
