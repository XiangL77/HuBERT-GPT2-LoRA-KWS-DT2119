# Lightweight Keyword Spotting with HuBERT and GPT-2 LoRA

## English

This project implements a lightweight keyword spotting pipeline using HuBERT tokenization and GPT-2 with LoRA fine-tuning.

The system first converts speech audio into discrete HuBERT tokens, then trains GPT-2 sequence classifiers for binary and multi-class keyword spotting.

## Results / 实验结果

| Metric | Binary Classification | Multi-class Classification |
| --- | ---: | ---: |
| Accuracy | 99.07% | 95.48% |
| Precision | 95.12% | 94.97% |
| Recall | 95.61% | 95.38% |
| F1-score | 95.36% | 95.16% |
| Loss | 0.0357 | 0.1340 |
| Epochs | 2 | 3 |

### Main Files

- `data_prepro.py`: preprocess raw Google Speech Commands audio.
- `HuBERT.ipynb`: extract HuBERT features and generate token files.
- `GPT2.ipynb`: train the binary keyword classifier.
- `GPT2_multi.ipynb`: train the multi-class keyword classifier.
- `kmeans_model.pkl`: KMeans model for HuBERT token quantization.
- `data/`: HuBERT-tokenized data and GPT-2 training JSONL files.
- `figures/`: result figures and pipeline images.
- `Lightweight Keyword Spotting with HuBERT and GPT 2 using LoRA Fine Tuning.pdf`: project report.

### Data

The original dataset is Google Speech Commands v0.02.

Raw audio files and preprocessed wav files are not included because they are large. The included JSONL files can be used directly for GPT-2 training.

### Run Order

```text
data_prepro.py
-> HuBERT.ipynb
-> GPT2.ipynb or GPT2_multi.ipynb
```

If `tokens_*.jsonl` already exists, training can start directly from `GPT2.ipynb` or `GPT2_multi.ipynb`.

### Notes

Large full-model weights such as `model.safetensors` are not suitable for normal GitHub upload. Use Git LFS or share only small LoRA adapter files.

## 中文

本项目实现了一个轻量级关键词识别流程，使用 HuBERT 将语音转换为离散 token，再使用 LoRA 微调 GPT-2 进行分类。

整体流程是：先把语音音频转换成 HuBERT token 序列，再训练 GPT-2 序列分类模型，用于二分类和多分类关键词识别。

### 主要文件

- `data_prepro.py`：预处理 Google Speech Commands 原始音频。
- `HuBERT.ipynb`：提取 HuBERT 特征并生成 token 文件。
- `GPT2.ipynb`：训练二分类关键词识别模型。
- `GPT2_multi.ipynb`：训练多分类关键词识别模型。
- `kmeans_model.pkl`：用于 HuBERT 特征离散化的 KMeans 模型。
- `data/`：HuBERT token 数据和 GPT-2 训练用 JSONL 文件。
- `figures/`：实验结果图片和流程图。
- `Lightweight Keyword Spotting with HuBERT and GPT 2 using LoRA Fine Tuning.pdf`：项目报告。

### 数据说明

原始数据集使用 Google Speech Commands v0.02。

原始音频和预处理后的 wav 文件较大，因此不上传到 GitHub。当前仓库中的 JSONL 文件已经可以直接用于 GPT-2 训练。

### 运行顺序

```text
data_prepro.py
-> HuBERT.ipynb
-> GPT2.ipynb 或 GPT2_multi.ipynb
```

如果已经有 `tokens_*.jsonl`，可以直接从 `GPT2.ipynb` 或 `GPT2_multi.ipynb` 开始训练。

### 注意

`model.safetensors` 这类完整模型权重文件较大，不适合普通 GitHub 上传。建议使用 Git LFS，或者只分享较小的 LoRA adapter 文件。
