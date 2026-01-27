# LoRA 大模型微调框架

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.5.1+-red.svg" alt="PyTorch Version">
  <img src="https://img.shields.io/badge/Transformers-4.40.0+-green.svg" alt="Transformers Version">
  <img src="https://img.shields.io/badge/PEFT-0.10.0+-purple.svg" alt="PEFT Version">
</div>

<br>

本项目提供了一个轻量级、高效的 LoRA (Low-Rank Adaptation) 微调框架，用于微调大语言模型 (LLM) 和视觉语言模型 (VLM)。通过参数高效的微调方法，实现对大型模型的快速适配和定制。

## 📑 目录

- [项目结构](#-项目结构)
- [核心特性](#-核心特性)
- [快速开始](#-快速开始)
  - [环境准备](#1-环境准备)
  - [安装依赖](#2-安装依赖)
  - [模型准备](#3-模型准备)
  - [数据准备](#4-数据准备)
  - [配置参数](#5-配置参数)
  - [开始训练](#6-开始训练)
  - [评估模型](#7-评估模型)
  - [使用模型推理](#8-使用模型推理)
- [LoRA 工作流程详解](#-lora-工作流程详解)
- [LoRA 与预训练模型的关系](#-lora-与预训练模型的关系)
- [命令行参数](#-命令行参数)
- [输出目录说明](#-输出目录说明)
- [模型精度与内存优化指南](#-模型精度与内存优化指南)
- [注意事项](#-注意事项)
- [常见问题排查](#-常见问题排查)
- [示例：LLM 和 VLM 微调](#-示例llm-和-vlm-微调)
- [更新日志](#-更新日志)

## 📋 项目结构

```
├── Qwen3-1.7B/               # 本地预训练LLM模型目录
├── Qwen3-4B/                 # 本地预训练LLM模型目录
├── Qwen3-VL-2B-Instruct/     # 本地预训练VLM模型目录
├── Qwen3-VL-4B-Instruct/     # 本地预训练VLM模型目录
├── Qwen3-VL-4B-Instruct-FP8/ # 本地预训练VLM模型目录
├── data/                     # 训练和验证数据
│   ├── llm/                  # LLM（大语言模型）数据
│   │   ├── train.json        # LLM 训练数据
│   │   └── validation.json   # LLM 验证数据
│   └── vlm/                  # VLM（视觉语言模型）数据
│       ├── images/           # VLM 图片目录
│       ├── prompt.json       # VLM 系统提示词
│       └── label_map.json    # VLM 标签映射
├── output/                   # 输出目录（包含训练检查点和LoRA模型）
├── triton/                   # triton whl文件目录
├── config.yaml               # 配置文件
├── main.py                   # 主执行脚本
├── data_processing.py        # 数据处理脚本
├── model_config.py           # 模型配置脚本
├── train.py                  # 训练脚本
├── evaluate.py               # 评估脚本
├── inference.py              # 推理脚本
├── requirements.txt          # 环境依赖包
└── README.md                 # 项目说明文档
```

## 🌟 核心特性

### 参数高效
- LoRA 只训练和存储少量参数（通常是原始模型的 1-3%）
- 训练生成的 LoRA 模块可灵活拼装到原始模型，即插即用

### 多模型支持
- **LLM**：兼容 Qwen3、GPT2、DistilGPT2 等多种大语言模型
- **VLM**：支持 Qwen3-VL 等视觉语言模型

### 内存优化
- **FP16 训练**：支持混合精度训练，大幅降低内存使用
- **量化支持**：内置 8-bit 和 4-bit 量化，训练和推理可独立配置
  - 训练时：预训练模型量化，LoRA 适配器全精度训练
  - 推理时：根据部署环境灵活调整量化配置
- **FP8 模型**：支持 FP8 量化模型（Windows 平台优化中）

### 分布式训练
- 多 GPU 分布式训练，自动设备分配
- 评估和推理默认单设备，确保稳定性
- 智能设备映射，避免多卡环境崩溃

### 完整流程
- 从数据准备到训练、评估、推理的全流程支持
- 模块化设计，易于扩展和维护
- 代码结构优化，提升可维护性

## 🚀 快速开始

### 1. 环境准备

确保您已经安装了 Python 3.10+ 环境，并且创建了名为 `Lora` 的 Conda 环境（本文档以3.11为例）：

```powershell
conda create -n Lora python=3.11
conda activate Lora
```

### 2. 安装依赖

#### 2.1 安装 GPU 版本 PyTorch（推荐）

根据您的 GPU 驱动版本选择合适的 CUDA 版本：

| 驱动版本        | 支持的 CUDA 版本 |
|---------------|----------------|
| ≥ 550.00      | CUDA 12.4      |
| ≥ 546.00      | CUDA 12.3      |
| ≥ 535.00      | CUDA 12.2      |
| ≥ 525.00      | CUDA 12.1      |
| ≥ 515.00      | CUDA 12.0      |

**安装命令示例（CUDA 12.1）**：

```powershell
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

#### 2.2 安装其他依赖

您可以直接使用 requirements.txt 安装所有依赖：

```powershell
pip install -r requirements.txt
```

#### 2.3 安装 Triton (Windows 用户必看)

某些依赖库（如 bitsandbytes）在 Windows 系统上可能需要 triton 库。由于 PyPI 中没有适用于 Windows 的官方版本，直接使用 `pip install triton` 会报错。

本项目已包含 triton 3.0.0 版本的 Windows 包，位于 `./triton/` 目录。请根据您的 Python 版本选择对应的包进行安装：

```powershell
# Python 3.10
pip install ./triton/triton-3.0.0-cp310-cp310-win_amd64.whl

# Python 3.11
pip install ./triton/triton-3.0.0-cp311-cp311-win_amd64.whl

# Python 3.12
pip install ./triton/triton-3.0.0-cp312-cp312-win_amd64.whl
```

#### 2.4 验证安装

```powershell
python -c "import torch; print('PyTorch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available())"
```

### 3. 模型准备

本框架支持以下两种方式使用模型：

#### 3.1 使用本地模型（推荐）

将下载好的模型放在项目根目录下，例如：
- `./Qwen3-1.7B` - [Hugging Face 模型链接](https://huggingface.co/Qwen/Qwen3-1.7B)
- `./Qwen3-4B` - [Hugging Face 模型链接](https://huggingface.co/Qwen/Qwen3-4B)
- `./Qwen3-VL-2B-Instruct` - [Hugging Face 模型链接](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- `./Qwen3-VL-4B-Instruct` - [Hugging Face 模型链接](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

### 3.2 使用 Hugging Face Hub 模型

直接在配置文件中设置模型名称，例如：
- `distilgpt2` - [Hugging Face 模型链接](https://huggingface.co/distilgpt2)
- `gpt2` - [Hugging Face 模型链接](https://huggingface.co/gpt2)
- `facebook/opt-125m` - [Hugging Face 模型链接](https://huggingface.co/facebook/opt-125m)
- `Qwen/Qwen3-VL-2B-Instruct` - [Hugging Face 模型链接](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

### 4. 数据准备

根据您要微调的模型类型，准备相应的数据：

> **📌 数据集划分策略**
> - **LLM (手动划分)**：需要手动拆分为 `train.json` 和 `validation.json`，便于精确控制
> - **VLM (自动划分)**：只需将所有图片放入一个文件夹，系统自动进行分层随机划分（建议每个类别至少 10-20 张图片）

#### 4.1 LLM（大语言模型）数据准备

##### 4.1.1 使用示例数据（快速测试）

项目已包含日常生活建议的示例数据：
- `data/llm/train.json` - 15 个训练样本
- `data/llm/validation.json` - 5 个验证样本

##### 4.1.2 准备自定义 LLM 数据

在 `data/llm/` 目录下创建 `train.json` 和 `validation.json` 文件，格式如下：

```json
[
    {
        "text": "如何保持良好的睡眠质量？",
        "target": "保持良好的睡眠质量可以尝试以下方法：1. 建立规律的作息时间..."
    }
]
```

> **注意**：LLM 数据需要手动划分为训练集和验证集，系统不会自动划分。

#### 4.2 VLM（视觉语言模型）数据准备

##### 4.2.1 目录结构

VLM 数据需要以下文件和目录：

```
data/vlm/
├── images/          # 图片目录，存放所有训练和验证图片
├── prompt.json      # 系统提示词，指导模型的行为和输出格式
└── label_map.json   # 标签映射，定义图片文件名与输出结果的对应关系
```

##### 4.2.2 准备图片数据

1. 将您的图片放入 `data/vlm/images/` 目录
2. 图片文件名格式建议为：`{tags}_{其他信息}.jpg`
   - 例如：`000_20260103.jpg`（000 为标签编码）
   - 编码规则由 `label_map.json` 定义，每一位代表一个属性。

##### 4.2.3 配置系统提示词

编辑 `data/vlm/prompt.json` 文件，定义系统提示词。您可以直接使用纯文本格式，无需 JSON 结构：

```text
你是一个工业安全视觉检测AI。你的任务是分析除尘室监控图像，判断是否存在“管道粉尘泄漏”。

注意：图片可能包含“日间彩色模式”或“夜间红外黑白模式”。

请严格按照以下步骤进行推理分析，并输出JSON结果：
...
```

##### 4.2.4 配置标签映射

编辑 `data/vlm/label_map.json` 文件，定义标签编码规则。支持**位置编码**模式，每一位对应一个字段：

```json
{
    "mapping_type": "positional",
    "positions": [
        {
            "field": "status",
            "map": { "0": "No Leakage", "1": "Leakage" }
        },
        {
            "field": "sub_type",
            "map": { 
                "0": "Normal", 
                "1": "Ambient Dust", 
                "2": "Light Beams",
                "3": "High Pressure", 
                "4": "Low Pressure" 
            }
        },
        {
            "field": "is_night_mode",
            "map": { "0": false, "1": true }
        }
    ],
    "defaults": {
        "confidence": 1.0,
        "reasoning": "Auto-generated."
    }
}
```

> **关于置信度 (confidence)**:
> `defaults` 中的 `confidence: 1.0` 仅用于生成训练数据的目标标签（Ground Truth）。这意味着我们教模型：“对于这类样本，你应该非常确信”。
> 在实际推理时，模型输出的 `confidence` 值是它根据图像特征和学到的模式自动生成的，不一定会一直是 1.0。

例如，文件名 `130_xxx.jpg` 将被解析为：
- 第1位 `1` -> `status: Leakage` (有泄漏)
- 第2位 `3` -> `sub_type: High Pressure` (高压泄漏)
- 第3位 `0` -> `is_night_mode: false` (日间模式)

> **数据集划分**: VLM 数据集会自动进行**分层随机划分**，确保训练集和验证集中各类别的样本比例均衡，从而提高模型评估的可靠性。

##### 4.2.5 (可选) VLM 数据增强
当图片数据量不足时，**数据增强**是提升模型性能和泛化能力的关键手段。它通过对现有图片进行旋转、翻转、调整亮度/对比度等操作，来模拟真实世界中的各种变化。

目前项目**尚未内置**自动数据增强功能，但您可以通过以下方式手动实现：
- **离线增强 (推荐)**: 使用脚本或工具 (如 `Pillow`, `OpenCV`) 提前生成增强后的图片副本，并将它们与原始图片一起放入 `images` 文件夹。这是最简单且无需修改代码的方式。
- **在线增强 (高级)**: 修改 `data_processing.py` 文件，在加载图片时使用 `torchvision.transforms` 或 `albumentations` 等库动态进行数据增强。

##### 4.2.6 (可选) 处理类别不均衡
在许多现实场景中（如缺陷检测），“正常”样本的数量远多于“异常”样本。这种**类别不均衡**问题会导致模型倾向于预测多数类，而忽略少数类。

**解决方案：手动过采样 (Manual Over-sampling)**
这是最简单且有效的平衡数据方法，无需修改代码：
1. **识别少数类**: 找到 `images` 文件夹中数量最少的类别（如“高压泄漏”）。
2. **复制文件**: 将这些少数类的图片文件复制多份，并确保新文件名**保持标签前缀不变**（如 `130_`），但文件名本身唯一（如 `130_original_copy1.jpg`）。

通过这种方式，您可以手动平衡训练数据中各类别的比例，从而提升模型对少数类的识别能力。

##### 4.2.7 如何修改/自定义 VLM 标签

本框架的设计实现了配置与代码的解耦。如果您想修改标签体系（如增加新的泄漏类型、改变检测目标），您**无需修改任何 Python (.py) 脚本**，只需关注以下三个文件：

1. **`data/vlm/label_map.json`**: 定义新的标签结构、每个位置的含义和编码。
2. **`data/vlm/prompt.json`**: 更新 System Prompt，告诉模型新的任务指令和期望的输出格式。
3. **`data/vlm/images/` 文件夹**: 根据新的 `label_map.json` 规则，重命名您的图片文件前缀。

代码会自动适应这些配置文件的变化。

### 5. 配置参数

编辑 `config.yaml` 文件，根据您的硬件和需求调整参数：

#### 5.1 基本配置

```yaml
# 模型相关参数
model_name_or_path: "./Qwen3-1.7B"  # 模型路径
model_type: "llm"  # 模型类型，llm（大语言模型）或 vlm（视觉语言模型）
train_device: "auto"  # 训练时使用的设备，例如 "cuda:0", "cuda:1", "cpu" 或 "auto" (多卡自动分配)
eval_device: "auto"  # 评估时使用的设备，例如 "cuda:0", "cuda:1", "cpu" 或 "auto" (默认使用单GPU)
inference_device: "auto"  # 推理时使用的设备，例如 "cuda:0", "cuda:1", "cpu" 或 "auto" (默认使用单GPU以确保稳定性)
dtype: "float16"  # 数据类型，GPU 上使用 float16，CPU 上使用 float32
train_quantization_bit: null  # 训练时量化位数：4, 8 或 null（如果不使用量化则为 null），用于减少训练时的显存使用
inference_quantization_bit: null  # 推理时量化位数：4, 8 或 null（如果不使用量化则为 null），用于减少推理时的显存使用

# 关于 dtype 和 quantization_bit 的区别
# - dtype: 基础数据类型，指定模型权重和计算时使用的精度（如 float16、float32）
# - quantization_bit: 模型压缩技术，通过 bitsandbytes 库在加载模型时对权重进行量化，进一步减少内存使用
# 两者的关系：
# 1. 即使使用了 quantization_bit，dtype 仍然会影响计算精度和内存使用
# 2. quantization_bit 是在 dtype 的基础上进一步优化内存使用的技术
# 3. 例如：使用 dtype: "float16" + quantization_bit: 8，可以获得比单独使用 float16 更低的内存使用

# 数据相关参数
# - dataset_name: 数据集名称，null 表示使用本地数据。如果设置为 Hugging Face 数据集名称（如 "imdb"），系统会自动下载该数据集
# - dataset_config_name: 数据集配置名称，仅当使用 Hugging Face 数据集时需要设置
# - data_dir: 本地数据目录，当 dataset_name 为 null 时使用，系统会从该目录加载本地数据

# 训练相关参数
output_dir: "./output"  # 输出目录，保存模型权重和评估结果
num_train_epochs: 10  # 训练轮数，增大可提高模型性能但会增加训练时间，通常 3-10 轮
per_device_train_batch_size: 2  # 批次大小，增大可加速训练但会增加内存消耗，建议根据 GPU 内存调整
learning_rate: 5e-4  # 学习率，增大会加速收敛但可能导致不稳定，减小会更稳定但收敛慢，LoRA 微调通常使用 1e-4 到 1e-3

# LoRA 相关参数
lora_r: 8  # LoRA 秩，控制 LoRA 矩阵的秩，增大可提高模型表达能力但会增加内存使用，常用值 4-32
lora_alpha: 16  # LoRA alpha 参数，控制 LoRA 的缩放因子，通常设为 lora_r 的 2 倍。alpha 值越大，LoRA 模块对模型输出的影响越大；值越小，影响越小，更接近原始模型
lora_dropout: 0.1  # LoRA dropout 率，增大可减少过拟合但可能降低模型性能，常用值 0.0-0.2
target_modules: ["q_proj", "v_proj"]
# 目标模块说明：
# - q_proj, k_proj, v_proj: 注意力机制中的 Query, Key, Value 投影层
# - o_proj: 注意力机制的输出投影层
# - gate_proj, up_proj, down_proj: 前馈神经网络 (MLP) 中的投影层
# 推荐配置：
# 1. 显存优先: ["q_proj", "v_proj"]
# 2. 效果优先: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

#### 5.2 根据硬件调整

| 硬件类型 | 推荐配置 | 适用模型 | 内存使用估算 |
|---------|---------|---------|------------|
| **CPU 训练** | `dtype: "float32"`, `per_device_train_batch_size: 1` | 小型模型（如 distilgpt2） | 内存使用取决于模型大小，通常需要 8GB+ 系统内存 |
| **小 GPU (8GB)** | `dtype: "float16"`, `per_device_train_batch_size: 2`, `gradient_accumulation_steps: 4` | 小型 LLM（如 Qwen3-1.7B） | 约 6-7GB GPU 内存 |
| **中 GPU (12GB)** | `dtype: "float16"`, `per_device_train_batch_size: 3`, `gradient_accumulation_steps: 2` | 中型 LLM（如 Qwen3-4B） | 约 10-11GB GPU 内存 |
| **大 GPU (16GB+)** | `dtype: "float16"`, `per_device_train_batch_size: 4`, `gradient_accumulation_steps: 2` | 大型 LLM 或小型 VLM（如 Qwen3-VL-2B-Instruct） | 约 14-15GB GPU 内存 |
| **超大 GPU (24GB+)** | `dtype: "float16"`, `per_device_train_batch_size: 6`, `gradient_accumulation_steps: 1` | 大型 VLM（如 Qwen3-VL-4B-Instruct） | 约 20-22GB GPU 内存 |
| **多卡服务器** | 推荐设置 `train_device: "auto"`，系统会自动启用分布式训练，充分利用多 GPU 资源 | 所有模型 | 总内存 = 单卡内存 × GPU 数量 |

**硬件配置实用建议：**
- **内存估算公式**：对于 LLM，内存使用 ≈ 模型参数 × 2（float16）× 1.5（优化器状态和中间激活值）
- **训练时间预估**：同等配置下，多卡训练时间 ≈ 单卡训练时间 / GPU 数量 × 1.1（通信开销）
- **多卡配置技巧**：
  - 确保所有 GPU 型号相同，以获得最佳性能
  - 对于不同型号的 GPU，系统会自动使用性能最差的 GPU 作为基准
  - 多卡训练时，建议使用 NVLink 或高速网络连接以减少通信开销
- **显存不足解决方案**：
  1. 降低 `per_device_train_batch_size`
  2. 增加 `gradient_accumulation_steps`
  3. 使用 `train_quantization_bit` 进行量化
  4. 选择更小的预训练模型

### 5.3 多卡环境下的设备管理

本项目已针对多 GPU 环境进行了全面优化，确保在各种硬件配置下都能稳定运行。

| 任务类型 | 设备设置 | 行为说明 |
|---------|---------|---------|
| **训练** | `train_device: "auto"` | 自动启用分布式训练，使用所有可用 GPU |
| **评估** | `eval_device: "auto"` | 默认使用单 GPU，确保稳定性 |
| **推理** | `inference_device: "auto"` | 默认使用单 GPU (cuda:0)，避免多卡切分崩溃 |

**分布式训练特性**：
- 使用 `torch.multiprocessing.spawn` 为每个 GPU 启动独立进程
- 自动切换后端：优先 NCCL，不可用时使用 GLOO
- 完善的错误处理和进程同步机制

**手动指定设备**：可在 `config.yaml` 中分别设置 `train_device: "cuda:0"`、`eval_device: "cuda:1"` 等

### 6. 开始训练

```powershell
python main.py --task train
```

### 7. 评估模型

训练完成后，可以评估模型在验证集上的表现。脚本会自动根据 `config.yaml` 中的配置推断 LoRA 模型路径。

```powershell
python main.py --task evaluate
```

评估结果将保存到输出目录下的 `eval_results.json` 文件中。

### 8. 使用模型推理

#### 不同场景的指令选择

以下指令均假设您已正确配置 `config.yaml`，脚本会自动推断 LoRA 模型路径。

| 场景 | 适用情况 | 命令 |
|------|---------|------|
| **基本问答** | 日常简单问题，不需要详细回答 | `python main.py --task inference` |
| **详细回答** | 需要详细解释的问题，如教程、步骤说明等 | `python main.py --task inference --max_new_tokens 4000` |
| **创意回答** | 需要创意性的问题，如故事创作、创意建议等 | `python main.py --task inference --max_new_tokens 2000 --temperature 1.0 --top_p 0.95` |
| **准确回答** | 需要准确信息的问题，如事实性问题、技术细节等 | `python main.py --task inference --max_new_tokens 2000 --temperature 0.1 --top_p 0.8` |
| **平衡设置** | 大多数日常问题，兼顾准确性和自然表达 | `python main.py --task inference --max_new_tokens 2000 --temperature 0.7 --top_p 0.95` |
| **比较模型** | 比较原始模型和微调模型的表现差异 | 使用微调模型：`python main.py --task inference`<br>使用原始模型：`python main.py --task inference --use_original_model` |
| **8bit量化推理** | 显存受限的环境，需要减少内存使用 | `python main.py --task inference --inference_quantization_bit 8` |

## 🎯 LoRA 工作流程详解

### 阶段 1：准备工作（原始模型 + 数据）

1. **原始模型准备**：加载预训练模型（如 Qwen3-1.7B）
2. **数据准备**：准备训练和验证数据，格式为 JSON 数组

### 阶段 2：LoRA 训练流程

1. **配置 LoRA 参数**：设置 lora_r、lora_alpha、target_modules 等参数
2. **加载原始模型**：根据模型类型选择加载方式
   - LLM：使用 `AutoModelForCausalLM.from_pretrained` 加载原始模型
   - VLM：使用 `AutoModelForImageTextToText.from_pretrained` 加载原始模型
3. **创建 LoRA 配置**：使用 `LoraConfig` 创建 LoRA 配置
4. **应用 LoRA 到模型**：使用 `get_peft_model` 将 LoRA 配置应用到原始模型
5. **执行训练**：使用 `Trainer` 执行模型训练
6. **保存 LoRA 模块**：训练完成后，只保存 LoRA 适配器权重（不保存完整模型）

### 阶段 3：LoRA 推理流程（拼装过程）

1. **加载原始模型**：再次加载原始预训练模型
2. **拼装 LoRA 模块**：使用 `PeftModel.from_pretrained` 将 LoRA 模块拼装到原始模型
3. **执行推理**：使用拼装后的模型进行交互式推理

### 核心技术点

1. **低秩分解**：LoRA 使用低秩分解技术，将大矩阵分解为两个小矩阵的乘积，大大减少参数量
2. **即插即用**：通过 `PeftModel` 可以在运行时动态拼装和解拼装 LoRA 模块
3. **参数高效**：只训练和存储 LoRA 相关参数，原始模型权重保持不变
4. **内存高效**：拼装过程不需要修改原始模型权重，只是在计算时应用 LoRA 调整

## 🔗 LoRA 与预训练模型的关系

### LoRA 为什么依赖预训练模型的资源？

许多用户会有这样的疑问："LoRA 不是独立于预训练模型的吗，怎么训练消耗的资源还和预训练模型有关呢？"

这是一个非常重要的问题，理解这一点对于合理配置训练环境至关重要：

#### 1. 预训练模型需要完整加载到内存
- **模型加载**：即使只训练 LoRA 适配器，整个预训练模型仍然需要加载到 GPU 内存中，因为前向传播和反向传播都需要通过完整的模型架构
- **内存占用**：预训练模型的大小直接决定了基础内存占用，例如 4B 参数的模型在 float16 精度下约占 8GB 内存，加上优化器状态和中间激活值，实际内存需求会更大

#### 2. 计算过程依赖预训练模型
- **前向传播**：输入数据需要通过完整的预训练模型进行前向传播，生成中间激活值
- **反向传播**：虽然只更新 LoRA 适配器的参数，但反向传播仍然需要计算整个模型的梯度流
- **批量大小**：预训练模型的大小会限制可使用的批量大小，因为更大的模型需要更多内存来存储中间激活值

#### 3. 精度配置影响内存使用
- **量化技术**：通过对预训练模型进行量化（如 8-bit 或 4-bit 量化），可以显著减少内存使用
- **混合精度训练**：使用 float16 等低精度格式也可以减少内存占用
- **计算效率**：不同精度的计算速度也会影响训练时间

#### 4. 模型架构的影响
- **层数和隐藏维度**：更深、更宽的模型需要更多的计算资源
- **注意力机制**：包含注意力机制的模型（如 Transformer）在计算时内存使用会随序列长度增长而显著增加
- **多模态模型**：像 Qwen3-VL 这样的多模态模型，由于包含视觉编码器，内存需求会更高

### 实际应用建议

- **根据显存选择模型**：如果显存有限，优先选择参数量较小的预训练模型
- **合理使用量化**：通过 `train_quantization_bit` 参数对预训练模型进行量化，减少内存使用
- **调整批量大小**：根据预训练模型的大小和显存情况，调整 `per_device_train_batch_size` 参数
- **监控内存使用**：训练过程中注意监控显存使用情况，及时调整配置

## 💻 命令行参数

您可以通过命令行参数覆盖 `config.yaml` 中的配置，方便快速实验。

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `config.yaml` |
| `--task` | 任务类型，可选值：`train`、`evaluate`、`inference` | (必需参数) |
| `--model_name_or_path` | 模型路径，覆盖配置文件中的设置 | (从 config 读取) |
| `--model_type` | 模型类型，可选值：`llm`、`vlm`，覆盖配置文件中的设置 | (从 config 读取) |
| `--output_dir` | 输出目录，覆盖配置文件中的设置 | (从 config 读取) |
| `--num_train_epochs` | 训练轮数，覆盖配置文件中的设置 | (从 config 读取) |
| `--per_device_train_batch_size` | 批次大小，覆盖配置文件中的设置 | (从 config 读取) |
| `--learning_rate` | 学习率，覆盖配置文件中的设置 | (从 config 读取) |
| `--lora_r` | LoRA 秩，覆盖配置文件中的设置 | (从 config 读取) |
| `--lora_alpha` | LoRA alpha 参数，覆盖配置文件中的设置 | (从 config 读取) |
| `--use_original_model` | 使用原始模型进行推理，不加载 LoRA 权重 | `False` |
| `--max_new_tokens` | 推理时生成的最大 token 数，覆盖配置文件中的设置 | (从 config 读取) |
| `--temperature` | 推理时的温度参数，控制生成文本的随机性，覆盖配置文件中的设置 | (从 config 读取) |
| `--top_p` | 推理时的 top-p 参数，控制生成文本的多样性，覆盖配置文件中的设置 | (从 config 读取) |
| `--repetition_penalty` | 推理时的重复惩罚参数，控制生成文本的重复程度，覆盖配置文件中的设置 | (从 config 读取) |
| `--train_device` | 训练时使用的设备，例如 "cuda:0", "cuda:1", "cpu" 或 "auto" | (从 config 读取) |
| `--eval_device` | 评估时使用的设备，例如 "cuda:0", "cuda:1", "cpu" 或 "auto" | (从 config 读取) |
| `--inference_device` | 推理时使用的设备，例如 "cuda:0", "cuda:1", "cpu" 或 "auto" | (从 config 读取) |

## 📁 输出目录说明

系统会自动根据模型名称创建子目录，以区分不同模型的训练结果。

例如，如果您微调的是 `Qwen3-1.7B`，输出目录结构如下：

```
output/
└── Qwen3-1.7B/          # 自动生成的模型子目录
    ├── checkpoint-X/    # 训练过程中的中间检查点
    ├── lora_model/      # 最终保存的 LoRA 模型（用于推理）
    ├── eval_results.json
    └── trainer_state.json
```

- **checkpoint-X**: 包含完整的训练状态，可用于从中断处恢复训练。
- **lora_model**: 仅包含 LoRA 权重和配置，体积小，用于最终推理。

## 📊 模型精度与内存优化指南

### 精度配置对比

| 配置方案 | 训练精度 | 推理精度 | 内存使用 | 适用场景 |
|---------|---------|---------|---------|---------|
| **标准配置** | `float16` | `float16` | 高 | 显存充足 (24GB+) |
| **FP8 模型** | FP8 + `float16` | FP8 + `float16` | 中 | 平衡性能与内存 |
| **8-bit 量化** | 8-bit | 8-bit | 低 | 显存中等 (16GB) |
| **4-bit 量化** | 4-bit | 4-bit | 极低 | 显存有限 (8GB) |

### 配置参数说明

#### 1. 基础精度 (`dtype`)
- **作用**：控制模型计算精度
- **推荐值**：`float16` (GPU) 或 `float32` (CPU)
- **注意**：即使使用量化，`dtype` 仍影响计算精度

#### 2. 量化配置
- **训练时** (`train_quantization_bit`)：预训练模型量化，LoRA 适配器全精度训练
- **推理时** (`inference_quantization_bit`)：根据部署环境独立配置
- **可选值**：`null`（不量化）、`8`（8-bit）、`4`（4-bit）
- **内存减少**：8-bit 约 50%，4-bit 约 75%

#### 3. FP8 模型
- 使用 FP8 权重，计算精度由 `dtype` 控制
- 可进一步配合量化参数使用

### 快速配置建议

| 显存大小 | 推荐配置 | 量化设置 |
|---------|---------|---------|
| **24GB+** | 标准配置 | `train_quantization_bit: null`<br>`inference_quantization_bit: null` |
| **16GB** | 8-bit 量化 | `train_quantization_bit: 8`<br>`inference_quantization_bit: 8` |
| **8GB** | 4-bit 量化 | `train_quantization_bit: 4`<br>`inference_quantization_bit: 4` |

### 性能与功能影响

| 量化类型 | 性能影响 | 功能影响 | 内存减少 |
|---------|---------|---------|---------|
| **8-bit** | 接近原始精度 | 几乎无影响 | 约 50% |
| **4-bit** | 略有下降 | 复杂任务轻微影响 | 约 75% |
| **FP8** | 接近 8-bit | 功能保留度高 | 约 50% |

### 参数调优指南

| 参数 | 增大效果 | 减小效果 | 推荐范围 |
|------|---------|---------|---------|
| `train_quantization_bit` | 性能↑，内存↑ | 内存↓，稳定性可能↓ | 4-8 |
| `per_device_train_batch_size` | 效率↑，内存↑ | 内存↓，效率↓ | 1-8 |
| `num_train_epochs` | 性能↑，时间↑ | 时间↓，性能可能↓ | 3-10 |
| `lora_r` | 适应能力↑，内存↑ | 内存↓，适应能力↓ | 4-32 |
| `lora_alpha` | LoRA 影响↑，可能过拟合 | 过拟合风险↓，影响↓ | 8-64 |
| `temperature` | 多样性↑，准确性可能↓ | 准确性↑，多样性↓ | 0.1-1.0 |

**最佳实践**：从最高精度开始，根据显存使用情况逐步降低精度，找到性能和内存的最佳平衡点。

## ⚠️ 注意事项

1. **硬件要求**：
   - **CPU 训练**：小型模型（如 distilgpt2）可在 CPU 上运行，但训练速度较慢
   - **GPU 训练**（推荐）：
     - 小型 LLM 模型（如 distilgpt2）：需要至少 4GB GPU 内存
     - 中型 LLM 模型（如 Qwen3-1.7B）：需要至少 8GB GPU 内存
     - 大型 LLM 模型（如 Qwen3-4B）：需要至少 16GB GPU 内存
     - 小型 VLM 模型（如 Qwen3-VL-2B-Instruct）：需要至少 16GB GPU 内存
     - 大型 VLM 模型（如 Qwen3-VL-4B-Instruct）：需要至少 24GB GPU 内存

2. **数据格式**：
   - **LLM 数据格式**：
     - 数据文件应为 JSON 格式
     - 每行包含 `text`（输入文本）和 `target`（目标输出）字段
   - **VLM 数据格式**：
     - 图片格式：支持 JPG、PNG 等常见图片格式
     - 提示词格式：JSON 格式，包含 `system_prompt` 字段
     - 标签映射格式：JSON 格式，键为图片文件名前缀，值为输出结果

3. **训练结果**：
   - 训练完成后，LoRA 权重将保存在 `output/lora_model` 目录
   - 评估结果将保存在 `output/eval_results.json` 文件

4. **推理模式**：
   - 推理时，模型会进入交互式模式。
   - **LLM 模式**：输入文本问题即可获得回答。
   - **VLM 模式**：输入图片路径（如 `./data/vlm/images/000_xxx.jpg`），模型会自动加载 `prompt.json` 中的系统提示词进行分析。
   - 输入 `exit` 退出推理模式。

## 🔧 常见问题排查

### 1. CUDA Out of Memory (OOM)
如果遇到显存不足错误：
- 减小 `per_device_train_batch_size`（例如从 4 减到 2 或 1）。
- 增加 `gradient_accumulation_steps` 以保持总批次大小不变。
- 确保 `dtype` 设置为 `float16`。
- 减小 `max_length`（对于 VLM，图片 token 占用较多，可能需要适当减小文本长度）。

### 2. 训练损失不下降
- 检查学习率 `learning_rate` 是否过大或过小（LoRA 通常使用 1e-4 到 5e-4）。
- 检查数据质量，确保 `target` 字段非空且有意义。
- 增加 `lora_r` 和 `lora_alpha` 尝试提高模型容量。

### 3. VLM 推理报错 "Image not found"
- 确保输入的图片路径是绝对路径或相对于当前工作目录的正确路径。
- 检查图片文件是否存在且可读。

### 4. Windows 下路径问题
- 尽量使用正斜杠 `/` 或双反斜杠 `\\`。
- 确保文件路径中没有特殊字符。



## 📚 示例：LLM 和 VLM 微调

本项目提供了两种开箱即用的微调示例：
- **LLM 示例**：以**日常生活建议**为主题，微调一个问答模型。
- **VLM 示例**：以**工业管道泄漏检测**为主题，微调一个视觉分析模型。

### 示例 1：微调 LLM 日常生活建议模型

本项目已包含一个日常生活建议的示例数据集，您可以直接运行以下命令开始微调：

#### 配置准备

1. 编辑 `config.yaml`，设置 LLM 相关参数：

```yaml
# 模型相关参数
model_name_or_path: "./Qwen3-1.7B"  # 本地语言模型路径
model_type: "llm"  # 模型类型设置为 llm

# LLM 数据文件 (相对于 data_dir 或绝对路径)
train_file: "llm/train.json" # 训练数据文件 (仅 LLM)
validation_file: "llm/validation.json" # 验证数据文件 (仅 LLM)
```

2. 准备好数据文件（参考 4.1 LLM 数据准备）

#### 运行微调

```powershell
# 使用 Qwen3-1.7B 模型微调日常生活建议模型
python main.py --task train

# 微调完成后进行推理
python main.py --task inference

# 使用原始模型进行推理（不加载 LoRA 权重）
python main.py --task inference --use_original_model
```

#### LLM 输入示例：
```
输入文本: 如何保持良好的睡眠质量？
```

#### LLM 输出示例：
```
输出结果: 保持良好的睡眠质量可以尝试以下方法：1. 建立规律的作息时间，每天固定上床和起床时间；2. 睡前避免使用电子设备，因为蓝光会抑制褪黑素分泌；3. 创造舒适的睡眠环境，保持房间安静、黑暗和适宜的温度；4. 睡前避免摄入咖啡因和大量食物；5. 可以尝试睡前放松活动，如阅读、听轻音乐或冥想。
```

### 示例 2：微调 VLM 视觉检测模型

以下是使用 Qwen3-VL-2B-Instruct 微调视觉检测模型的示例：

#### 配置准备

1. 编辑 `config.yaml`，设置 VLM 相关参数：

```yaml
# 模型相关参数
model_name_or_path: "./Qwen3-VL-2B-Instruct"  # 本地视觉语言模型路径
model_type: "vlm"  # 模型类型设置为 vlm

# VLM 数据文件 (相对于 data_dir 或绝对路径)
image_dir: "./data/vlm/images"  # 图片目录 (仅 VLM)
prompt_file: "./data/vlm/prompt.json"  # 提示词文件 (仅 VLM)
label_map_file: "./data/vlm/label_map.json"  # 标签映射文件 (仅 VLM)
```

2. 准备好图片数据、提示词和标签映射文件（参考 4.2 VLM 数据准备）

#### 运行微调

```powershell
# 使用 Qwen3-VL-2B-Instruct 模型微调视觉检测模型
python main.py --task train

# 微调完成后进行推理
python main.py --task inference --max_new_tokens 500 --temperature 0.1 --top_p 0.8
```

#### VLM 输入示例：
```
# 在推理时，输入图片路径进行分析
输入图片路径: ./data/vlm/images/0_20260103_121525_raw.jpg
```

#### VLM 输出示例：
```
输出结果: {
    "status": "No Leakage",
    "sub_type": "Normal",
    "is_night_mode": false,
    "confidence": 1.0,
    "reasoning": "Normal operation."
}
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来帮助改进这个项目！

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [PyTorch](https://pytorch.org/)

---



## 📝 更新日志

> 更新日志按时间倒序排列，最新版本在前。

### 2026.1.27

**代码重构优化**
- 重构 `model_config.py`，提取设备映射、量化配置、模型类型检测等通用逻辑
- `load_lora_model()` 函数从 460 行减少到约 60 行（减少 87%）
- 提升代码可维护性和可读性，保持向后兼容

**评估功能修复**
- 修复单独评估脚本 (`python main.py --task evaluate`) 的设备识别错误
- 统一单独评估和训练完成后评估的方法，确保两者使用完全相同的评估逻辑
- 修复 `load_lora_model()` 中量化加载时的 `device_map` 设置，使用 `device_map="auto"` 与训练时保持一致
- 统一 `TrainingArguments` 配置，确保单独评估时使用与训练时相同的参数设置
- 修复 accelerator 在模型使用 `device_map` 时的设备识别问题

### 2026.1.22

**分布式训练与设备管理优化**
- 支持多 GPU 分布式训练，自动设备分配
- 评估和推理默认单设备，避免多卡崩溃
- 引入 `train_device`、`eval_device`、`inference_device` 独立参数
- 修复 PyTorch 2.9+ API 兼容性问题

### 2026.1.21

- **文档优化**：将 Triton 安装指南移动到“安装依赖”部分，提高文档可读性，防止用户遗漏关键配置步骤
- **多卡推理优化**：修复了在多 GPU 服务器环境下，模型推理时因自动设备分配导致的崩溃问题。现在默认在多卡环境下强制使用单卡加载模型，除非用户明确指定
- **分布式训练支持**：引入了 torch.distributed 和 torch.multiprocessing 技术，支持多 GPU 分布式训练，充分利用多 GPU 资源提高训练速度
- **后端自动切换**：分布式训练时，首先尝试使用 NCCL 后端（性能最佳），如果 NCCL 不可用，自动切换到 GLOO 后端，确保在不同环境下都能正常运行
- **错误处理增强**：完善了分布式训练的错误处理机制，确保在任何情况下都能正确处理异常，避免二次错误
- **设备检测优化**：改进了设备检测逻辑，确保能正确识别和使用可用设备，提高系统兼容性
- **设备控制优化**：移除了统一的 `device` 参数，引入了 `train_device`、`eval_device` 和 `inference_device` 三个独立参数，分别控制训练、评估和推理使用的设备
  - **训练**：`train_device: "auto"` (多卡自动分配)，自动启用分布式训练
  - **评估**：`eval_device: "auto"` (默认使用单GPU)
  - **推理**：`inference_device: "auto"` (默认使用单GPU以确保稳定性)
- **命令行参数增强**：添加了 `--train_device`、`--eval_device` 和 `--inference_device` 命令行参数，支持在命令行中指定不同任务的设备
- **配置文件更新**：在 `config.yaml` 中更新了设备相关配置，添加了新的设备参数并更新了相关注释
- **TF32 精度优化**：修复了 PyTorch 2.9+ 版本中 TF32 控制 API 过时的警告，使用新的 fp32_precision API 替代旧的 allow_tf32 API，确保代码在新版本 PyTorch 上无警告运行

---

<div align="center">
  <p>⭐ 如果这个项目对您有帮助，请给它一个星标！</p>
</div>
