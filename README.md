# LoRA 大模型微调框架

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.5.1+-red.svg" alt="PyTorch Version">
  <img src="https://img.shields.io/badge/Transformers-4.40.0+-green.svg" alt="Transformers Version">
  <img src="https://img.shields.io/badge/PEFT-0.10.0+-purple.svg" alt="PEFT Version">
</div>

<br>

本项目提供了一个轻量级、高效的 LoRA (Low-Rank Adaptation) 微调框架，用于微调大语言模型 (LLM) 和视觉语言模型 (VLM)。通过参数高效的微调方法，实现对大型模型的快速适配和定制。

## 📋 项目结构

```
├── Qwen3-1.7B/          # 本地预训练模型目录
├── Qwen3-4B/            # 本地预训练模型目录
├── Qwen3-VL-2B-Instruct/ # 本地视觉语言模型目录
├── Qwen3-VL-4B-Instruct/ # 本地视觉语言模型目录
├── data/                # 训练和验证数据
│   ├── train.json       # 训练数据
│   └── validation.json  # 验证数据
├── output/              # 输出目录（包含训练检查点和LoRA模型）
├── config.yaml          # 配置文件
├── main.py              # 主执行脚本
├── data_processing.py   # 数据处理脚本
├── model_config.py      # 模型配置脚本
├── train.py             # 训练脚本
├── evaluate.py          # 评估脚本
├── inference.py         # 推理脚本
└── README.md            # 项目说明文档
```

## 🌟 核心特性

- **参数高效**：LoRA 只训练和存储少量参数（通常是原始模型的 1-3%）
- **内存友好**：支持 FP16 训练，大幅降低内存使用
- **即插即用**：训练生成的 LoRA 模块可灵活拼装到原始模型
- **多模型支持**：兼容 Qwen3、GPT2、DistilGPT2 等多种模型
- **完整流程**：从数据准备到模型训练、评估和推理的全流程支持
- **易于扩展**：模块化设计，便于添加新功能和支持新模型

## 🚀 快速开始

### 1. 环境准备

确保您已经安装了 Python 3.9+ 环境，并且创建了名为 `Lora` 的 Conda 环境：

```powershell
conda create -n Lora python=3.9
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

```powershell
pip install transformers datasets peft pyyaml
```

#### 2.3 验证安装

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

#### 4.1 使用示例数据（快速测试）

项目已包含日常生活建议的示例数据：
- `data/train.json` - 15 个训练样本
- `data/validation.json` - 5 个验证样本

#### 4.2 准备自定义数据

如果您想使用自己的数据：
1. 在 `data/` 目录下创建 `train.json` 和 `validation.json` 文件
2. 数据格式为 JSON 数组，每个元素包含 `text` 和 `target` 字段：

```json
[
    {"text": "如何保持良好的睡眠质量？", "target": "保持良好的睡眠质量可以尝试以下方法：1. 建立规律的作息时间..."}
]
```

### 5. 配置参数

编辑 `config.yaml` 文件，根据您的硬件和需求调整参数：

#### 5.1 基本配置

```yaml
# 模型相关参数
model_name_or_path: "./Qwen3-1.7B"  # 模型路径
torch_dtype: "float16"  # 数据类型，GPU 上使用 float16，CPU 上使用 float32

# 训练相关参数
output_dir: "./output"  # 输出目录，保存模型权重和评估结果
num_train_epochs: 10  # 训练轮数，增大可提高模型性能但会增加训练时间，通常 3-10 轮
per_device_train_batch_size: 2  # 批次大小，增大可加速训练但会增加内存消耗，建议根据 GPU 内存调整
learning_rate: 5e-4  # 学习率，增大会加速收敛但可能导致不稳定，减小会更稳定但收敛慢，LoRA 微调通常使用 1e-4 到 1e-3

# LoRA 相关参数
lora_r: 8  # LoRA 秩，控制 LoRA 矩阵的秩，增大可提高模型表达能力但会增加内存使用，常用值 4-32
lora_alpha: 16  # LoRA alpha 参数，控制 LoRA 的缩放因子，通常设为 lora_r 的 2 倍。alpha 值越大，LoRA 模块对模型输出的影响越大；值越小，影响越小，更接近原始模型
lora_dropout: 0.1  # LoRA dropout 率，增大可减少过拟合但可能降低模型性能，常用值 0.0-0.2
target_modules: ["q_proj", "v_proj"]  # 目标模块，选择要应用 LoRA 的注意力机制模块，通常包括 q_proj 和 v_proj
```

#### 5.2 根据硬件调整

| 硬件类型 | 推荐配置 |
|---------|---------|
| **CPU 训练** | `torch_dtype: "float32"`, `per_device_train_batch_size: 1` |
| **小 GPU (8GB)** | `torch_dtype: "float16"`, `per_device_train_batch_size: 2`, `gradient_accumulation_steps: 4` |
| **大 GPU (16GB+)** | `torch_dtype: "float16"`, `per_device_train_batch_size: 4`, `gradient_accumulation_steps: 2` |

### 6. 开始训练

```powershell
python main.py --task train
```

### 7. 评估模型

```powershell
python main.py --task evaluate --lora_model_path ./output/lora_model
```

### 8. 使用模型推理

#### 不同场景的指令选择

| 场景 | 适用情况 | 命令 |
|------|---------|------|
| **基本问答** | 日常简单问题，不需要详细回答 | `python main.py --task inference --lora_model_path ./output/lora_model` |
| **详细回答** | 需要详细解释的问题，如教程、步骤说明等 | `python main.py --task inference --lora_model_path ./output/lora_model --max_new_tokens 2000` |
| **创意回答** | 需要创意性的问题，如故事创作、创意建议等 | `python main.py --task inference --lora_model_path ./output/lora_model --max_new_tokens 1000 --temperature 1.0 --top_p 0.95` |
| **准确回答** | 需要准确信息的问题，如事实性问题、技术细节等 | `python main.py --task inference --lora_model_path ./output/lora_model --max_new_tokens 1000 --temperature 0.1 --top_p 0.8` |
| **平衡设置** | 大多数日常问题，兼顾准确性和自然表达 | `python main.py --task inference --lora_model_path ./output/lora_model --max_new_tokens 1000 --temperature 0.7 --top_p 0.95` |
| **比较模型** | 比较原始模型和微调模型的表现差异 | 使用微调模型：`python main.py --task inference --lora_model_path ./output/lora_model`<br>使用原始模型：`python main.py --task inference --use_original_model` |

## 🎯 LoRA 工作流程详解

### 阶段 1：准备工作（原始模型 + 数据）

1. **原始模型准备**：加载预训练模型（如 Qwen3-1.7B）
2. **数据准备**：准备训练和验证数据，格式为 JSON 数组

### 阶段 2：LoRA 训练流程

1. **配置 LoRA 参数**：设置 lora_r、lora_alpha、target_modules 等参数
2. **加载原始模型**：使用 `AutoModelForCausalLM.from_pretrained` 加载原始模型
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

## 📊 命令行参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `config.yaml` |
| `--task` | 任务类型，可选值：`train`、`evaluate`、`inference` | - |
| `--model_name_or_path` | 模型路径，覆盖配置文件中的设置 | - |
| `--output_dir` | 输出目录，覆盖配置文件中的设置 | - |
| `--num_train_epochs` | 训练轮数，覆盖配置文件中的设置 | - |
| `--per_device_train_batch_size` | 批次大小，覆盖配置文件中的设置 | - |
| `--learning_rate` | 学习率，覆盖配置文件中的设置 | - |
| `--lora_r` | LoRA 秩，覆盖配置文件中的设置 | - |
| `--lora_alpha` | LoRA alpha 参数，覆盖配置文件中的设置 | - |
| `--lora_model_path` | LoRA 模型路径，用于推理，覆盖配置文件中的设置 | - |
| `--use_original_model` | 使用原始模型进行推理，不加载 LoRA 权重 | - |
| `--max_new_tokens` | 推理时生成的最大 token 数，覆盖配置文件中的设置 | - |
| `--temperature` | 推理时的温度参数，控制生成文本的随机性，覆盖配置文件中的设置 | - |
| `--top_p` | 推理时的 top-p 参数，控制生成文本的多样性，覆盖配置文件中的设置 | - |
| `--repetition_penalty` | 推理时的重复惩罚参数，控制生成文本的重复程度，覆盖配置文件中的设置 | - |

## ⚠️ 注意事项

1. **硬件要求**：
   - **CPU 训练**：小型模型（如 distilgpt2）可在 CPU 上运行，但训练速度较慢
   - **GPU 训练**（推荐）：
     - 小型模型（如 distilgpt2）：需要至少 4GB GPU 内存
     - 中型模型（如 Qwen3-1.7B）：需要至少 8GB GPU 内存
     - 大型模型（如 Qwen3-4B）：需要至少 16GB GPU 内存
     - 视觉语言模型（如 Qwen3-VL-4B-Instruct）：需要至少 24GB GPU 内存

2. **数据格式**：
   - 数据文件应为 JSON 格式
   - 每行包含 `text`（输入文本）和 `target`（目标输出）字段

3. **训练结果**：
   - 训练完成后，LoRA 权重将保存在 `output/lora_model` 目录
   - 评估结果将保存在 `output/eval_results.json` 文件

4. **推理模式**：
   - 推理时，模型会进入交互式模式，输入问题即可获得回答
   - 输入 `exit` 退出推理模式
   - 模型会在回答完成后自动停止生成，不会一直生成长度到最大限制
   - 系统会自动清理输出文本中的重复内容，确保回答更加简洁

## 📚 示例：微调日常生活建议模型

本项目已包含一个日常生活建议的示例数据集，您可以直接运行以下命令开始微调：

```powershell
# 使用 Qwen3-1.7B 模型微调
python main.py --task train

# 微调完成后进行推理
python main.py --task inference --lora_model_path ./output/lora_model --max_new_tokens 1000 --temperature 0.7 --top_p 0.95

# 使用原始模型进行推理（不加载 LoRA 权重）
python main.py --task inference --use_original_model --max_new_tokens 1000 --temperature 0.7 --top_p 0.95
```

#### 输入示例：
```
输入文本: 如何保持良好的睡眠质量？
```

#### 输出示例：
```
输出结果: 保持良好的睡眠质量可以尝试以下方法：1. 建立规律的作息时间，每天固定上床和起床时间；2. 睡前避免使用电子设备，因为蓝光会抑制褪黑素分泌；3. 创造舒适的睡眠环境，保持房间安静、黑暗和适宜的温度；4. 睡前避免摄入咖啡因和大量食物；5. 可以尝试睡前放松活动，如阅读、听轻音乐或冥想。
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

<div align="center">
  <p>⭐ 如果这个项目对您有帮助，请给它一个星标！</p>
</div>