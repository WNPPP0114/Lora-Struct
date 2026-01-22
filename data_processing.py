import os
import json
import torch
import traceback
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoProcessor, ProcessorMixin
from PIL import Image

def load_and_process_data(config):
    """
    加载和处理数据
    Args:
        config: 配置对象，包含数据处理相关参数
    Returns:
        processed_dataset: 处理后的数据集
        tokenizer_or_processor: 分词器或处理器
    """
    try:
        # 检查是否在分布式训练环境中
        is_distributed = False
        if os.environ.get('RANK') is not None:
            is_distributed = True
            rank = int(os.environ['RANK'])
        else:
            rank = 0
        
        # 只在主进程打印详细日志
        def print_dist(msg):
            if rank == 0:
                print(msg)
        
        # 检查是否是 VLM
        if getattr(config, "model_type", "llm") == "vlm":
            return load_and_process_vlm_data(config)

        # 加载分词器
        print_dist(f"加载分词器: {config.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        # 设置 pad_token（如果没有）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print_dist(f"设置 pad_token 为 eos_token: {tokenizer.eos_token}")
        print_dist("分词器加载完成")
        
        # 根据配置加载数据集
        if config.dataset_name:
            # 从 Hugging Face Hub 加载数据集
            print_dist(f"从 Hugging Face Hub 加载数据集: {config.dataset_name}")
            dataset = load_dataset(config.dataset_name, config.dataset_config_name)
        else:
            # 从本地文件加载数据集
            print_dist(f"从本地文件加载数据集: {config.data_dir}")
            dataset = load_local_dataset(config)
        
        print_dist(f"数据集加载完成，包含以下 split: {list(dataset.keys())}")
        
        # 处理数据集
        print_dist("处理数据集...")
        processed_dataset = dataset.map(
            lambda examples: process_function(examples, tokenizer, config),
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        print_dist("数据集处理完成")
        
        # 设置数据集格式
        print_dist("设置数据集格式...")
        processed_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        print_dist("数据集格式设置完成")
        
        return processed_dataset, tokenizer
    except Exception as e:
        print(f"数据处理过程中发生错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        raise

def parse_label(label_key, label_map):
    """
    解析标签键，支持简单的 Key-Value 映射和位置编码映射
    Args:
        label_key: 文件名前缀，如 "0" 或 "010"
        label_map: 标签映射字典
    Returns:
        label_data: 解析后的标签数据字典
    """
    # 检查是否是位置编码模式
    if label_map.get("mapping_type") == "positional":
        positions = label_map.get("positions", [])
        defaults = label_map.get("defaults", {})
        
        if len(label_key) != len(positions):
            raise ValueError(f"Label key length ({len(label_key)}) does not match positions length ({len(positions)})")
            
        result = defaults.copy()
        
        for i, char in enumerate(label_key):
            position_config = positions[i]
            field_name = position_config["field"]
            mapping = position_config["map"]
            
            if char not in mapping:
                raise ValueError(f"Invalid character '{char}' at position {i} for field '{field_name}'")
                
            result[field_name] = mapping[char]
            
        return result
    else:
        # 传统的 Key-Value 映射
        if label_key not in label_map:
            raise ValueError(f"Label key '{label_key}' not found in label map")
        return label_map[label_key]

def load_and_process_vlm_data(config):
    """
    加载和处理 VLM 数据
    """
    try:
        # 检查是否在分布式训练环境中
        is_distributed = False
        if os.environ.get('RANK') is not None:
            is_distributed = True
            rank = int(os.environ['RANK'])
        else:
            rank = 0
        
        # 只在主进程打印详细日志
        def print_dist(msg):
            if rank == 0:
                print(msg)
        
        print_dist(f"加载 Processor: {config.model_name_or_path}")
        processor = AutoProcessor.from_pretrained(config.model_name_or_path)
        print_dist("Processor 加载完成")

        # 加载 Prompt
        if not os.path.exists(config.prompt_file):
            raise ValueError(f"Prompt 文件不存在: {config.prompt_file}")
            
        # 尝试作为 JSON 加载
        try:
            with open(config.prompt_file, "r", encoding="utf-8") as f:
                prompt_data = json.load(f)
                if isinstance(prompt_data, dict) and "system_prompt" in prompt_data:
                    system_prompt = prompt_data["system_prompt"]
                else:
                    # 如果不是预期的 JSON 结构，回退到读取整个文件
                    # 重新读取文件指针
                    f.seek(0)
                    system_prompt = f.read()
        except json.JSONDecodeError:
            # 如果不是 JSON，作为普通文本读取
            with open(config.prompt_file, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        
        # 加载标签映射
        if not os.path.exists(config.label_map_file):
            raise ValueError(f"标签映射文件不存在: {config.label_map_file}")
            
        with open(config.label_map_file, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        
        # 扫描图片目录
        if not os.path.exists(config.image_dir):
            raise ValueError(f"图片目录不存在: {config.image_dir}")
            
        image_files = [f for f in os.listdir(config.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print_dist(f"找到 {len(image_files)} 张图片")
        
        if not image_files:
            raise ValueError(f"在 {config.image_dir} 中没有找到任何图片")

        data = []
        for img_file in image_files:
            label_key = img_file.split("_")[0]
            
            try:
                label_data = parse_label(label_key, label_map)
            except ValueError as e:
                print(f"Warning: {e}. Skipping {img_file}")
                continue
            
            data.append({
                "image_path": os.path.join(config.image_dir, img_file),
                "system_prompt": system_prompt,
                "output": json.dumps(label_data, ensure_ascii=False)
            })
            
        # 创建数据集
        # 使用分层随机划分，确保每个类别都按比例划分到训练集和验证集
        from collections import defaultdict
        import random

        # 按标签分组
        data_by_label = defaultdict(list)
        for item in data:
            # 从 image_path 中提取文件名，再提取标签
            filename = os.path.basename(item["image_path"])
            label_key = filename.split("_")[0]
            data_by_label[label_key].append(item)

        train_data = []
        val_data = []

        for label_key, items in data_by_label.items():
            # 随机打乱每个类别的数据
            random.shuffle(items)
            
            # 计算切分点
            split_idx = int(len(items) * 0.9)
            
            # 如果某个类别只有一个样本，则只放入训练集
            if len(items) == 1:
                train_data.extend(items)
            else:
                # 如果切分后验证集为空，至少放一个样本到验证集
                if split_idx == len(items):
                    split_idx -= 1
                
                train_data.extend(items[:split_idx])
                val_data.extend(items[split_idx:])

        # 再次打乱整个训练集和验证集
        random.shuffle(train_data)
        random.shuffle(val_data)
        
        dataset_dict = {"train": Dataset.from_list(train_data)}
        if val_data:
            dataset_dict["validation"] = Dataset.from_list(val_data)
            
        dataset = DatasetDict(dataset_dict)
        
        print_dist("处理 VLM 数据集...")
        # 注意：这里 batched=False，因为我们需要逐个加载图片
        processed_dataset = dataset.map(
            lambda example: process_vlm_function(example, processor, config),
            batched=False,
            remove_columns=dataset["train"].column_names
        )
        print_dist("数据集处理完成")
        
        # 设置格式
        # 动态获取列名，因为不同 VLM 的 processor 输出可能不同
        columns = [col for col in processed_dataset["train"].column_names if col != "token_type_ids"]
        processed_dataset.set_format(type="torch", columns=columns)
        
        return processed_dataset, processor
        
    except Exception as e:
        print(f"VLM 数据处理过程中发生错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        raise

def process_vlm_function(example, processor, config):
    """
    处理单个 VLM 样本
    """
    try:
        image_path = example["image_path"]
        system_prompt = example["system_prompt"]
        output = example["output"]
        
        # 构建对话
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": output}],
            }
        ]
        
        # 准备文本
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        
        # 处理
        # 不进行 padding，留给 collator 处理
        inputs = processor(
            text=[text],
            images=[image],
            padding=False,
            return_tensors="pt",
        )
        
        # 去掉 batch 维度
        result = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # 创建 labels
        # 简单的将 input_ids 作为 labels
        result["labels"] = result["input_ids"].clone()
        
        return result
    except Exception as e:
        print(f"处理 VLM 样本时发生错误: {e}")
        traceback.print_exc()
        raise

def load_local_dataset(config):
    """
    从本地文件加载数据集
    Args:
        config: 配置对象
    Returns:
        dataset: 加载的数据集
    """
    try:
        data_dir = config.data_dir
        file_format = config.file_format
        
        files = {}
        
        # 检查训练文件
        train_file = getattr(config, "train_file", "llm_train.json")
        train_path = os.path.join(data_dir, train_file)
        if os.path.exists(train_path):
            files["train"] = train_path
            print(f"找到训练文件: {train_path}")
        else:
            # 尝试旧的命名方式作为后备
            old_train_path = os.path.join(data_dir, f"train.{file_format}")
            if os.path.exists(old_train_path):
                files["train"] = old_train_path
                print(f"找到训练文件 (旧命名): {old_train_path}")

        # 检查验证文件
        validation_file = getattr(config, "validation_file", "llm_validation.json")
        val_path = os.path.join(data_dir, validation_file)
        if os.path.exists(val_path):
            files["validation"] = val_path
            print(f"找到验证文件: {val_path}")
        else:
            # 尝试旧的命名方式作为后备
            old_val_path = os.path.join(data_dir, f"validation.{file_format}")
            if os.path.exists(old_val_path):
                files["validation"] = old_val_path
                print(f"找到验证文件 (旧命名): {old_val_path}")
        
        if not files:
            raise ValueError(f"在 {data_dir} 中没有找到任何训练或验证文件")
        
        if file_format == "json":
            dataset = load_dataset("json", data_files=files)
        elif file_format == "csv":
            dataset = load_dataset("csv", data_files=files)
        elif file_format == "txt":
            dataset = load_dataset("text", data_files=files)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        return dataset
    except Exception as e:
        print(f"加载本地数据集时发生错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        raise

def process_function(examples, tokenizer, config):
    """
    处理单个样本
    Args:
        examples: 样本字典
        tokenizer: 分词器
        config: 配置对象
    Returns:
        processed_examples: 处理后的样本
    """
    try:
        # 构建输入文本
        if config.text_column and config.target_column:
            # 检查 prompt_template 是否包含关键字参数
            if "{text}" in config.prompt_template:
                texts = [f"{config.prompt_template.format(text=text)} {target}" 
                         for text, target in zip(examples[config.text_column], examples[config.target_column])]
            else:
                # 使用位置参数
                texts = [f"{config.prompt_template.format(text)} {target}" 
                         for text, target in zip(examples[config.text_column], examples[config.target_column])]
        elif config.text_column:
            texts = examples[config.text_column]
        else:
            raise ValueError("Either text_column or both text_column and target_column must be specified")
        
        # 分词
        tokenized = tokenizer(
            texts,
            max_length=config.max_length,
            truncation=True,
            padding="max_length"
        )
        
        # 设置标签
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    except Exception as e:
        print(f"处理样本时发生错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        raise

def get_data_collator(tokenizer_or_processor):
    """
    获取数据整理器
    Args:
        tokenizer_or_processor: 分词器或处理器
    Returns:
        data_collator: 数据整理器
    """
    # 如果是 Processor (VLM)
    if hasattr(tokenizer_or_processor, "image_processor") or isinstance(tokenizer_or_processor, ProcessorMixin):
        return get_vlm_data_collator(tokenizer_or_processor)
        
    # 否则假设是 Tokenizer (LLM)
    tokenizer = tokenizer_or_processor
    def data_collator(examples):
        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        labels = torch.stack([example["labels"] for example in examples])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    return data_collator

def get_vlm_data_collator(processor):
    def data_collator(features):
        # 使用 processor 的 pad 方法
        # features 是 list of dict
        # processor.pad 接受 list of dict
        # 注意：Qwen2-VL 的 processor.pad 可能需要特定的参数
        
        # 尝试直接使用 processor.pad
        try:
            batch = processor.pad(features, padding=True, return_tensors="pt")
        except Exception:
            # 如果 processor.pad 失败，尝试手动处理
            # 这里假设 input_ids 需要 padding
            # pixel_values 可能需要特殊处理
            
            # 这是一个简化的 fallback，可能不适用于所有 VLM
            from torch.nn.utils.rnn import pad_sequence
            
            input_ids = [f["input_ids"] for f in features]
            attention_mask = [f["attention_mask"] for f in features]
            labels = [f["labels"] for f in features]
            
            # Pad input_ids
            input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
            attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
            labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
            
            batch = {
                "input_ids": input_ids_padded,
                "attention_mask": attention_mask_padded,
                "labels": labels_padded
            }
            
            # 处理 pixel_values
            if "pixel_values" in features[0]:
                pixel_values = [f["pixel_values"] for f in features]
                # 如果 pixel_values 形状一致，直接 stack
                try:
                    batch["pixel_values"] = torch.stack(pixel_values)
                except:
                    # 如果形状不一致（动态分辨率），可能需要 flatten 并 pad，或者保持 list
                    # Qwen2-VL 通常处理为 list 或 flatten
                    batch["pixel_values"] = pixel_values
            
            if "image_grid_thw" in features[0]:
                batch["image_grid_thw"] = torch.stack([f["image_grid_thw"] for f in features])
                
        return batch
    return data_collator
