import os
import json
import torch
import traceback
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def load_and_process_data(config):
    """
    加载和处理数据
    Args:
        config: 配置对象，包含数据处理相关参数
    Returns:
        processed_dataset: 处理后的数据集
        tokenizer: 分词器
    """
    try:
        # 加载分词器
        print(f"加载分词器: {config.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        # 设置 pad_token（如果没有）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"设置 pad_token 为 eos_token: {tokenizer.eos_token}")
        print("分词器加载完成")
        
        # 根据配置加载数据集
        if config.dataset_name:
            # 从 Hugging Face Hub 加载数据集
            print(f"从 Hugging Face Hub 加载数据集: {config.dataset_name}")
            dataset = load_dataset(config.dataset_name, config.dataset_config_name)
        else:
            # 从本地文件加载数据集
            print(f"从本地文件加载数据集: {config.data_dir}")
            dataset = load_local_dataset(config.data_dir, config.file_format)
        
        print(f"数据集加载完成，包含以下 split: {list(dataset.keys())}")
        
        # 处理数据集
        print("处理数据集...")
        processed_dataset = dataset.map(
            lambda examples: process_function(examples, tokenizer, config),
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        print("数据集处理完成")
        
        # 设置数据集格式
        print("设置数据集格式...")
        processed_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        print("数据集格式设置完成")
        
        return processed_dataset, tokenizer
    except Exception as e:
        print(f"数据处理过程中发生错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        raise

def load_local_dataset(data_dir, file_format):
    """
    从本地文件加载数据集
    Args:
        data_dir: 数据目录
        file_format: 文件格式，支持 json, csv, txt 等
    Returns:
        dataset: 加载的数据集
    """
    try:
        files = {}
        for split in ["train", "validation", "test"]:
            file_path = os.path.join(data_dir, f"{split}.{file_format}")
            if os.path.exists(file_path):
                files[split] = file_path
                print(f"找到文件: {file_path}")
        
        if not files:
            raise ValueError(f"在 {data_dir} 中没有找到任何 {file_format} 文件")
        
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
            print(f"使用文本列: {config.text_column}, 目标列: {config.target_column}")
            # 检查 prompt_template 是否包含关键字参数
            if "{text}" in config.prompt_template:
                texts = [f"{config.prompt_template.format(text=text)} {target}" 
                         for text, target in zip(examples[config.text_column], examples[config.target_column])]
            else:
                # 使用位置参数
                texts = [f"{config.prompt_template.format(text)} {target}" 
                         for text, target in zip(examples[config.text_column], examples[config.target_column])]
        elif config.text_column:
            print(f"使用文本列: {config.text_column}")
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

def get_data_collator(tokenizer):
    """
    获取数据整理器
    Args:
        tokenizer: 分词器
    Returns:
        data_collator: 数据整理器
    """
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