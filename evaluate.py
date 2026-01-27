import torch
import os
import json
from transformers import Trainer, TrainingArguments, set_seed
from data_processing import load_and_process_data, get_data_collator
from model_config import load_lora_model

def run_evaluation(trainer, config, save_results=True):
    """
    执行评估并保存结果（统一的评估逻辑）
    Args:
        trainer: Trainer 对象
        config: 配置对象
        save_results: 是否保存评估结果到文件
    Returns:
        eval_result: 评估结果字典
    """
    print("开始评估...")
    eval_result = trainer.evaluate()
    
    # 打印评估结果
    print("评估结果:")
    for key, value in eval_result.items():
        print(f"{key}: {value}")
    
    # 保存评估结果
    if save_results:
        os.makedirs(config.output_dir, exist_ok=True)
        with open(f"{config.output_dir}/eval_results.json", "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=2, ensure_ascii=False)
        print(f"评估结果已保存到 {config.output_dir}/eval_results.json")
    
    return eval_result

def evaluate(config):
    """
    评估模型性能
    Args:
        config: 配置对象，包含评估相关参数
    """
    # 设置随机种子，和训练时一致
    set_seed(config.seed)
    
    # 确定评估设备
    eval_device = config.eval_device
    print(f"评估使用设备: {eval_device}")
    
    # 设置 CUDA_VISIBLE_DEVICES（如果需要）
    if eval_device and eval_device != "auto":
        if eval_device.startswith("cuda:"):
            try:
                device_id = eval_device.split(":")[1]
                os.environ["CUDA_VISIBLE_DEVICES"] = device_id
                print(f"根据评估设备设置 CUDA_VISIBLE_DEVICES={device_id}")
            except IndexError:
                print(f"警告: 无法解析评估设备 ID: {eval_device}")
        elif eval_device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            print("根据评估设备设置使用 CPU")
    
    # 加载和处理数据
    dataset, tokenizer_or_processor = load_and_process_data(config)
    
    # 处理 tokenizer/processor
    if hasattr(tokenizer_or_processor, "tokenizer"):
        tokenizer = tokenizer_or_processor.tokenizer
        processor = tokenizer_or_processor
    else:
        tokenizer = tokenizer_or_processor
        processor = None
    
    # 加载模型 - 使用 load_lora_model 加载已训练的 LoRA 模型
    model = load_lora_model(config.model_name_or_path, config.lora_model_path, config)
    
    # 检查模型是否使用了 device_map（量化加载时）
    model_uses_device_map = hasattr(model, "device_map") and model.device_map is not None
    if model_uses_device_map:
        print(f"检测到模型使用 device_map: {model.device_map}")
    
    # 如果使用 device_map，确保 CUDA 设备已正确初始化
    if model_uses_device_map and torch.cuda.is_available():
        # 确保默认 CUDA 设备已设置
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        # 设置默认设备为 cuda:0（对于使用 device_map 的模型）
        torch.cuda.set_device(0)
    
    # 配置评估参数 - 使用和训练时完全相同的 TrainingArguments 配置
    # 确保学习率是浮点数
    learning_rate = float(config.learning_rate) if isinstance(config.learning_rate, str) else config.learning_rate
    
    # 使用和训练时完全相同的 TrainingArguments 配置
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_steps=1,
        eval_steps=1,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        fp16=config.fp16,
        fp16_opt_level=config.fp16_opt_level,
        bf16=config.bf16,
        max_grad_norm=config.max_grad_norm,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        hub_token=config.hub_token,
        report_to=config.report_to,
        run_name=config.run_name,
        # 分布式训练相关参数 - 单独评估时使用 local_rank=0（单进程，和训练时一致）
        local_rank=0,
        deepspeed=None,
        # 如果模型使用了 device_map，禁用 pin_memory 以避免设备冲突
        dataloader_pin_memory=False if model_uses_device_map else True,
    )
    
    # 创建训练器 - 使用和训练时完全相同的配置
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.get("train", None),  # 评估时不需要训练数据，但为了保持一致性可以传入
        eval_dataset=dataset.get("validation", None),
        processing_class=tokenizer,  # 使用 processing_class 替代 tokenizer
        data_collator=get_data_collator(tokenizer_or_processor)
    )
    
    # 执行评估（使用统一的评估逻辑）
    run_evaluation(trainer, config, save_results=True)

if __name__ == "__main__":
    # 导入配置
    from main import Config
    
    # 加载配置
    config = Config("config.yaml")
    
    # 执行评估
    evaluate(config)