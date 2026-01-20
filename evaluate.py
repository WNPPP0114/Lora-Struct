import torch
from transformers import Trainer, TrainingArguments
from data_processing import load_and_process_data, get_data_collator
from model_config import load_lora_model

def evaluate(config):
    """
    评估模型性能
    Args:
        config: 配置对象，包含评估相关参数
    """
    # 加载和处理数据
    dataset, tokenizer = load_and_process_data(config)
    
    # 加载模型
    model = load_lora_model(config.model_name_or_path, config.lora_model_path)
    
    # 配置评估参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        report_to=config.report_to
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset.get("validation", None),
        tokenizer=tokenizer,
        data_collator=get_data_collator(tokenizer)
    )
    
    # 执行评估
    print("开始评估...")
    eval_result = trainer.evaluate()
    
    # 打印评估结果
    print("评估结果:")
    for key, value in eval_result.items():
        print(f"{key}: {value}")
    
    # 保存评估结果
    import json
    with open(f"{config.output_dir}/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)
    
    print(f"评估结果已保存到 {config.output_dir}/eval_results.json")

if __name__ == "__main__":
    # 导入配置
    from main import Config
    
    # 加载配置
    config = Config("config.yaml")
    
    # 执行评估
    evaluate(config)