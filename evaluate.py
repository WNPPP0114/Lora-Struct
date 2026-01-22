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
    # 确定评估设备
    eval_device = config.eval_device
    print(f"评估使用设备: {eval_device}")
    
    # 设置 CUDA_VISIBLE_DEVICES（如果需要）
    if eval_device and eval_device != "auto":
        if eval_device.startswith("cuda:"):
            try:
                device_id = eval_device.split(":")[1]
                import os
                os.environ["CUDA_VISIBLE_DEVICES"] = device_id
                print(f"根据评估设备设置 CUDA_VISIBLE_DEVICES={device_id}")
            except IndexError:
                print(f"警告: 无法解析评估设备 ID: {eval_device}")
        elif eval_device == "cpu":
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            print("根据评估设备设置使用 CPU")
    
    # 加载和处理数据
    dataset, tokenizer = load_and_process_data(config)
    
    # 加载模型
    model = load_lora_model(config.model_name_or_path, config.lora_model_path, config)
    
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
        processing_class=tokenizer,
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