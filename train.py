import os
import torch
import traceback
from transformers import Trainer, TrainingArguments, set_seed
from data_processing import load_and_process_data, get_data_collator
from model_config import load_model_with_lora, save_lora_model

def train(config):
    """
    执行模型训练
    Args:
        config: 配置对象，包含训练相关参数
    """
    try:
        # 打印配置信息
        print("=== 训练配置 ===")
        print(f"模型路径: {config.model_name_or_path}")
        print(f"输出目录: {config.output_dir}")
        print(f"训练轮数: {config.num_train_epochs}")
        print(f"批次大小: {config.per_device_train_batch_size}")
        print(f"学习率: {config.learning_rate}")
        print(f"LoRA 秩: {config.lora_r}")
        print(f"LoRA alpha: {config.lora_alpha}")
        print("===============")
        
        # 设置随机种子
        set_seed(config.seed)
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"创建输出目录: {config.output_dir}")
        
        # 加载和处理数据
        print("加载和处理数据...")
        dataset, tokenizer = load_and_process_data(config)
        print(f"训练数据大小: {len(dataset['train'])}")
        print(f"验证数据大小: {len(dataset.get('validation', []))}")
        
        # 加载模型
        print("加载模型...")
        model, model_config = load_model_with_lora(config)
        print("模型加载完成")
        
        # 配置训练参数
        print("配置训练参数...")
        # 确保学习率是浮点数
        learning_rate = float(config.learning_rate) if isinstance(config.learning_rate, str) else config.learning_rate
        
        # 计算总步数
        train_dataset_size = len(dataset["train"])
        total_steps_per_epoch = (train_dataset_size + config.per_device_train_batch_size - 1) // config.per_device_train_batch_size
        total_steps = total_steps_per_epoch * config.num_train_epochs
        
        # 禁用训练损失日志，只显示评估损失
        # 将 logging_steps 设置为一个非常大的值，确保训练过程中不打印训练损失
        logging_steps = total_steps * 10  # 远大于总步数，确保不触发训练损失日志
        print(f"计算得到的 logging_steps: {logging_steps}")
        print("禁用训练损失日志，只显示评估损失")
        
        # 确保评估时也使用整数 epoch
        # 将评估策略设置为 "epoch"，确保评估只在整数 epoch 结束时进行
        config.evaluation_strategy = "epoch"
        config.save_strategy = "epoch"
        
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
            logging_steps=logging_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_steps=1,
            eval_steps=1,
            load_best_model_at_end=config.load_best_model_at_end,
            metric_for_best_model=config.metric_for_best_model,
            fp16=config.fp16,
            fp16_opt_level=config.fp16_opt_level,
            bf16=config.bf16,
            tf32=config.tf32,
            max_grad_norm=config.max_grad_norm,
            push_to_hub=config.push_to_hub,
            hub_model_id=config.hub_model_id,
            hub_token=config.hub_token,
            report_to=config.report_to,
            run_name=config.run_name
        )
        
        # 创建训练器
        print("创建训练器...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation", None),
            tokenizer=tokenizer,
            data_collator=get_data_collator(tokenizer)
        )
        
        # 开始训练
        print("开始训练...")
        train_result = trainer.train()
        print("训练完成!")
        
        # 保存模型
        print("保存模型...")
        save_lora_model(model, os.path.join(config.output_dir, "lora_model"))
        print("模型保存完成!")
        
        # 评估模型
        if dataset.get("validation", None):
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
        
        # 保存训练结果
        trainer.save_state()
        print("训练状态保存完成!")
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()

def main():
    """
    主函数，用于测试训练流程
    """
    # 导入配置
    from config import Config
    
    # 加载配置
    config = Config()
    
    # 执行训练
    train(config)

if __name__ == "__main__":
    main()