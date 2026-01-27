import os
import torch
import traceback
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import Trainer, TrainingArguments, set_seed
from data_processing import load_and_process_data, get_data_collator
from model_config import load_model_with_lora, save_lora_model
from evaluate import run_evaluation

def setup_distributed(rank, world_size):
    """
    设置分布式训练环境
    Args:
        rank: 进程编号
        world_size: 总进程数
    """
    # 只有在多进程训练时才初始化分布式环境
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        # 尝试使用 nccl 后端，如果失败则使用 gloo
        try:
            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        except RuntimeError:
            print(f"NCCL 后端不可用，尝试使用 GLOO 后端")
            dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

def cleanup_distributed():
    """
    清理分布式训练环境
    """
    # 检查分布式进程组是否已经初始化
    if dist.is_initialized():
        dist.destroy_process_group()

def train_distributed(rank, world_size, config):
    """
    分布式训练函数
    Args:
        rank: 进程编号
        world_size: 总进程数
        config: 配置对象，包含训练相关参数
    """
    try:
        # 设置分布式环境
        setup_distributed(rank, world_size)
        
        # 只在主进程打印配置信息
        if rank == 0:
            print("=== 训练配置 ===")
            print(f"模型路径: {config.model_name_or_path}")
            print(f"输出目录: {config.output_dir}")
            print(f"训练轮数: {config.num_train_epochs}")
            print(f"批次大小: {config.per_device_train_batch_size}")
            print(f"学习率: {config.learning_rate}")
            print(f"LoRA 秩: {config.lora_r}")
            print(f"LoRA alpha: {config.lora_alpha}")
            print(f"分布式训练: 启用 (进程数: {world_size})")
            print("===============")
        
        # 设置随机种子
        set_seed(config.seed)
        
        # 设置 TF32 精度
        if torch.cuda.is_available():
            # 使用新 API 控制 TF32 行为 (PyTorch 2.9+ 推荐)
            try:
                # 全局设置
                if hasattr(torch.backends, "fp32_precision"):
                    torch.backends.fp32_precision = "tf32"
                
                # 设置 CUDA matmul 的 TF32 行为
                if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                    torch.backends.cuda.matmul.fp32_precision = "tf32"  # 启用 TF32 加速
                
                # 设置 cuDNN 的 TF32 行为
                if hasattr(torch.backends.cudnn, "fp32_precision"):
                    torch.backends.cudnn.fp32_precision = "tf32"  # 启用 TF32 加速
                elif hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
                    torch.backends.cudnn.conv.fp32_precision = "tf32"  # 启用卷积的 TF32 加速
                elif hasattr(torch.backends.cudnn, "rnn") and hasattr(torch.backends.cudnn.rnn, "fp32_precision"):
                    torch.backends.cudnn.rnn.fp32_precision = "tf32"  # 启用 RNN 的 TF32 加速
            except Exception as e:
                # 兼容旧版本 PyTorch，但不使用会触发警告的 API
                pass

        # 创建输出目录
        if rank == 0:
            os.makedirs(config.output_dir, exist_ok=True)
            print(f"创建输出目录: {config.output_dir}")
        
        # 只在多进程训练时同步
        if world_size > 1:
            dist.barrier()
        
        # 加载和处理数据
        if rank == 0:
            print("加载和处理数据...")
        dataset, tokenizer_or_processor = load_and_process_data(config)
        
        # 处理 tokenizer/processor
        if hasattr(tokenizer_or_processor, "tokenizer"):
            tokenizer = tokenizer_or_processor.tokenizer
            processor = tokenizer_or_processor
        else:
            tokenizer = tokenizer_or_processor
            processor = None

        if rank == 0:
            print(f"训练数据大小: {len(dataset['train'])}")
            print(f"验证数据大小: {len(dataset.get('validation', []))}")
        
        # 加载模型
        if rank == 0:
            print("加载模型...")
        model, model_config = load_model_with_lora(config)
        if rank == 0:
            print("模型加载完成")
        
        # 配置训练参数
        if rank == 0:
            print("配置训练参数...")
        # 确保学习率是浮点数
        learning_rate = float(config.learning_rate) if isinstance(config.learning_rate, str) else config.learning_rate
        
        # 计算总步数
        train_dataset_size = len(dataset["train"])
        total_steps_per_epoch = (train_dataset_size + config.per_device_train_batch_size * world_size - 1) // (config.per_device_train_batch_size * world_size)
        total_steps = total_steps_per_epoch * config.num_train_epochs
        
        # 禁用训练损失日志，只显示评估损失
        # 将 logging_steps 设置为一个非常大的值，确保训练过程中不打印训练损失
        logging_steps = total_steps * 10  # 远大于总步数，确保不触发训练损失日志
        if rank == 0:
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
            max_grad_norm=config.max_grad_norm,
            push_to_hub=config.push_to_hub,
            hub_model_id=config.hub_model_id,
            hub_token=config.hub_token,
            report_to=config.report_to,
            run_name=config.run_name,
            # 分布式训练相关参数
            local_rank=rank,
            deepspeed=None,  # 如需使用 DeepSpeed，可在此配置
        )
        
        # 创建训练器
        if rank == 0:
            print("创建训练器...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation", None),
            processing_class=tokenizer, # 使用 processing_class 替代 tokenizer
            data_collator=get_data_collator(tokenizer_or_processor)
        )
        
        # 开始训练
        if rank == 0:
            print("开始训练...")
        train_result = trainer.train()
        if rank == 0:
            print("训练完成!")
        
        # 只在多进程训练时同步
        if world_size > 1:
            dist.barrier()
        
        # 只在主进程保存模型
        if rank == 0:
            print("保存模型...")
            save_lora_model(model, os.path.join(config.output_dir, "lora_model"))
            if processor:
                print("保存 Processor...")
                processor.save_pretrained(os.path.join(config.output_dir, "lora_model"))
            print("模型保存完成!")
        
        # 只在多进程训练时同步
        if world_size > 1:
            dist.barrier()
        
        # 评估模型（使用统一的评估逻辑）
        if dataset.get("validation", None) and rank == 0:
            run_evaluation(trainer, config, save_results=True)
        
        # 只在主进程保存训练结果
        if rank == 0:
            trainer.save_state()
            print("训练状态保存完成!")
        
        # 清理分布式环境
        cleanup_distributed()
        
    except Exception as e:
        print(f"训练过程中发生错误 (进程 {rank}): {e}")
        print("详细错误信息:")
        traceback.print_exc()
        cleanup_distributed()

def train(config):
    """
    执行模型训练
    Args:
        config: 配置对象，包含训练相关参数
    """
    try:
        # 检查是否启用分布式训练
        if hasattr(config, "distributed_training") and config.distributed_training:
            # 获取可用的 GPU 数量
            world_size = torch.cuda.device_count()
            if world_size > 1:
                print(f"检测到 {world_size} 张 GPU，启用分布式训练")
                # 使用 torch.multiprocessing.spawn 启动多进程训练
                mp.spawn(train_distributed, args=(world_size, config), nprocs=world_size, join=True)
            else:
                print("检测到少于 2 张 GPU，使用单进程训练")
                # 单进程训练
                train_distributed(0, 1, config)
        else:
            # 单进程训练
            print("分布式训练: 禁用，使用单进程训练")
            train_distributed(0, 1, config)
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()

def main():
    """
    主函数，用于测试训练流程
    """
    # 导入配置
    from main import Config
    
    # 加载配置
    config = Config()
    
    # 执行训练
    train(config)

if __name__ == "__main__":
    main()