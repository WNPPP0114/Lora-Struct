import argparse
import yaml
import os
from train import train
from evaluate import evaluate
from inference import inference

class Config:
    """
    配置类，用于加载和管理配置参数
    """
    def __init__(self, config_file=None):
        """
        初始化配置
        Args:
            config_file: 配置文件路径
        """
        # 初始化默认参数
        self.initialize_defaults()
        
        # 从配置文件加载参数
        if config_file:
            self.load_from_file(config_file)
    
    def initialize_defaults(self):
        """
        初始化默认参数
        """
        # 模型相关参数
        self.model_name_or_path = "facebook/opt-125m"
        self.train_device = "auto"  # 训练时使用的设备，例如 "cuda:0", "cuda:1", "cpu" 或 "auto"
        self.eval_device = "auto"  # 评估时使用的设备，例如 "cuda:0", "cuda:1", "cpu" 或 "auto"
        self.inference_device = "auto"  # 推理时使用的设备，例如 "cuda:0", "cuda:1", "cpu" 或 "auto"
        self.dtype = "float16"
        self.train_quantization_bit = None  # 训练时量化位数: 4, 8 or None
        self.inference_quantization_bit = None  # 推理时量化位数: 4, 8 or None
        
        # 分布式训练相关参数
        self.distributed_training = False  # 是否启用分布式训练
        
        # 数据相关参数
        self.dataset_name = None
        self.dataset_config_name = None
        self.data_dir = "./data"
        self.train_file = "llm/train.json"
        self.validation_file = "llm/validation.json"
        self.image_dir = "./data/vlm/images"
        self.prompt_file = "./data/vlm/prompt.json"
        self.label_map_file = "./data/vlm/label_map.json"
        self.file_format = "json"
        self.text_column = "text"
        self.target_column = "target"
        self.prompt_template = "{}"
        self.max_length = 512
        
        # LoRA 相关参数
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.1
        self.target_modules = ["q_proj", "v_proj"]
        
        # 训练相关参数
        self.output_dir = "./output"
        self.seed = 42
        self.num_train_epochs = 3
        self.per_device_train_batch_size = 4
        self.per_device_eval_batch_size = 4
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.warmup_steps = 0
        self.logging_steps = 100
        self.evaluation_strategy = "epoch"
        self.save_strategy = "epoch"
        self.save_steps = 1
        self.eval_steps = 1
        self.load_best_model_at_end = True
        self.metric_for_best_model = "loss"
        self.fp16 = True
        self.fp16_opt_level = "O1"
        self.bf16 = False
        self.tf32 = False
        self.max_grad_norm = 1.0
        self.push_to_hub = False
        self.hub_model_id = None
        self.hub_token = None
        self.report_to = "tensorboard"
        self.run_name = None
        
        # LoRA 推理相关参数
        self.lora_model_path = None
        self.use_original_model = False
        
        # 推理相关参数
        self.max_new_tokens = 100
        self.temperature = 0.7
        self.top_p = 0.95
        self.top_k = 50
        self.repetition_penalty = 1.0  # 默认值改回 1.0
        self.do_sample = True
    
    def load_from_file(self, config_file):
        """
        从配置文件加载参数
        Args:
            config_file: 配置文件路径
        """
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        # 更新参数
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def update_from_args(self, args):
        """
        从命令行参数更新配置
        Args:
            args: 命令行参数
        """
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

def parse_args():
    """
    解析命令行参数
    Returns:
        args: 命令行参数
    """
    parser = argparse.ArgumentParser(description="LoRA 微调实验")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--task", type=str, default="train", choices=["train", "evaluate", "inference"], help="任务类型")
    parser.add_argument("--model_name_or_path", type=str, help="预训练模型路径")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    parser.add_argument("--data_dir", type=str, help="数据目录")
    parser.add_argument("--num_train_epochs", type=int, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, help="每设备训练批次大小")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--lora_r", type=int, help="LoRA 秩")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha 参数")
    parser.add_argument("--lora_model_path", type=str, help="LoRA 模型路径，用于推理")
    parser.add_argument("--use_original_model", action="store_true", help="使用原始模型进行推理，不加载 LoRA 权重")
    parser.add_argument("--max_new_tokens", type=int, help="推理时生成的最大 token 数")
    parser.add_argument("--temperature", type=float, help="推理时的温度参数")
    parser.add_argument("--top_p", type=float, help="推理时的 top-p 参数")
    parser.add_argument("--repetition_penalty", type=float, help="推理时的重复惩罚参数")
    parser.add_argument("--distributed_training", action="store_true", help="启用分布式训练")
    parser.add_argument("--train_device", type=str, help="训练时使用的设备，例如 'cuda:0', 'cuda:1', 'cpu' 或 'auto'")
    parser.add_argument("--eval_device", type=str, help="评估时使用的设备，例如 'cuda:0', 'cuda:1', 'cpu' 或 'auto'")
    parser.add_argument("--inference_device", type=str, help="推理时使用的设备，例如 'cuda:0', 'cuda:1', 'cpu' 或 'auto'")
    parser.add_argument("--train_quantization_bit", type=int, help="训练时量化位数，例如 4 或 8")
    parser.add_argument("--inference_quantization_bit", type=int, help="推理时量化位数，例如 4 或 8")
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = Config(args.config)
    
    # 从命令行参数更新配置
    config.update_from_args(args)
    
    # 根据任务类型确定使用的设备参数
    task_device = None
    if args.task == "train":
        task_device = config.train_device
        print(f"训练使用设备: {task_device}")
    elif args.task == "evaluate":
        task_device = config.eval_device
        print(f"评估使用设备: {task_device}")
    elif args.task == "inference":
        task_device = config.inference_device
        print(f"推理使用设备: {task_device}")
    
    # 设置设备环境变量
    if task_device and task_device != "auto":
        # 如果指定了具体设备（如 cuda:0），则设置 CUDA_VISIBLE_DEVICES
        if task_device.startswith("cuda:"):
            try:
                device_id = task_device.split(":")[1]
                os.environ["CUDA_VISIBLE_DEVICES"] = device_id
                print(f"根据配置设置 CUDA_VISIBLE_DEVICES={device_id}")
            except IndexError:
                print(f"警告: 无法解析设备 ID: {task_device}")
        elif task_device == "cpu":
             os.environ["CUDA_VISIBLE_DEVICES"] = ""
             print("根据配置设置使用 CPU")
    
    # 检查训练设备设置，当设置为 "auto" 时自动启用分布式训练
    if args.task == "train" and task_device == "auto":
        import torch
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"检测到 {num_gpus} 张 GPU，train_device 设置为 'auto'，自动启用分布式训练")
            config.distributed_training = True
        else:
            print(f"检测到 {num_gpus} 张 GPU，train_device 设置为 'auto'，使用单进程训练")
            config.distributed_training = False
    
    # 自动调整输出目录结构
    if config.model_name_or_path:
        # 提取模型名称
        model_name = os.path.basename(os.path.normpath(config.model_name_or_path))
        # 处理类似 "path/to/model/" 的情况
        if not model_name:
             model_name = os.path.basename(os.path.dirname(os.path.normpath(config.model_name_or_path)))
        
        # 如果 output_dir 是默认值 "./output"，则追加模型名称
        if config.output_dir == "./output":
            config.output_dir = os.path.join(config.output_dir, model_name)
            
        # 如果 lora_model_path 是默认值 "./output/lora_model"，则更新为新的 output_dir 下
        if config.lora_model_path == "./output/lora_model":
            config.lora_model_path = os.path.join(config.output_dir, "lora_model")
            
    print(f"最终输出目录: {config.output_dir}")
    print(f"最终 LoRA 模型路径: {config.lora_model_path}")
    print(f"分布式训练: {'启用' if hasattr(config, 'distributed_training') and config.distributed_training else '禁用'}")
    
    # 执行任务
    if args.task == "train":
        train(config)
    elif args.task == "evaluate":
        evaluate(config)
    elif args.task == "inference":
        inference(config)
    else:
        raise ValueError(f"未知任务类型: {args.task}")

if __name__ == "__main__":
    main()