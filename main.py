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
        self.dtype = "float16"
        self.train_quantization_bit = None  # 训练时量化位数: 4, 8 or None
        self.inference_quantization_bit = None  # 推理时量化位数: 4, 8 or None
        
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