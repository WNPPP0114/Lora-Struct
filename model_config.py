import os
import traceback
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType

def load_model_with_lora(config):
    """
    加载预训练模型并配置 LoRA
    Args:
        config: 配置对象，包含模型和 LoRA 相关参数
    Returns:
        model: 配置了 LoRA 的模型
        model_config: 模型配置
    """
    try:
        # 检查模型路径是否存在（仅适用于本地路径）
        # 对于 Hugging Face Hub 模型（如 "distilgpt2"），会自动下载
        if os.path.exists(config.model_name_or_path):
            # 本地路径存在，正常加载
            pass
        elif os.path.isabs(config.model_name_or_path):
            # 绝对路径不存在，抛出错误
            raise ValueError(f"模型路径不存在: {config.model_name_or_path}")
        # 否则，假设是 Hugging Face Hub 模型名称，允许自动下载
        
        # 加载模型配置
        print(f"加载模型配置: {config.model_name_or_path}")
        model_config = AutoConfig.from_pretrained(config.model_name_or_path)
        print(f"模型配置加载完成，模型类型: {model_config.model_type}")
        
        # 加载预训练模型
        print(f"加载预训练模型: {config.model_name_or_path}")
        print(f"使用数据类型: {config.torch_dtype}")
        
        # 转换 torch_dtype
        if config.torch_dtype == "float16":
            torch_dtype = "float16"
        elif config.torch_dtype == "bfloat16":
            torch_dtype = "bfloat16"
        else:
            torch_dtype = "float32"
        
        # 检查是否有可用的 GPU
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            config=model_config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
        
        # 将模型移到指定设备
        model.to(device)
        print("预训练模型加载完成")
        
        # 配置 LoRA
        print("配置 LoRA...")
        print(f"LoRA 秩: {config.lora_r}")
        print(f"LoRA alpha: {config.lora_alpha}")
        print(f"LoRA dropout: {config.lora_dropout}")
        
        # 根据模型类型设置目标模块
        model_type = model_config.model_type
        if model_type == "distilgpt2":
            # DistilGPT2 模型使用的模块名称
            target_modules = ["q_lin", "v_lin"]
        elif model_type == "gpt2":
            # GPT2 模型使用的模块名称
            target_modules = ["c_attn"]
        elif model_type in ["qwen3", "llama", "mistral"]:
            # Qwen3、Llama 和 Mistral 模型使用的模块名称
            target_modules = ["q_proj", "v_proj"]
        else:
            # 默认使用配置中的模块名称
            target_modules = config.target_modules
        
        print(f"目标模块: {target_modules}")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules
        )
        
        # 应用 LoRA 配置到模型
        print("应用 LoRA 配置到模型...")
        model = get_peft_model(model, lora_config)
        print("LoRA 配置应用完成")
        
        # 打印可训练参数
        print_trainable_parameters(model)
        
        return model, model_config
    except Exception as e:
        print(f"模型加载过程中发生错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        raise

def print_trainable_parameters(model):
    """
    打印模型可训练参数数量
    Args:
        model: 模型对象
    """
    try:
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"可训练参数: {trainable_params} ({trainable_params / all_param * 100:.2f}%)")
        print(f"总参数: {all_param}")
    except Exception as e:
        print(f"计算参数时发生错误: {e}")
        traceback.print_exc()

def save_lora_model(model, output_dir):
    """
    保存 LoRA 模型
    Args:
        model: 训练后的模型
        output_dir: 输出目录
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建模型保存目录: {output_dir}")
        
        # 保存模型
        model.save_pretrained(output_dir)
        print(f"LoRA 模型已保存到: {output_dir}")
    except Exception as e:
        print(f"保存模型时发生错误: {e}")
        traceback.print_exc()

def load_lora_model(model_name_or_path, lora_model_path):
    """
    加载 LoRA 模型
    Args:
        model_name_or_path: 基础模型路径
        lora_model_path: LoRA 模型路径
    Returns:
        model: 加载了 LoRA 的模型
    """
    try:
        from peft import PeftModel
        
        # 检查路径是否存在
        if not os.path.exists(model_name_or_path):
            raise ValueError(f"基础模型路径不存在: {model_name_or_path}")
        if not os.path.exists(lora_model_path):
            raise ValueError(f"LoRA 模型路径不存在: {lora_model_path}")
        
        # 加载基础模型
        print(f"加载基础模型: {model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        print("基础模型加载完成")
        
        # 加载 LoRA 权重
        print(f"加载 LoRA 权重: {lora_model_path}")
        model = PeftModel.from_pretrained(model, lora_model_path)
        print("LoRA 权重加载完成")
        
        return model
    except Exception as e:
        print(f"加载 LoRA 模型时发生错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        raise