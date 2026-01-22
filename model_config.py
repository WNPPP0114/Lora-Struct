import os
import traceback
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

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
        print(f"使用数据类型: {config.dtype}")
        
        # 转换 dtype
        if config.dtype == "float16":
            dtype_val = "float16"
        elif config.dtype == "bfloat16":
            dtype_val = "bfloat16"
        else:
            dtype_val = "float32"
        
        # 检查是否有可用的 GPU
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 调试：打印 config.model_type
        print(f"Config model_type: {getattr(config, 'model_type', 'None')}")
        print(f"Model config model_type: {model_config.model_type}")
        
        # 根据模型类型选择加载类
        # 优先使用 config 中的设置，如果未设置或不明确，则根据 model_config 推断
        is_vlm = False
        if getattr(config, "model_type", "llm") == "vlm":
            is_vlm = True
        elif model_config.model_type in ["qwen2_vl", "qwen3_vl", "llava", "idefics", "vip_llava"]:
            print(f"检测到 VLM 模型架构: {model_config.model_type}，自动切换为 VLM 模式")
            is_vlm = True
            
        if is_vlm:
            print("使用 AutoModelForImageTextToText 加载 VLM 模型")
            model_class = AutoModelForImageTextToText
        else:
            print("使用 AutoModelForCausalLM 加载 LLM 模型")
            model_class = AutoModelForCausalLM

        # 量化配置
        quantization_config = None
        quantization_bit = None
        if hasattr(config, "train_quantization_bit") and config.train_quantization_bit is not None:
            quantization_bit = config.train_quantization_bit
        elif hasattr(config, "quantization_bit") and config.quantization_bit is not None:
            quantization_bit = config.quantization_bit
        
        if quantization_bit is not None:
            print(f"启用 {quantization_bit}-bit 量化加载")
            try:
                q_bit = int(quantization_bit)
                if q_bit == 8:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                elif q_bit == 4:
                    # 确定计算类型
                    bnb_4bit_compute_dtype = torch.float16
                    if config.dtype == "bfloat16":
                        bnb_4bit_compute_dtype = torch.bfloat16
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
            except ValueError:
                print(f"警告: 无效的量化位数 {quantization_bit}，忽略量化配置")

        # 如果启用了 bitsandbytes 量化，需要先移除模型配置中的 quantization_config
        # 否则会与 FineGrainedFP8Config 冲突
        if quantization_config and hasattr(model_config, "quantization_config"):
            print("检测到模型自带 quantization_config，正在移除以应用 bitsandbytes 量化...")
            del model_config.quantization_config

        # 加载模型参数
        load_kwargs = {
            "config": model_config,
            "dtype": dtype_val,
            "low_cpu_mem_usage": True,
        }
        
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
            print("使用 device_map='auto' 进行量化加载")

        model = model_class.from_pretrained(
            config.model_name_or_path,
            **load_kwargs
        )
        
        # 如果没有使用 device_map (即没有量化)，则手动移动到设备
        if not quantization_config:
            model.to(device)
            
        print("预训练模型加载完成")
        
        # 如果使用了量化，需要预处理模型以进行 k-bit 训练
        if quantization_config:
            print("预处理模型以进行 k-bit 训练...")
            model = prepare_model_for_kbit_training(model)
        
        # 针对 FP8 模型的特殊处理
        # 如果模型本身是 FP8 量化的（例如 Qwen3-VL-4B-Instruct-FP8），但没有使用 bitsandbytes 量化加载
        # 我们需要确保模型处于可以训练的状态。
        # 目前 transformers 对 FP8 模型的训练支持可能有限，通常需要将其视为 float16/bfloat16 进行 LoRA 微调
        # 或者确保 requires_grad 设置正确。
        # 注意：如果模型加载时已经是 FP8，通常意味着它使用了某种量化后端。
        # 如果遇到 "QuantizationMethod.FP8 ... do not support training" 错误，
        # 可能需要禁用量化配置或者使用特定的加载方式。
        
        # 检查是否是 FP8 模型且未启用 bitsandbytes 量化
        if "FP8" in config.model_name_or_path and not quantization_config:
             print("检测到 FP8 模型，尝试启用梯度检查点以支持训练...")
             # 某些情况下，启用梯度检查点可以绕过一些限制，或者帮助节省显存
             if hasattr(model, "gradient_checkpointing_enable"):
                 model.gradient_checkpointing_enable()
             
             # 尝试禁用 FP8 量化配置以支持训练
             # 如果模型加载时带有 quantization_config 且 method 为 FP8，可能会阻止训练
             if hasattr(model, "quantization_method"):
                 print(f"模型量化方法: {model.quantization_method}")
                 # 这是一个 hack，尝试绕过 transformers 的检查
                 # 注意：这可能会导致显存增加，因为模型可能会被反量化
                 # 但对于 Qwen3-VL-FP8，它可能只是权重是 FP8，计算时会转为 BF16/FP16
                 pass
        
        # 配置 LoRA
        print("配置 LoRA...")
        print(f"LoRA 秩: {config.lora_r}")
        print(f"LoRA alpha: {config.lora_alpha}")
        print(f"LoRA dropout: {config.lora_dropout}")
        
        # 根据模型类型设置目标模块
        # 优先使用 config.yaml 中的配置
        if hasattr(config, "target_modules") and config.target_modules:
            target_modules = config.target_modules
        else:
            # 如果 config 中未指定，则使用默认策略
            model_type = model_config.model_type
            if model_type == "distilgpt2":
                target_modules = ["q_lin", "v_lin"]
            elif model_type == "gpt2":
                target_modules = ["c_attn"]
            elif model_type in ["qwen3", "llama", "mistral", "qwen2_vl"]:
                target_modules = ["q_proj", "v_proj"]
            else:
                target_modules = ["q_proj", "v_proj"]
        
        print(f"目标模块: {target_modules}")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, # VLM 通常也兼容 CAUSAL_LM 的 LoRA 配置
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

def load_lora_model(model_name_or_path, lora_model_path, config=None):
    """
    加载 LoRA 模型
    Args:
        model_name_or_path: 基础模型路径
        lora_model_path: LoRA 模型路径
        config: 配置对象，包含 quantization_bit 等参数
    Returns:
        model: 加载了 LoRA 的模型
    """
    try:
        import torch
        from peft import PeftModel
        
        # 检查路径是否存在
        if not os.path.exists(model_name_or_path):
            raise ValueError(f"基础模型路径不存在: {model_name_or_path}")
        if not os.path.exists(lora_model_path):
            raise ValueError(f"LoRA 模型路径不存在: {lora_model_path}")
        
        # 加载基础模型
        print(f"加载基础模型: {model_name_or_path}")
        
        # 尝试检测模型类型
        try:
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(model_name_or_path)
            architectures = getattr(model_config, "architectures", [])
            is_vlm = any("Vision" in arch or "Qwen2VL" in arch or "Qwen3VL" in arch for arch in architectures)
        except:
            is_vlm = False
        
        # 量化配置
        quantization_config = None
        quantization_bit = None
        if config:
            if hasattr(config, "inference_quantization_bit") and config.inference_quantization_bit is not None:
                quantization_bit = config.inference_quantization_bit
            elif hasattr(config, "quantization_bit") and config.quantization_bit is not None:
                quantization_bit = config.quantization_bit
        
        if quantization_bit is not None:
            print(f"启用 {quantization_bit}-bit 量化加载")
            try:
                q_bit = int(quantization_bit)
                if q_bit == 8:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                elif q_bit == 4:
                    # 确定计算类型
                    bnb_4bit_compute_dtype = torch.float16
                    if config and hasattr(config, "dtype") and config.dtype == "bfloat16":
                        bnb_4bit_compute_dtype = torch.bfloat16
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
            except ValueError:
                print(f"警告: 无效的量化位数 {quantization_bit}，忽略量化配置")
        
        # 加载模型参数
        load_kwargs = {
            "dtype": "auto",
            "low_cpu_mem_usage": True,
        }
        
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
            
            # 确定 device_map 策略
            # 在多卡环境下，device_map="auto" 可能会导致小模型推理崩溃
            device_map = "auto"
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                if "CUDA_VISIBLE_DEVICES" not in os.environ:
                    print(f"检测到 {torch.cuda.device_count()} 张显卡。为避免多卡推理崩溃，默认使用第一张显卡 (cuda:0)。")
                    device_map = {"": "cuda:0"}
            
            load_kwargs["device_map"] = device_map
            print(f"使用 device_map='{device_map}' 进行量化加载")
        
        if is_vlm:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, **load_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
            
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