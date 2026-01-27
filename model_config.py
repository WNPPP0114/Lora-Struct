import os
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel

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
        is_vlm = detect_model_type(config.model_name_or_path, config)
        
        if is_vlm:
            print("使用 AutoModelForImageTextToText 加载 VLM 模型")
            model_class = AutoModelForImageTextToText
        else:
            print("使用 AutoModelForCausalLM 加载 LLM 模型")
            model_class = AutoModelForCausalLM

        # 量化配置
        quantization_bit = getattr(config, "train_quantization_bit", None) or \
                          getattr(config, "quantization_bit", None)
        quantization_config = create_quantization_config(quantization_bit, dtype_val)
        
        if quantization_config:
            print(f"启用 {quantization_bit}-bit 量化加载")

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

def detect_model_type(model_name_or_path, config=None):
    """
    检测模型类型（VLM 或 LLM）
    Args:
        model_name_or_path: 模型路径
        config: 配置对象（可选）
    Returns:
        is_vlm: 是否为 VLM 模型
    """
    try:
        # 优先使用 config 中的设置
        if config and getattr(config, "model_type", None) == "vlm":
            return True
        
        # 尝试从模型配置推断
        model_config = AutoConfig.from_pretrained(model_name_or_path)
        architectures = getattr(model_config, "architectures", [])
        if any("Vision" in arch or "Qwen2VL" in arch or "Qwen3VL" in arch for arch in architectures):
            return True
        
        # 检查模型类型
        if model_config.model_type in ["qwen2_vl", "qwen3_vl", "llava", "idefics", "vip_llava"]:
            return True
        
        return False
    except Exception:
        return False

def create_quantization_config(quantization_bit, dtype="float16"):
    """
    创建量化配置
    Args:
        quantization_bit: 量化位数 (4 或 8)
        dtype: 数据类型，用于确定 4-bit 的计算类型
    Returns:
        quantization_config: BitsAndBytesConfig 对象，如果无效则返回 None
    """
    if quantization_bit is None:
        return None
    
    try:
        q_bit = int(quantization_bit)
        if q_bit == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        elif q_bit == 4:
            bnb_4bit_compute_dtype = torch.float16
            if dtype == "bfloat16":
                bnb_4bit_compute_dtype = torch.bfloat16
            
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            print(f"警告: 不支持的量化位数 {q_bit}，仅支持 4 或 8")
            return None
    except (ValueError, TypeError) as e:
        print(f"警告: 无效的量化位数 {quantization_bit}，忽略量化配置: {e}")
        return None

def determine_device_map(model_name_or_path, is_distributed=False, prefer_single_device=True):
    """
    确定设备映射策略
    Args:
        model_name_or_path: 模型路径
        is_distributed: 是否在分布式训练环境中
        prefer_single_device: 是否优先使用单设备（用于推理稳定性）
    Returns:
        device_map: 设备映射配置，如果不应设置则返回 None
    """
    # 在分布式训练环境中，不设置 device_map
    if is_distributed:
        return None
    
    if not torch.cuda.is_available():
        print("未检测到 GPU，使用 CPU")
        return {"": "cpu"}
    
    # 检查环境变量
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        print(f"CUDA_VISIBLE_DEVICES: {visible_devices}")
        print("策略选择: 环境变量指定了可见设备，使用第一张可见设备")
        return {"": "cuda:0"}
    
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 张 GPU")
    
    # 打印GPU内存信息
    for i in range(num_gpus):
        try:
            gpu_mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            if hasattr(torch.cuda, 'memory_free'):
                gpu_mem_free = torch.cuda.memory_free(i) / 1024**3
            elif hasattr(torch.cuda, 'mem_get_info'):
                gpu_mem_free = torch.cuda.mem_get_info(i)[0] / 1024**3
            else:
                gpu_mem_free = gpu_mem_total * 0.8
            print(f"GPU {i}: {gpu_mem_free:.2f}GB / {gpu_mem_total:.2f}GB 可用")
        except Exception as e:
            gpu_mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: 总内存 {gpu_mem_total:.2f}GB")
    
    if num_gpus == 1:
        print("策略选择: 单GPU环境，使用默认CUDA设备")
        return {"": "cuda"}
    
    # 多GPU环境
    if prefer_single_device:
        print("策略选择: 多GPU环境，使用单设备策略以确保稳定性")
        return {"": "cuda:0"}
    
    # 尝试根据模型大小选择策略
    try:
        model_config = AutoConfig.from_pretrained(model_name_or_path)
        num_layers = getattr(model_config, "num_hidden_layers", 0)
        print(f"模型层数: {num_layers}")
        
        if num_layers < 24:  # 小模型
            print("策略选择: 检测到小模型，使用单设备策略避免崩溃")
            return {"": "cuda:0"}
        else:  # 大模型
            print("策略选择: 检测到大模型，尝试使用 auto 设备映射策略")
            return "auto"
    except Exception as e:
        print(f"获取模型信息时出错: {e}，默认使用单设备策略")
        return {"": "cuda:0"}

def load_base_model_with_fallback(model_name_or_path, is_vlm, load_kwargs):
    """
    加载基础模型，带自动回退机制
    Args:
        model_name_or_path: 模型路径
        is_vlm: 是否为 VLM 模型
        load_kwargs: 加载参数（字典，会被复制以避免修改原始参数）
    Returns:
        model: 加载的模型
    """
    model_class = AutoModelForImageTextToText if is_vlm else AutoModelForCausalLM
    
    # 尝试加载策略列表
    fallback_strategies = [
        ("原始策略", lambda: None),
        ("单设备策略", lambda: {"": "cuda:0"}),
        ("CPU策略", lambda: {"": "cpu"}),
    ]
    
    for strategy_name, get_device_map in fallback_strategies:
        try:
            # 创建参数副本，避免修改原始参数
            current_kwargs = load_kwargs.copy()
            device_map = get_device_map()
            
            if device_map is not None:
                current_kwargs["device_map"] = device_map
                if strategy_name != "原始策略":
                    print(f"尝试使用{strategy_name}...")
            elif "device_map" in current_kwargs and strategy_name != "原始策略":
                # 如果原始策略没有 device_map，但后续策略需要，则移除它
                del current_kwargs["device_map"]
            
            model = model_class.from_pretrained(model_name_or_path, **current_kwargs)
            print(f"基础模型加载完成（{strategy_name}）")
            return model
        except Exception as e:
            if strategy_name == "原始策略":
                print(f"模型加载失败: {e}")
            else:
                print(f"{strategy_name}也失败: {e}")
            continue
    
    # 所有策略都失败
    raise RuntimeError(f"所有加载策略都失败，无法加载模型: {model_name_or_path}")

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
        # 检查路径是否存在
        if not os.path.exists(model_name_or_path):
            raise ValueError(f"基础模型路径不存在: {model_name_or_path}")
        if not os.path.exists(lora_model_path):
            raise ValueError(f"LoRA 模型路径不存在: {lora_model_path}")
        
        print(f"加载基础模型: {model_name_or_path}")
        
        # 检测模型类型
        is_vlm = detect_model_type(model_name_or_path, config)
        
        # 获取量化配置
        quantization_bit = None
        if config:
            quantization_bit = getattr(config, "inference_quantization_bit", None) or \
                              getattr(config, "quantization_bit", None)
        
        dtype = getattr(config, "dtype", "float16") if config else "float16"
        quantization_config = create_quantization_config(quantization_bit, dtype)
        
        if quantization_config:
            print(f"启用 {quantization_bit}-bit 量化加载")
        
        # 准备加载参数
        load_kwargs = {
            "dtype": "auto",
            "low_cpu_mem_usage": True,
        }
        
        # 确定设备映射策略
        is_distributed = os.environ.get('RANK') is not None or os.environ.get('LOCAL_RANK') is not None
        if is_distributed:
            print("检测到分布式训练环境")
        
        # 如果使用量化，使用 device_map="auto"（和训练时一致）
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
            print("最终选择: 使用 device_map='auto' 进行量化加载（和训练时一致）")
        else:
            # 如果不使用量化，使用 determine_device_map 的策略
            device_map = determine_device_map(model_name_or_path, is_distributed, prefer_single_device=True)
            if device_map:
                load_kwargs["device_map"] = device_map
                print(f"最终选择: 使用 device_map='{device_map}' 进行加载")
            else:
                print("最终选择: 不设置 device_map，由训练框架处理设备分配")
        
        # 加载基础模型（带自动回退）
        model = load_base_model_with_fallback(model_name_or_path, is_vlm, load_kwargs)
        
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