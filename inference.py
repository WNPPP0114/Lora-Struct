import torch
import os
import json
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
from model_config import load_lora_model

def inference(config):
    """
    使用模型进行推理
    Args:
        config: 配置对象，包含推理相关参数
    """
    # 自动检测是否是 VLM
    try:
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(config.model_name_or_path)
        architectures = getattr(model_config, "architectures", [])
        is_vlm = any("Vision" in arch or "Qwen2VL" in arch or "Qwen3VL" in arch for arch in architectures)
    except:
        is_vlm = getattr(config, "model_type", "llm") == "vlm"
        
    if is_vlm:
        print(f"加载 Processor: {config.model_name_or_path}")
        processor = AutoProcessor.from_pretrained(config.model_name_or_path)
        tokenizer = processor.tokenizer
        
        # 加载 System Prompt
        system_prompt = ""
        if config.prompt_file and os.path.exists(config.prompt_file):
            try:
                with open(config.prompt_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    # 尝试解析 JSON
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict) and "system_prompt" in data:
                            system_prompt = data["system_prompt"]
                        else:
                            system_prompt = content
                    except json.JSONDecodeError:
                        system_prompt = content
                print(f"已加载 System Prompt (长度: {len(system_prompt)})")
            except Exception as e:
                print(f"加载 Prompt 文件失败: {e}")
    else:
        # 加载分词器
        # 如果自动检测失败，尝试加载 Processor，如果失败则回退到 Tokenizer
        try:
            if is_vlm: # 理论上这里不会进，但为了保险
                 processor = AutoProcessor.from_pretrained(config.model_name_or_path)
                 tokenizer = processor.tokenizer
            else:
                 tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
                 processor = None
        except:
             # 最后的兜底
             tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
             processor = None
    
    # 加载模型
    if hasattr(config, 'use_original_model') and config.use_original_model:
        # 使用原始模型
        print(f"加载原始模型: {config.model_name_or_path}")
        if is_vlm:
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(config.model_name_or_path, torch_dtype="auto", device_map="auto")
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, torch_dtype="auto", device_map="auto")
        print("原始模型加载完成")
    else:
        # 使用带有 LoRA 权重的模型
        model = load_lora_model(config.model_name_or_path, config.lora_model_path)
    
    model.eval()
    
    # 将模型移到 GPU 设备（如果可用且未自动分配）
    if not hasattr(model, "device_map") or not model.device_map:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"模型已移到设备: {device}")
    else:
        device = model.device
    
    # 推理循环
    print("开始推理，输入 'exit' 退出")
    while True:
        # 获取输入
        if is_vlm:
            image_path = input("输入图片路径: ")
            if image_path.lower() == "exit":
                break
            
            if not os.path.exists(image_path):
                print("图片文件不存在，请重新输入")
                continue
                
            # 构建 VLM 输入
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": system_prompt},
                    ],
                }
            ]
            
            # 准备输入
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs = Image.open(image_path).convert("RGB")
            inputs = processor(text=[text], images=[image_inputs], padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
        else:
            input_text = input("输入文本: ")
            if input_text.lower() == "exit":
                break
            
            # 构建 LLM 输入
            messages = [{"role": "user", "content": input_text}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample
            )
        
        # 解码输出
        if is_vlm:
            # VLM 输出处理
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.get("input_ids"), outputs)]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            # LLM 输出处理
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.get("input_ids"), outputs)]
            output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 打印输出
        print("输出结果:")
        print(output_text.strip())
        print("-" * 50)

if __name__ == "__main__":
    # 导入配置
    from main import Config
    
    # 加载配置
    config = Config("config.yaml")
    
    # 设置推理参数
    config.lora_model_path = "./output/lora_model"
    config.max_new_tokens = 100
    config.temperature = 0.7
    config.top_p = 0.95
    config.top_k = 50
    config.repetition_penalty = 1.0
    config.do_sample = True
    
    # 执行推理
    inference(config)