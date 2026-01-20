import torch
from transformers import AutoTokenizer
from model_config import load_lora_model

def inference(config):
    """
    使用模型进行推理
    Args:
        config: 配置对象，包含推理相关参数
    """
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    
    # 加载模型
    if hasattr(config, 'use_original_model') and config.use_original_model:
        # 使用原始模型
        print(f"加载原始模型: {config.model_name_or_path}")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
        print("原始模型加载完成")
    else:
        # 使用带有 LoRA 权重的模型
        model = load_lora_model(config.model_name_or_path, config.lora_model_path)
    model.eval()
    
    # 将模型移到 GPU 设备（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"模型已移到设备: {device}")
    
    # 推理循环
    print("开始推理，输入 'exit' 退出")
    while True:
        # 获取输入
        input_text = input("输入文本: ")
        if input_text.lower() == "exit":
            break
        
        # === 修改开始 ===
        # 1. 构建消息列表
        messages = [
            {"role": "user", "content": input_text}
        ]
        
        # 2. 应用聊天模板 (apply_chat_template 会自动添加 <|im_start|> 等特殊标记)
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # 如果模型没有模板（比较少见），回退到原始方式
            text = input_text
            
        print(f"实际输入模型的文本: {text}") # 调试用，看看格式对不对

        # 3. 分词
        inputs = tokenizer(text, return_tensors="pt")
        # === 修改结束 ===
        
        # 推理
        with torch.no_grad():
            # 将输入移到与模型相同的设备
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 获取模型特定的所有结束符
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                tokenizer.convert_tokens_to_ids("<|im_end|>") # Qwen 特有
            ]
            # 过滤掉 None
            terminators = [t for t in terminators if t is not None]

            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                eos_token_id=terminators,  # 传入列表
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 解码输出
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 清理输出文本，去除可能的重复内容
        # 1. 移除输入提示的重复
        if input_text in output_text:
            # 只保留输入提示后的内容
            cleaned_output = output_text.split(input_text, 1)[1].strip()
        else:
            cleaned_output = output_text
        
        # 2. 移除重复的"答案："和"最终答案："部分
        import re
        # 检测并移除重复的答案模式
        # 匹配"答案："、"最终答案："等模式
        answer_patterns = r'(答案：|最终答案：|总结：).*?(?=(答案：|最终答案：|总结：|$))'
        
        # 提取所有答案部分
        answer_matches = re.findall(answer_patterns, cleaned_output, re.DOTALL)
        
        # 如果有答案部分，只保留最后一个
        if answer_matches:
            # 找到最后一个答案部分的位置
            last_answer_pos = cleaned_output.rfind('答案：')
            if last_answer_pos == -1:
                last_answer_pos = cleaned_output.rfind('最终答案：')
            if last_answer_pos == -1:
                last_answer_pos = cleaned_output.rfind('总结：')
            
            # 只保留答案部分之前的内容
            if last_answer_pos != -1:
                cleaned_output = cleaned_output[:last_answer_pos].strip()
        
        # 3. 检测并移除重复段落
        # 分割文本为段落
        paragraphs = [p.strip() for p in cleaned_output.split('\n\n') if p.strip()]
        
        # 去重段落
        seen_paragraphs = set()
        unique_paragraphs = []
        
        for para in paragraphs:
            # 标准化段落（移除空白字符差异）
            normalized_para = re.sub(r'\s+', ' ', para)
            if normalized_para not in seen_paragraphs:
                seen_paragraphs.add(normalized_para)
                unique_paragraphs.append(para)
        
        # 重新组合段落
        cleaned_output = '\n\n'.join(unique_paragraphs)
        
        # 4. 处理特殊标记和重复内容
        import re
        # 移除重复的 ``` 标记
        cleaned_output = re.sub(r'`{3,}\s*`{3,}', '```', cleaned_output)
        # 移除多余的 ``` 标记
        cleaned_output = re.sub(r'`{3,}', '', cleaned_output)
        
        # 5. 移除末尾的空白和不完整句子
        cleaned_output = cleaned_output.strip()
        if cleaned_output:
            # 移除末尾可能的不完整句子
            sentences = re.split(r'[。！？]', cleaned_output)
            if sentences:
                # 只保留完整的句子
                complete_sentences = [s for s in sentences if s.strip()]
                if complete_sentences:
                    cleaned_output = '。'.join(complete_sentences) + '。'
        
        # 6. 最终清理
        # 移除多余的空行
        cleaned_output = re.sub(r'\n{3,}', '\n\n', cleaned_output)
        # 移除首尾空白
        cleaned_output = cleaned_output.strip()
        
        # 打印输出
        print("输出结果:")
        print(cleaned_output)
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