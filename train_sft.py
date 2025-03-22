# 首先设置环境变量 - 必须在导入之前
import os
# 禁用所有 Unsloth 补丁
os.environ["UNSLOTH_DISABLE_RL_PATCH"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import logging
import json
import gc
from tqdm import tqdm
from pathlib import Path
import time
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入必要的库
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import Trainer, TrainingArguments
    from transformers import DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model, PeftModel
    from datasets import Dataset
except ImportError as e:
    logger.error(f"导入库失败: {e}")
    sys.exit(1)

# 模型配置
MODEL_PATH = "./deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16

# 训练配置  
BATCH_SIZE = 2
GRAD_ACCUMULATION = 4
MAX_STEPS = 5000  # 增加到5000步
WARMUP_STEPS = 100
EVAL_STEPS = 10  # 每10步评估一次
OUTPUT_DIR = "./outputs"
DATASET_PATH = "./JEC-QA/1_train.json"
MAX_SAMPLES = None

# 评估样本
EVAL_SAMPLES = [
    {
        "answer": ["B"], 
        "id": "1_4269", 
        "option_list": {
            "A": "我国商务部在确定进口橡胶制品是否存在补贴时必须证明出国(地区)政府直接向出口商提供了现金形式的财政资助", 
            "B": "在反补贴调查期间，该八国政府或橡胶制品的出口经营者，可以向中国商务部作出承诺，取消、限制补贴或改变价格", 
            "C": "如果我国商务部终局裁定决定对该八国进口橡胶制品征收反补贴税，该反补贴税的征收期限不得超过10年", 
            "D": "如果中国橡胶制品进口商对商务部征收反补贴税的终局裁定不服，必须首先向商务部请求行政复审，对行政复审决定还不服，才能向中国有管辖权的法院起诉"
        }, 
        "statement": "中国商务部决定对原产于马来西亚等八国的橡胶制品展开反补贴调查。根据我国《反补贴条例》以及相关法律法规，下列关于此次反补贴调查的哪项判断是正确的?"
    },
    {
        "answer": ["D"], 
        "id": "3_6654", 
        "option_list": {
            "A": "该法典体现了‘个人最大限度的自由，法律最小限度的干涉’这一立法精神", 
            "B": "该法典具有鲜明的革命性和时代性", 
            "C": "该法典的影响后来传播到美洲、非洲和亚洲广大地区", 
            "D": "该法典首次全面规定了法人制度"
        }, 
        "statement": "1804年的《法国民法典》是世界近代法制史上的第一部民法典，是大陆法系的核心和基础。下列关于《法国民法典》的哪一项表述不正确?"
    }
]

# 提示模板
LAW_PROMPT_TEMPLATE = """请你作为一位法律专家，分析下面的法律问题并给出你认为正确的答案。请先思考分析题目，然后从选项中选择一个最合适的答案。

### 问题：
{question}

### 回应：
<think>

</think>

"""

# 评估提示模板
EVAL_PROMPT_TEMPLATE = """请你作为一位法律专家，分析下面的法律问题并给出你认为正确的答案。请先思考分析题目，然后从选项中选择一个最合适的答案。

### 问题：
{question}

### 回应：
<think>
"""

def load_dataset(file_path, max_samples=None):
    """加载数据集并处理"""
    logger.info(f"加载数据集: {file_path}")
    
    # 尝试加载 JSON 文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                logger.info("成功加载 JSON 数据")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败: {e}，尝试按行解析...")
                f.seek(0)
                data = []
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析第 {i+1} 行")
                
                if not data:
                    # 创建一个示例数据
                    logger.warning("创建示例数据...")
                    data = [{
                        "statement": "这是一个示例问题",
                        "option_list": ["选项A", "选项B", "选项C", "选项D"],
                        "answer": "A"
                    }]
    except Exception as e:
        logger.error(f"加载文件失败: {e}")
        # 创建一个示例数据
        data = [{
            "statement": "这是一个示例问题",
            "option_list": ["选项A", "选项B", "选项C", "选项D"],
            "answer": "A"
        }]
    
    # 确保数据是列表形式
    if not isinstance(data, list):
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        else:
            data = [data]
            
    # 限制样本数量
    if max_samples and len(data) > max_samples:
        logger.info(f"限制样本数量为: {max_samples}")
        data = data[:max_samples]
        
    logger.info(f"加载了 {len(data)} 条数据记录")
    return data

def format_dataset(data, prompt_template):
    """格式化数据集"""
    logger.info("格式化数据集...")
    formatted_data = []
    
    for item in tqdm(data, desc="处理数据"):
        try:
            # 提取问题陈述和选项
            statement = item.get("statement", "")
            option_list = item.get("option_list", {})
            
            # 处理选项 - 支持列表和字典两种格式
            options_text = ""
            if isinstance(option_list, dict):
                for key, value in option_list.items():
                    options_text += f"{key}. {value}\n"
            elif isinstance(option_list, list):
                for i, option in enumerate(option_list):
                    option_letter = chr(65 + i)  # A, B, C, D...
                    options_text += f"{option_letter}. {option}\n"
            
            # 格式化问题
            question = f"{statement}\n\n{options_text.strip()}"
            
            # 将提示模板格式化为最终输入文本
            formatted_prompt = prompt_template.format(question=question)
            
            formatted_data.append({
                "text": formatted_prompt
            })
        except Exception as e:
            logger.warning(f"处理数据时出错: {e}, 跳过")
            continue
            
    logger.info(f"格式化完成，共 {len(formatted_data)} 条记录")
    return formatted_data

def format_eval_sample(sample):
    """格式化评估样本"""
    statement = sample["statement"]
    option_list = sample["option_list"]
    
    # 处理选项 - 支持列表和字典两种格式
    options_text = ""
    if isinstance(option_list, dict):
        for key, value in option_list.items():
            options_text += f"{key}. {value}\n"
    elif isinstance(option_list, list):
        for i, option in enumerate(option_list):
            option_letter = chr(65 + i)  # A, B, C, D...
            options_text += f"{option_letter}. {option}\n"
    
    # 格式化问题
    question = f"{statement}\n\n{options_text.strip()}"
    return question

def manual_tokenize(texts, tokenizer):
    """手动对文本进行分词，避免使用多进程"""
    logger.info(f"手动分词 {len(texts)} 条文本...")
    result = []
    
    for i, item in enumerate(tqdm(texts, desc="分词处理")):
        try:
            text = item["text"]
            # 手动分词
            tokenized = tokenizer.encode_plus(
                text,
                max_length=MAX_SEQ_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )
            result.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            })
        except Exception as e:
            logger.warning(f"分词错误 (项 {i}): {e}")
            # 创建一个空的分词结果
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            result.append({
                "input_ids": [pad_id] * MAX_SEQ_LENGTH,
                "attention_mask": [0] * MAX_SEQ_LENGTH,
            })
    
    logger.info("分词完成")
    return result

def extract_thinking(response, question):
    """从响应中提取思维链"""
    try:
        # 尝试使用正则表达式提取<think>和</think>之间的内容
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            return thinking
        
        # 尝试提取回应部分
        response_parts = response.split("### 回应:")
        if len(response_parts) > 1:
            return response_parts[1].strip()
            
        # 如果以上方法都失败，返回整个响应
        return response
    except Exception:
        return response

class EvalCallback:
    """用于评估模型的回调类，在CPU上执行评估"""
    def __init__(self, eval_samples, tokenizer, output_dir, base_model_path):
        self.eval_samples = eval_samples
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.base_model_path = base_model_path
        self.eval_results = []
        
        # 创建评估日志目录
        self.eval_log_dir = os.path.join(output_dir, "eval_logs")
        os.makedirs(self.eval_log_dir, exist_ok=True)
        
    def evaluate(self, step):
        """在CPU上评估当前模型"""
        print("\n" + "="*80)
        print(f"步骤 {step}: 开始评估...")
        print("="*80 + "\n")
        
        # 确保有检查点
        adapter_path = os.path.join(self.output_dir, f"checkpoint-{step}")
        if not os.path.exists(adapter_path):
            print(f"检查点 {adapter_path} 不存在，等待5秒...")
            time.sleep(5)  # 等待检查点保存完成
            
            if not os.path.exists(adapter_path):
                print(f"检查点 {adapter_path} 仍然不存在，跳过评估")
                return
        
        try:
            # 在CPU上加载模型进行评估
            print("在CPU上加载模型进行评估...")
            
            # 加载基础模型到CPU
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float32,  # 使用float32以在CPU上更好地兼容
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            # 加载LoRA adapter
            peft_model = PeftModel.from_pretrained(
                model, 
                adapter_path,
                device_map="cpu"
            )
            peft_model.eval()
            
            # 评估每个样本
            eval_results = []
            for i, sample in enumerate(self.eval_samples):
                print(f"\n--- 评估样本 {i+1}/{len(self.eval_samples)} ---\n")
                question = format_eval_sample(sample)
                correct_answer = sample["answer"]
                if isinstance(correct_answer, list) and len(correct_answer) > 0:
                    correct_answer = correct_answer[0]
                
                prompt = EVAL_PROMPT_TEMPLATE.format(question=question)
                
                # 打印问题
                print("\n问题:")
                print(question)
                print("\n" + "-"*40 + "\n")
                
                # 准备输入
                inputs = self.tokenizer(prompt, return_tensors="pt")
                
                # 在CPU上生成回答
                with torch.no_grad():
                    outputs = peft_model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=500,
                        do_sample=False
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                thinking = extract_thinking(response, question)
                
                # 打印思维链
                print("思维链:")
                print(thinking)
                print("\n" + "-"*40 + "\n")
                
                # 打印正确答案
                print(f"正确答案: {correct_answer}")
                print("\n" + "="*40 + "\n")
                
                # 记录结果
                result = {
                    "step": step,
                    "sample_id": sample["id"],
                    "question": question,
                    "prompt": prompt,
                    "response": response,
                    "thinking": thinking,
                    "correct_answer": correct_answer,
                }
                eval_results.append(result)
                
                # 保存到文件
                with open(os.path.join(self.eval_log_dir, f"step_{step}_sample_{i}.json"), "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            
            # 添加到评估结果列表
            self.eval_results.extend(eval_results)
            
            # 释放内存
            del model
            del peft_model
            gc.collect()
            
            print(f"\n步骤 {step}: 评估完成")
            print("="*80 + "\n")
        except Exception as e:
            print(f"评估出错: {e}")
            import traceback
            traceback.print_exc()

def main():
    try:
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 1. 加载模型和分词器 - 使用原生 HuggingFace 方法
        logger.info("加载模型和分词器...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            use_fast=True
        )
        
        # 确保分词器有 pad_token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = "<|pad|>"
                
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        logger.info("模型和分词器加载成功")
        
        # 2. 加载并处理数据集 
        raw_data = load_dataset(DATASET_PATH, MAX_SAMPLES)
        formatted_data = format_dataset(raw_data, LAW_PROMPT_TEMPLATE)
        
        # 3. 手动对文本进行分词（不使用多进程）
        tokenized_data = manual_tokenize(formatted_data, tokenizer)
        
        # 4. 创建数据集对象
        logger.info("创建数据集对象...")
        dataset = Dataset.from_dict({
            "input_ids": [item["input_ids"] for item in tokenized_data],
            "attention_mask": [item["attention_mask"] for item in tokenized_data],
        })
        
        # 5. 配置LoRA
        logger.info("配置LoRA模型...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # 6. 创建评估回调
        eval_callback = EvalCallback(
            eval_samples=EVAL_SAMPLES,
            tokenizer=tokenizer,
            output_dir=OUTPUT_DIR,
            base_model_path=MODEL_PATH
        )
        
        # 7. 设置训练参数
        logger.info("设置训练参数...")
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUMULATION,
            learning_rate=2e-4,
            max_steps=MAX_STEPS,
            warmup_steps=WARMUP_STEPS,
            logging_steps=10,
            save_steps=EVAL_STEPS,  # 每10步保存一次
            fp16=True,  # 使用FP16
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            # 重要：完全禁用多进程
            dataloader_num_workers=0,
            report_to="none",
            # 其他设置
            remove_unused_columns=False,
        )
        
        # 8. 创建数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # 9. 创建训练器
        logger.info("创建标准 Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # 增加回调以手动评估
        from transformers.trainer_callback import TrainerCallback
        
        class EvalStepCallback(TrainerCallback):
            def __init__(self, eval_callback):
                self.eval_callback = eval_callback
                
            def on_save(self, args, state, control, **kwargs):
                """每次保存模型后评估"""
                if state.global_step > 0:
                    print(f"模型已保存，步骤 {state.global_step}，开始评估...")
                    # 等待模型完全保存
                    time.sleep(2)
                    self.eval_callback.evaluate(state.global_step)
                return control
        
        trainer.add_callback(EvalStepCallback(eval_callback))
        
        # 10. 开始训练
        logger.info("开始训练...")
        trainer.train()
        
        # 11. 保存最终模型
        logger.info("保存最终模型...")
        final_output_dir = os.path.join(OUTPUT_DIR, "final_model")
        os.makedirs(final_output_dir, exist_ok=True)
        
        # 保存LoRA模型
        model.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        
        logger.info("训练完成!")
        
        # 12. 生成评估报告
        logger.info("生成评估报告...")
        eval_summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.json")
        with open(eval_summary_path, "w", encoding="utf-8") as f:
            json.dump(eval_callback.eval_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估报告已保存到: {eval_summary_path}")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"运行时错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()