import os, torch
import ahocorasick
from peft import PeftModel
from loguru import logger
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from param_model import ChatMessage
from typing import List, Dict
from transformers_stream_generator.main import NewGenerationMixin

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_BASE_DIR = os.path.join(BASE_DIR, "llms")


def auto_download(model_type: str, revision: str = None, repair_name: str = None):
    from modelscope import snapshot_download
    cache_dir = f"{MODEL_BASE_DIR}/{model_type}"
    if model_type == "chatglm":
        model_id = "ZhipuAI/chatglm3-6b"
    elif model_type == "baichuan":
        model_id = "baichuan-inc/Baichuan2-13B-Chat"
    elif model_type == "qwen":
        model_id = "qwen/Qwen1.5-7B-Chat"
    elif model_type == "llama":
        model_id = "LLM-Research/Meta-Llama-3-8B-Instruct"
    elif model_type == "embedding":
        model_id = "AI-ModelScope/bge-large-zh-v1.5"
    elif model_type == "reranker":
        model_id = "quietnight/bge-reranker-large"
    else:
        raise ValueError(f'Unsupported model type {model_type} yet.')
    out_dir = snapshot_download(model_id, revision=revision, cache_dir=cache_dir)
    if repair_name is not None:
        from pathlib import Path
        model_path = Path(out_dir)
        repair_path = Path(os.path.join(cache_dir, repair_name))
        if not model_path.exists():
            raise ValueError(f'Model path {model_path} is not exists.')
        if repair_path.parent != ".":
            model_path.parent.rename(repair_path.parent.absolute())
            model_path = repair_path.parent / Path(model_path.name)
        model_path.rename(repair_path.absolute())


def get_bge_large_zh():
    model_type = "embedding"
    model_name_or_path = f"{model_type}/BAAI/bge-large-zh-v1.5"
    if not os.path.exists(os.path.join(MODEL_BASE_DIR, model_name_or_path)):
        auto_download(model_type, repair_name="BAAI/bge-large-zh-v1.5")
    logger.info(f'正在加载模型>>>>{model_name_or_path}\n')
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(os.path.join(MODEL_BASE_DIR, model_name_or_path))
    model = model.eval()
    return {
        "model_type": model_type,
        'model': model,
    }


def get_bge_reranker_large():
    model_type = "reranker"
    model_name_or_path = f"{model_type}/BAAI/bge-reranker-large"
    if not os.path.exists(os.path.join(MODEL_BASE_DIR, model_name_or_path)):
        auto_download(model_type, repair_name="BAAI/bge-reranker-large")
    logger.info(f'正在加载模型>>>>{model_name_or_path}\n')
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(os.path.join(MODEL_BASE_DIR, model_name_or_path))
    return {
        "model_type": model_type,
        'model': model,
    }


def get_llama3():
    model_type = 'llama'
    model_name_or_path = f"{model_type}/MetaAI/Meta-Llama-3-8B-Instruct"
    if not os.path.exists(os.path.join(MODEL_BASE_DIR, model_name_or_path)):
        auto_download(model_type, repair_name='MetaAI/Meta-Llama-3-8B-Instruct')
    SFT_MODEL_DIR = os.path.join(MODEL_BASE_DIR, f"{model_type}/ft_models")
    SFT_MODELS = ['baseline']
    logger.info(f'正在加载模型>>>>{model_name_or_path}\n')
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(MODEL_BASE_DIR, model_name_or_path),
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(MODEL_BASE_DIR, model_name_or_path))
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(MODEL_BASE_DIR, model_name_or_path))
    generation_config = GenerationConfig.from_pretrained(
        os.path.join(MODEL_BASE_DIR, model_name_or_path))
    generation_config.max_length = 8192
    model.generation_config = generation_config
    model.__class__.generate_stream = NewGenerationMixin.generate
    model.__class__.sample_stream = NewGenerationMixin.sample_stream

    if os.path.exists(SFT_MODEL_DIR):
        logger.info(f"==== 正在加载sft模型 ====\n")
        sft_models = os.listdir(SFT_MODEL_DIR)
        for model_name in sft_models:
            model_id = os.path.join(SFT_MODEL_DIR, model_name)
            try:
                if isinstance(model, PeftModel):
                    model.load_adapter(model_id, adapter_name=model_name)
                else:
                    model = PeftModel.from_pretrained(model, model_id, adapter_name=model_name)
            except Exception as e:
                logger.error(f"---> 加载 {model_id} 失败：{str(e)}\n")
            else:
                SFT_MODELS.append(model_name)
                logger.info(f"---> 加载 {model_id} 成功\n")

    model = model.eval()
    logger.info(f"==== sft_models {SFT_MODELS} ====\n")
    return {
        "model_type": model_type,
        'model': model,
        'tokenizer': tokenizer,
        'adapters': SFT_MODELS
    }


def get_chatglm3():
    model_type = "chatglm"
    model_name_or_path = f"{model_type}/ZhipuAI/chatglm3-6b"
    if not os.path.exists(os.path.join(MODEL_BASE_DIR, model_name_or_path)):
        auto_download(model_type, revision="v1.0.2")
    SFT_MODEL_DIR = os.path.join(MODEL_BASE_DIR, f"{model_type}/ft_models")
    SFT_MODELS = ['baseline']

    logger.info(f'正在加载模型>>>>{model_name_or_path}\n')
    if torch.cuda.is_available():
        model = AutoModel.from_pretrained(
            os.path.join(MODEL_BASE_DIR, model_name_or_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        model = AutoModel.from_pretrained(
            os.path.join(MODEL_BASE_DIR, model_name_or_path),
            trust_remote_code=True
        ).float()
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(MODEL_BASE_DIR, model_name_or_path),
        trust_remote_code=True)

    model.generation_config = GenerationConfig(
        max_length=8192,
        num_beams=1,
        do_sample=True,
        top_p=0.8,
        temperature=0.8
    )
    model.__class__.generate_stream = NewGenerationMixin.generate
    model.__class__.sample_stream = NewGenerationMixin.sample_stream

    if os.path.exists(SFT_MODEL_DIR):
        logger.info(f"==== 正在加载sft模型 ====\n")
        sft_models = os.listdir(SFT_MODEL_DIR)
        for model_name in sft_models:
            model_id = os.path.join(SFT_MODEL_DIR, model_name)
            try:
                if isinstance(model, PeftModel):
                    model.load_adapter(model_id, adapter_name=model_name)
                else:
                    model = PeftModel.from_pretrained(model, model_id, adapter_name=model_name)
            except Exception as e:
                logger.error(f"---> 加载 {model_id} 失败：{str(e)}\n")
            else:
                SFT_MODELS.append(model_name)
                logger.info(f"---> 加载 {model_id} 成功\n")

    model = model.eval()
    logger.info(f"==== sft_models {SFT_MODELS} ====\n")
    return {
        "model_type": model_type,
        "model": model,
        'tokenizer': tokenizer,
        'adapters': SFT_MODELS
    }


def get_baichuan2():
    '''加载百川基座模型'''
    model_type = 'baichuan'
    model_name_or_path = f"{model_type}/baichuan-inc/Baichuan2-13B-Chat"
    if not os.path.exists(os.path.join(MODEL_BASE_DIR, model_name_or_path)):
        auto_download(model_type, revision="v2.0.1")
    SFT_MODEL_DIR = os.path.join(MODEL_BASE_DIR, f"{model_type}/ft_models")
    SFT_MODELS = ['baseline']

    logger.info(f'正在加载模型>>>>{model_name_or_path}\n')
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(MODEL_BASE_DIR, model_name_or_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(MODEL_BASE_DIR, model_name_or_path),
            trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(MODEL_BASE_DIR, model_name_or_path),
        trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(
        os.path.join(MODEL_BASE_DIR, model_name_or_path),
        trust_remote_code=True)
    model.generation_config = generation_config

    if os.path.exists(SFT_MODEL_DIR):
        logger.info(f"==== 正在加载sft模型 ====\n")
        sft_models = os.listdir(SFT_MODEL_DIR)
        for model_name in sft_models:
            model_id = os.path.join(SFT_MODEL_DIR, model_name)
            try:
                if isinstance(model, PeftModel):
                    model.load_adapter(model_id, adapter_name=model_name)
                else:
                    model = PeftModel.from_pretrained(model, model_id, adapter_name=model_name)
            except Exception as e:
                logger.error(f"---> 加载 {model_id} 失败：{str(e)}\n")
            else:
                SFT_MODELS.append(model_name)
                logger.info(f"---> 加载 {model_id} 成功\n")

    model = model.eval()
    logger.info(f"==== sft_models {SFT_MODELS} ====\n")
    return {
        "model_type": model_type,
        'model': model,
        'tokenizer': tokenizer,
        'adapters': SFT_MODELS
    }


def get_qwen2():
    '''加载千问大模型'''
    model_type = "qwen"
    model_name_or_path = f"{model_type}/qwen/Qwen1.5-7B-Chat"
    if not os.path.exists(os.path.join(MODEL_BASE_DIR, model_name_or_path)):
        auto_download(model_type, repair_name="Qwen1.5-7B-Chat")
    SFT_MODEL_DIR = os.path.join(MODEL_BASE_DIR, f"{model_type}/ft_models")
    SFT_MODELS = ['baseline']

    logger.info(f'正在加载模型>>>>{model_name_or_path}\n')
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(MODEL_BASE_DIR, model_name_or_path),
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(MODEL_BASE_DIR, model_name_or_path))
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(MODEL_BASE_DIR, model_name_or_path))
    generation_config = GenerationConfig.from_pretrained(
        os.path.join(MODEL_BASE_DIR, model_name_or_path))
    generation_config.max_length = 8192
    model.generation_config = generation_config
    model.__class__.generate_stream = NewGenerationMixin.generate
    model.__class__.sample_stream = NewGenerationMixin.sample_stream

    if os.path.exists(SFT_MODEL_DIR):
        logger.info(f"==== 正在加载sft模型 ====\n")
        sft_models = os.listdir(SFT_MODEL_DIR)
        for model_name in sft_models:
            model_id = os.path.join(SFT_MODEL_DIR, model_name)
            try:
                if isinstance(model, PeftModel):
                    model.load_adapter(model_id, adapter_name=model_name)
                else:
                    model = PeftModel.from_pretrained(model, model_id, adapter_name=model_name)
            except Exception as e:
                logger.error(f"---> 加载 {model_id} 失败：{str(e)}\n")
            else:
                SFT_MODELS.append(model_name)
                logger.info(f"---> 加载 {model_id} 成功\n")

    model = model.eval()
    logger.info(f"==== sft_models {SFT_MODELS} ====\n")
    return {
        "model_type": model_type,
        'model': model,
        'tokenizer': tokenizer,
        'adapters': SFT_MODELS
    }


def process_msg(messages: List[ChatMessage]):
    _messages = messages
    messages = []

    for i, m in enumerate(_messages):

        if i != 0 and m.role == "system":
            continue
        # if i == 0 and m.role == "system":
        # messages.append({
        #     'role': m.role,
        #     'content': f'\n你是小花，一个由小花AI训练的大语言模型。\n知识截至日期：2023-04\n当前型号：Baichuan2-13B-Chat\n当前时间：{datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")}\n\n'
        # })
        # messages.append({
        #     'role': m.role,
        #     'content': ''
        # })
        # else:
        messages.append({
            "role": m.role,
            "content": m.content
        })
    return messages


def load_completions(file) -> ahocorasick.Automaton:
    # 加载数据到AC自动机
    A = ahocorasick.Automaton()
    with open(file, 'r', encoding='utf8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        a, b = line.split()
        A.add_word(a, f'{a}&&{b}')
    A.make_automaton()
    return A


def enhance_text(A: ahocorasick.Automaton, text: str) -> str:
    for _, words in A.iter(text):
        src, dst = words.split("&&")
        text = text.replace(src, dst)
    return text


if __name__ == '__main__':
    A = load_completions("completions.txt")
    enhance_text(A, "你擅长什么呢？当冲对技术的要求比较高")