import os, torch
from peft import PeftModel
from loguru import logger
from llms import LLMS
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from param_model import ChatMessage
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_BASE_DIR = os.path.join(BASE_DIR, "llms")


def get_chatglm():
    model_type = "chatglm"
    model_name_or_path = f"{model_type}/chatglm3-6b"
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
        "model_type": "chatglm",
        "model": model,
        'tokenizer': tokenizer,
        'adapters': SFT_MODELS
    }


def get_baichuan():
    '''加载百川基座模型'''
    model_type = 'baichuan'
    model_name_or_path = f"{model_type}/Baichuan2-13B-Chat"

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
        "model_type": "baichuan",
        'model': model,
        'tokenizer': tokenizer,
        'adapters': SFT_MODELS
    }


def get_models(*model_names) -> Dict[str, LLMS]:
    model_dict = {
        "chatglm": get_chatglm,
        "baichuan": get_baichuan
    }
    return {name: LLMS(**model_dict[name]()) for name in model_names}


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
