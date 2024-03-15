import torch
# from queue import Queue
from typing import Union, List, Optional
from peft import PeftModel
from loguru import logger
# from threading import Thread
from typing import Dict
from transformers import GenerationConfig, PreTrainedTokenizer
from transformers_stream_generator.main import StreamGenerationConfig
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    StoppingCriteria as _StoppingCriteria,
)


class ChatStreamer:
    def __init__(self, tokenizer: PreTrainedTokenizer, generation_stream):
        self.tokenizer = tokenizer
        self.generation_stream = generation_stream

    def __iter__(self):
        return self

    def __next__(self):
        return self.tokenizer.decode(self.generation_stream.__next__(), skip_special_tokens=True)


class StoppingCriteria(_StoppingCriteria):

    def __init__(self, stop_token_ids: List[int] = None):
        self.stop_token_ids = stop_token_ids

    def __call__(self,  input_ids: torch.Tensor, scores: torch.Tensor) -> bool:
        return input_ids[0].tolist()[-1] in self.stop_token_ids


class LLMS:
    def __init__(self, model_type, model, tokenizer, adapters):
        self.__model_type = model_type
        self.__model = model
        self.__tokenizer = tokenizer
        self.__adapters = adapters

    '''
    # 旧版glm build_input
    def _chatglm_build_input(self, tokenizer, query, history):
    user_token_id = tokenizer.get_command("<|user|>")
    assistant_token_id = tokenizer.get_command("<|assistant|>")
    system_token_id = tokenizer.get_command("<|system|>")
    inner_tokenizer = tokenizer.tokenizer
    line_content = tokenizer.tokenizer.encode("\n")
    line_token = [13]
    input_ids = []
    system = None
    if len(history) > 0:
        if history[0]["role"] == "system":
            system = history.pop(0)
            input_ids += [system_token_id] + line_content
            input_ids += inner_tokenizer.encode(system["content"])
    for his in history:
        if his["role"] == "user":
            if system is not None:
                input_ids += line_token
            input_ids += [user_token_id] + line_content
            input_ids += inner_tokenizer.encode(his["content"])
        elif his["role"] == "assistant":
            input_ids += [assistant_token_id] + line_content
            input_ids += inner_tokenizer.encode(his["content"])
    if system is not None:
        input_ids += line_token
    input_ids += [user_token_id] + line_content
    input_ids += inner_tokenizer.encode(query) + [assistant_token_id] + line_content
    return tokenizer.batch_encode_plus([input_ids], return_tensors="pt", is_split_into_words=True)
    '''

    def _stream_chat(
            self,
            inputs,
            tokenizer: PreTrainedTokenizer,
            generation_config: GenerationConfig,
            eos_token_id: List[int]
    ) -> ChatStreamer:
        stream_config = StreamGenerationConfig(
            **generation_config.to_dict(),
            do_stream=True
        )
        if tokenizer.eos_token_id is not None:
            stream_config.eos_token_id = tokenizer.eos_token_id
        if tokenizer.pad_token_id is not None:
            stream_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.bos_token_id is not None:
            stream_config.bos_token_id = tokenizer.bos_token_id
        stopping_criteria = StoppingCriteriaList([StoppingCriteria(eos_token_id)])
        return ChatStreamer(
            tokenizer,
            self.__model.generate_stream(
                **inputs,
                generation_config=stream_config,
                stopping_criteria=stopping_criteria,
                seed=-1
            )
        )

    def _qwen2_chat(self, tokenizer, messages, generation_config: Optional[GenerationConfig] = None, stream: bool = True):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        eos_token_id = self.__model.generation_config.eos_token_id
        inputs = tokenizer.batch_encode_plus([text], return_tensors="pt").to(self.__model.device)
        if stream:
            response = self._stream_chat(inputs, tokenizer, generation_config, eos_token_id)
        else:
            outputs = self.__model.generate(**inputs, generation_config=generation_config,
                                            eos_token_id=eos_token_id)
            response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        return response

    def _chatglm_chat(self, tokenizer, messages, generation_config: Optional[GenerationConfig] = None, stream: bool = True) -> Union[ChatStreamer, str]:
        query, role = messages[-1]["content"], messages[-1]["role"]
        history = messages[:-1]
        # 旧版 ---> 已舍弃
        # inputs = self._chatglm_build_input(tokenizer, query, history).to(self.__model.device)
        inputs = tokenizer.build_chat_input(query, history).to(self.__model.device)
        eos_token_id = [
            tokenizer.eos_token_id,
            tokenizer.get_command("<|user|>"),
            tokenizer.get_command("<|observation|>")
        ]
        if stream:
            response = self._stream_chat(inputs, tokenizer, generation_config, eos_token_id)
        else:
            outputs = self.__model.generate(**inputs, generation_config=generation_config,
                                            eos_token_id=eos_token_id)
            response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        return response

    @property
    def generation_config(self):
        return self.__model.generation_config

    @torch.inference_mode()
    def chat(self, checkpoint_id: str, messages: List, generation_config: Optional[GenerationConfig] = None, stream: bool = True):
        logger.info(f"=== checkpoint_id ====\n{checkpoint_id}\n")
        logger.info(f"==== messages ====\n{messages}\n")
        assert checkpoint_id in self.__adapters, f'{checkpoint_id} is not exists'
        if isinstance(self.__model, PeftModel):
            if checkpoint_id == "baseline":
                self.__model.base_model.disable_adapter_layers()
            else:
                self.__model.base_model.enable_adapter_layers()
                self.__model.set_adapter(checkpoint_id)
        if generation_config is None:
            generation_config = self.__model.generation_config
        logger.info(f"==== generation_config ====\n{generation_config}\n")
        if self.__model_type == "chatglm":
            return self._chatglm_chat(
                self.__tokenizer, messages,
                stream=stream,
                generation_config=generation_config)
        elif self.__model_type == "qwen2":
            return self._qwen2_chat(
                self.__tokenizer,
                messages,
                stream=stream,
                generation_config=generation_config)
        return self.__model.chat(self.__tokenizer, messages, stream=stream, generation_config=generation_config)


MODEL_DICT: Dict[str, LLMS] = None


def load_custom_models(*model_names):
    global MODEL_DICT
    from utils import (
        get_qwen2,
        get_chatglm,
        get_baichuan,
    )
    model_dict = {
        "chatglm": get_chatglm,
        "baichuan": get_baichuan,
        "qwen2": get_qwen2,
    }
    MODEL_DICT = {name: LLMS(**model_dict[name]()) for name in model_names}


'''
# 旧版 ChatStreamer
# class ChatStreamer:
#     def __init__(self, tokenizer, skip_prompt=False, skip_special_tokens=False):
#         self.tokenizer = tokenizer
#         self.skip_prompt = skip_prompt
#         self.skip_special_tokens = skip_special_tokens
#         self.tokens = []
#         self.text_queue = Queue()
#         self.next_tokens_are_prompt = True
#
#     def put(self, value):
#         if self.skip_prompt and self.next_tokens_are_prompt:
#             self.next_tokens_are_prompt = False
#         else:
#             if len(value.shape) > 1:
#                 value = value[0]
#             self.tokens.extend(value.tolist())
#             self.text_queue.put(
#                 self.tokenizer.decode(self.tokens, skip_special_tokens=self.skip_special_tokens))
#
#     def end(self):
#         self.text_queue.put(None)
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         value = self.text_queue.get()
#         if value is None:
#             raise StopIteration()
#         else:
#             return value
'''
