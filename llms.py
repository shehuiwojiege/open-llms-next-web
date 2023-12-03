import torch
from queue import Queue
from peft import PeftModel
from loguru import logger
from threading import Thread


class LLMS:
    def __init__(self, model_type, model, tokenizer, adapters):
        self.__model_type = model_type
        self.__model = model
        self.__tokenizer = tokenizer
        self.__adapters = adapters

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

    def _chatglm_chat(self, tokenizer, messages, generation_config, stream=True):
        query, role = messages[-1]["content"], messages[-1]["role"]
        history = messages[:-1]
        inputs = self._chatglm_build_input(tokenizer, query, history)
        inputs = inputs.to(self.__model.device)
        eos_token_id = [
            tokenizer.eos_token_id,
            tokenizer.get_command("<|user|>"),
        ]
        if stream:
            streamer = ChatGLMStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True)
            Thread(target=self.__model.generate, kwargs=dict(
                **inputs, streamer=streamer,
                eos_token_id=eos_token_id, generation_config=generation_config
            )).start()
            return streamer
        else:
            outputs = self.__model.generate(**inputs, generation_config=generation_config,
                                            eos_token_id=eos_token_id)
            response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            return response

    @torch.inference_mode()
    def chat(self, checkpoint_id, messages, generation_config, stream=True):
        logger.info(f"=== checkpoint_id ====\n{checkpoint_id}\n")
        logger.info(f"==== messages ====\n{messages}\n")
        assert checkpoint_id in self.__adapters, f'{checkpoint_id} is not exists'
        if isinstance(self.__model, PeftModel):
            if checkpoint_id == "baseline":
                self.__model.base_model.disable_adapter_layers()
            else:
                self.__model.base_model.enable_adapter_layers()
                self.__model.set_adapter(checkpoint_id)
        if self.__model_type == "chatglm":
            return self._chatglm_chat(self.__tokenizer, messages, stream=stream,
                                      generation_config=generation_config)
        return self.__model.chat(self.__tokenizer, messages, stream=stream, generation_config=generation_config)


class ChatGLMStreamer:
    def __init__(self, tokenizer, skip_prompt=False, skip_special_tokens=False):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.tokens = []
        self.text_queue = Queue()
        self.next_tokens_are_prompt = True

    def put(self, value):
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
        else:
            if len(value.shape) > 1:
                value = value[0]
            self.tokens.extend(value.tolist())
            self.text_queue.put(
                self.tokenizer.decode(self.tokens, skip_special_tokens=self.skip_special_tokens))

    def end(self):
        self.text_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get()
        if value is None:
            raise StopIteration()
        else:
            return value