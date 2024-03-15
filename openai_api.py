from fastapi import HTTPException
from loguru import logger
from utils import (
    process_msg,
)
from llms import MODEL_DICT
from param_model import (
    ModelCard,
    ModelList,
    DeltaMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
)
from sse_starlette.sse import EventSourceResponse


async def list_models():
    model_card = ModelCard(id="baichuan2-13B-chat")
    return ModelList(data=[model_card])


async def create_chat_completion(request: ChatCompletionRequest):
    model_splits = request.model.split('__', 1)
    if len(model_splits) != 2:
        raise HTTPException(status_code=400, detail=f"model name error: {request.model}")
    prefix, checkpoint_id = model_splits
    if prefix.startswith('chatglm'):
        model = MODEL_DICT["chatglm"]
        model_type = "chatglm"
    elif prefix.startswith('Baichuan'):
        model = MODEL_DICT["baichuan"]
        model_type = "baichuan"
    elif prefix.startswith("Qwen"):
        model = MODEL_DICT["qwen2"]
        model_type = "qwen2"
    else:
        raise HTTPException(status_code=400, detail=f"unsupported model type: {prefix}")

    logger.info(f"==== model type  ==== \n{model_type}")

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        model=request.model,
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 512,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
        functions=request.functions,
    )
    if request.stream:
        generate = predict(model_type, checkpoint_id, gen_params)
        return EventSourceResponse(generate, media_type="text/event-stream")

    messages = process_msg(request.messages)
    # logger.debug(f"==== messages ====\n{messages}")
    generate_config = model.generation_config
    if request.temperature:
        generate_config.temperature = request.temperature
    if request.top_p:
        generate_config.top_p = request.top_p
    if request.max_tokens:
        generate_config.max_tokens = request.max_tokens
    if request.repetition_penalty:
        generate_config.repetition_penalty = request.repetition_penalty

    res_text = model.chat(checkpoint_id, messages, generate_config, stream=False)

    message = ChatMessage(
        role="assistant",
        content=res_text,
    )

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
        finish_reason="stop",
    )
    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion", usage=None)


async def predict(model_type, checkpoint_id, params: dict):
    if model_type == "baichuan":
        model = MODEL_DICT["baichuan"]
    elif model_type == "qwen2":
        model = MODEL_DICT["qwen2"]
    else:
        model = MODEL_DICT["chatglm"]

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(
        model=params["model"],
        choices=[choice_data],
        object="chat.completion.chunk"
    )
    yield "{}".format(chunk.json(exclude_unset=True))

    gen_config = model.generation_config
    if params["temperature"]:
        gen_config.temperature = params["temperature"]
    if params["top_p"]:
        gen_config.top_p = params["top_p"]
    if params["max_tokens"]:
        gen_config.max_tokens = params["max_tokens"]
    if params["repetition_penalty"]:
        gen_config.repetition_penalty = params["repetition_penalty"]

    messages = process_msg(params["messages"])
    # logger.debug(f"==== messages ====\n{messages}")
    streamer = model.chat(checkpoint_id, messages, generation_config=gen_config)
    # current = ""

    for token in streamer:
        # tt = token
        # token = token.replace(current, "")
        delta = DeltaMessage(
            content=token,
            role="assistant",
            function_call=None,
        )
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=delta,
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=params["model"], choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.json(exclude_unset=True))
        # current = tt

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=params["model"], choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True))
    yield '[DONE]'