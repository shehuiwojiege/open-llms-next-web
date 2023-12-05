import torch, uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from loguru import logger
from transformers import GenerationConfig
from fastapi.middleware.cors import CORSMiddleware
from utils import (
    get_models,
    process_msg,
)
from param_model import (
    ModelCard,
    ModelList,
    ChatMessage,
    DeltaMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
)
from sse_starlette.sse import EventSourceResponse


MODEL_DICT = get_models("chatglm", "baichuan")


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/v1/models', response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="baichuan2-13B-chat")
    return ModelList(data=[model_card])


@app.post('/v1/chat/completions', response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    model_splits = request.model.split('__', 1)
    if len(model_splits) != 2:
        raise HTTPException(status_code=400, detail=f"model name error: {request.model}")
    prefix, checkpoint_id = model_splits
    if prefix.startswith('Baichuan'):
        model = MODEL_DICT["baichuan"]
        model_type = "baichuan"
    elif prefix.startswith('chatglm'):
        model = MODEL_DICT["chatglm"]
        model_type = "chatglm"
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
    generate_config = GenerationConfig(
        temperature=request.temperature,
        top_p=request.top_p,
        do_sample=True,
        max_new_tokens=request.max_tokens or 512,
        repetition_penalty=request.repetition_penalty,
    )

    logger.info(f"==== generation_config ====\n{generate_config}")

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

    gen_config = GenerationConfig(
        temperature=params["temperature"],
        top_p=params["top_p"],
        max_new_tokens=params["max_tokens"],
        do_sample=True,
        repetition_penalty=params["repetition_penalty"]
    )

    logger.info(f"==== generation_config ====\n{gen_config}")

    messages = process_msg(params["messages"])
    # logger.debug(f"==== messages ====\n{messages}")
    streamer = model.chat(checkpoint_id, messages, generation_config=gen_config)
    current = ""

    for token in streamer:
        tt = token
        token = token.replace(current, "")

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
        current = tt

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=params["model"], choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True))
    yield '[DONE]'


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)
