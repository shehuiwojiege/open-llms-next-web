import torch, uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from param_model import (
    ModelList,
    BaseResponse,
    ChatCompletionResponse,
)
from llms import load_custom_models
from lingua import Language, LanguageDetectorBuilder
languages = [Language.CHINESE, Language.ENGLISH, Language.JAPANESE]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

# 加载模型
load_custom_models("qwen2")

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

'''the following interfaces are not public'''
# from posts_api import (
#     copy_writing,
#     create_chat,
#     create_comment,
#     acquire_prompt,
#     create_translate,
# )
# app.post(
#     '/v1/copywriting',
#     response_model=BaseResponse)(copy_writing)
# app.post(
#     '/api/gpt/comments',
#     summary="帖子评论",
#     operation_id="create_comment",
#     response_model=BaseResponse)(create_comment)
# app.get(
#     "/api/gpt/prompts",
#     summary="获取预置提示词",
#     operation_id="acquire_prompt",
#     response_model=BaseResponse)(acquire_prompt)
# app.post(
#     '/api/gpt/trans',
#     summary="文本翻译",
#     operation_id="create_translate",
#     response_model=BaseResponse)(create_translate)
# app.post(
#     '/api/gpt/chat/completions',
#     summary="与gpt对话",
#     operation_id="create_chat",
#     response_model=ChatCompletionResponse)(create_chat)

'''openai api'''
from openai_api import create_chat_completion, list_models
app.get(
    '/v1/models',
    response_model=ModelList)(list_models)
app.post(
    '/v1/chat/completions',
    response_model=ChatCompletionResponse)(create_chat_completion)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)
