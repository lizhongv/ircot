import os
import time
from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import Optional
import json
from openai import OpenAI

app = FastAPI()

# vLLM客户端配置（保持与原始代码相同的参数默认值）
VLLM_CLIENT = OpenAI(
    base_url="http://localhost:8020/v1",
    api_key="EMPTY"
)

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 200
    min_length: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    num_return_sequences: int = 1
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    eos_text: Optional[str] = None
    keep_prompt: bool = False

@app.get("/")
async def index():
    return {"message": "Hello! This is a server for Qwen2.5-7B. Go to /generate/ for generation requests."}

@app.post("/generate/")
async def generate_post(request: GenerationRequest):
    """完全兼容原始POST接口的输入输出格式"""
    start_time = time.time()
    
    # 构造messages（保持原始prompt处理逻辑）
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": request.prompt}
    ]
    
    # 调用vLLM（映射原始参数）
    completion = VLLM_CLIENT.chat.completions.create(
        model="Qwen2-7B-Instruct",
        messages=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_length,
        stop=request.eos_text,
        n=request.num_return_sequences
    )
    
    # 构造与原始代码完全一致的返回结构
    generated_texts = [choice.message.content for choice in completion.choices]
    if request.keep_prompt:
        generated_texts = [request.prompt + text for text in generated_texts]
    
    result = {
        "generated_texts": generated_texts,
        "generated_num_tokens": [len(text.split()) for text in generated_texts],
        "run_time_in_seconds": time.time() - start_time,
        "model_name": "Qwen2.5-7B"
    }
    
    return Response(content=json.dumps(result), media_type="application/json")

@app.get("/generate/")
async def generate_get(
    prompt: str,
    max_length: int = 200,
    min_length: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
    repetition_penalty: Optional[float] = None,
    length_penalty: Optional[float] = None,
    eos_text: Optional[str] = None,
    keep_prompt: bool = False,
):
    """完全兼容原始GET接口的输入输出格式"""
    request = GenerationRequest(
        prompt=prompt,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        eos_text=eos_text,
        keep_prompt=keep_prompt
    )
    return await generate_post(request)