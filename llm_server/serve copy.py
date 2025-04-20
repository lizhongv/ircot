import os
import time
import torch
import sys
import json

from functools import lru_cache
from fastapi import FastAPI, Response

from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.utils.import_utils import is_torch_bf16_gpu_available
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if True:
    from log.logging_config import logger, LYELLOW, RESET


@lru_cache(maxsize=None)
def get_model_and_tokenizer():
    """
    Load the model and tokenizer based on the MODEL_NAME environment variable.
    """
    model_shortname = os.environ["MODEL_NAME"]
    valid_model_shortnames = [
        "flan-t5-base",
        "flan-t5-large",
        "flan-t5-xl",
        "flan-t5-xxl",

        "qwen2.5-0.5B",
        "qwen2.5-1.5B",
        "qwen2.5-7B",
    ]

    assert model_shortname in valid_model_shortnames, f"Model name {model_shortname} not in {valid_model_shortnames}"

    hf_device_map = {"shared": 1, "encoder": 0, "decoder": 0, "lm_head": 1}

    model_mapping = {
        "flan-t5-small": ("/data1/Public/LLMs/flan-t5-small", T5ForConditionalGeneration, {"device_map": "auto", "torch_dtype": "auto"}, T5Tokenizer, {"legacy": True}),
        "flan-t5-base": ("/data1/Public/LLMs/flan-t5-base", T5ForConditionalGeneration, {"device_map": "auto", "torch_dtype": "auto"}, T5Tokenizer, {"legacy": True}),
        "flan-t5-large": ("/data1/Public/LLMs/flan-t5-large", T5ForConditionalGeneration, {"device_map": "auto", "torch_dtype": "auto"}, T5Tokenizer, {"legacy": True}),
        "flan-t5-xl": ("/data1/Public/LLMs/flan-t5-xl", T5ForConditionalGeneration, {"device_map": "auto", "torch_dtype": "auto"}, T5Tokenizer, {"legacy": True}),
        "flan-t5-xxl": ("/data1/Public/LLMs/flan-t5-xxl", T5ForConditionalGeneration, {"device_map": hf_device_map, "torch_dtype": "auto"}, T5Tokenizer, {"legacy": True}),

        "qwen2.5-0.5B": ("/data1/Public/LLMs/Qwen2.5-0.5B-Instruct", AutoModelForCausalLM, {"device_map": "auto", "torch_dtype": "auto"}, AutoTokenizer, {}),
        "qwen2.5-1.5B": ("/data1/Public/LLMs/Qwen2.5-1.5B-Instruct", AutoModelForCausalLM, {"device_map": "auto", "torch_dtype": "auto"}, AutoTokenizer, {}),
        "qwen2.5-7B": ("/data1/Public/LLMs/Qwen2.5-7B-Instruct", AutoModelForCausalLM, {"device_map": "auto", "torch_dtype": "auto"}, AutoTokenizer, {}),
    }
    # "google/flan-t5-base" "Qwen/Qwen2.5-1.5B-Instruct"

    try:
        model_name, model_cls, model_kwargs, tokenizer_cls, tokenizer_kwargs = model_mapping[model_shortname]

        model = model_cls.from_pretrained(model_name, **model_kwargs)
        tokenizer = tokenizer_cls.from_pretrained(model_name, **tokenizer_kwargs)

        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model and tokenizer: {e}.")
        raise


class EOSReachedCriteria(StoppingCriteria):
    # Use this when EOS is not a single id, but a sequence of ids, e.g. for a custom EOS text.
    def __init__(self, tokenizer: AutoTokenizer, eos_text: str):
        self.tokenizer = tokenizer
        self.eos_text = eos_text
        assert (len(self.tokenizer.encode(eos_text)) < 10), "EOS text can't be longer then 10 tokens. It makes stopping_criteria check slow."

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        decoded_text = self.tokenizer.decode(input_ids[0][-10:])
        condition1 = decoded_text.endswith(self.eos_text)
        condition2 = decoded_text.strip().endswith(self.eos_text.strip())
        return condition1 or condition2


app = FastAPI()


@app.get("/")
async def index():
    model_shortname = os.environ["MODEL_NAME"]
    return {"message": f"Hello! This is a server for {model_shortname}. " "Go to /generate/ for generation requests."}


@app.get("/generate/")  # not post
async def generate(
    prompt: str,
    max_input: int = None,  # TODO
    max_length: int = 200,
    min_length: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
    repetition_penalty: float = None,
    length_penalty: float = None,
    eos_text: str = None,
    keep_prompt: bool = False,
):
    start_time = time.time()

    model_shortname = os.environ["MODEL_NAME"]
    model, tokenizer = get_model_and_tokenizer()

    # Add more stopping criteria
    stopping_criteria_list = StoppingCriteriaList()
    if eos_text:
        stopping_criteria = EOSReachedCriteria(tokenizer=tokenizer, eos_text=eos_text)
        stopping_criteria_list = StoppingCriteriaList([stopping_criteria])

    if model_shortname.startswith("qwen2.5"):
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)  # TODO max_new_tokens
        generated_ids = model.generate(**model_inputs, max_new_tokens=max_length, min_length=min_length, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
                                       num_return_sequences=num_return_sequences, repetition_penalty=repetition_penalty, length_penalty=length_penalty, stopping_criteria=stopping_criteria_list)
        generated_num_tokens = [len(ids) for ids in generated_ids]

        if not keep_prompt:
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)]
            generated_num_tokens = [len(ids) for ids in generated_ids]

        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        serializable_model_inputs = {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v for k, v in model_inputs.data.items()}
        logger.debug(f"Input text: {LYELLOW}{text}{RESET}")
        logger.debug(f"Model inputs: {json.dumps(serializable_model_inputs, indent=4)}")
        logger.debug(f"Generated texts: {LYELLOW}{generated_texts}{RESET}")

    elif model_shortname.startswith("flan-t5"):
        text = prompt
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)  # TODO max_length, t5 is encoder-decoder only output response
        generated_ids = model.generate(**model_inputs, max_length=max_length,   min_length=min_length, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
                                       num_return_sequences=num_return_sequences, repetition_penalty=repetition_penalty, length_penalty=length_penalty, stopping_criteria=stopping_criteria_list)
        generated_num_tokens = [len(ids) for ids in generated_ids]

        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        if keep_prompt:
            generated_texts = [prompt + text for text in generated_texts]
            generated_num_tokens = [len(tokenizer.encode(text)) for text in generated_texts]

        serializable_model_inputs = {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v for k, v in model_inputs.data.items()}
        logger.debug(f"Input text: {LYELLOW}{text}{RESET}")
        logger.debug(f"Model inputs: {json.dumps(serializable_model_inputs, indent=4)}")
        logger.debug(f"Generated texts: {LYELLOW}{generated_texts}{RESET}")

    end_time = time.time()
    run_time_in_seconds = end_time - start_time

    result = {
        "generated_num_tokens": generated_num_tokens,
        "generated_texts": generated_texts,
        "run_time_in_seconds": run_time_in_seconds,
        "model_name": model_shortname,
    }
    return Response(content=json.dumps(result), media_type="application/json")


# if __name__ == "__main__":
#     # os.environ['MODEL_NAME'] = 'qwen2.5-1.5B'
#     os.environ['MODEL_NAME'] = 'flan-t5-base'
#     get_model_and_tokenizer()
#     logger.info("Loaded model and tokenizer.")

# MODEL_NAME="qwen2.5-1.5B" uvicorn serve:app --host 0.0.0.0 --port 8010 --app-dir llm_server
# curl http://localhost:8010/
# curl "http://localhost:8010/generate/?prompt=Give%20me%20a%20short%20introduction%20to%20large%20language%20model.&max_length=200&do_sample=false&temperature=1.0&top_k=50&top_p=1.0&eos_text=multilingual%2C%20multilingual%2C&keep_prompt=true"
# curl "http://localhost:8010/generate/?prompt=Give%20me%20a%20short%20introduction%20to%20large%20language%20model.&max_length=200&do_sample=true&temperature=0.7&top_k=50&top_p=0.9&eos_text=end&keep_prompt=true"

# MODEL_NAME="flan-t5-large" uvicorn serve:app --host 0.0.0.0 --port 8010 --app-dir llm_server
# curl "http://127.0.0.1:8010/generate/?prompt=Transformers%20are%20a%20type%20of%20model%20architecture%20that&max_length=200&do_sample=false&temperature=1.0&top_k=50&top_p=1.0&eos_text=end&keep_prompt=true"
# curl "http://127.0.0.1:8010/generate/?prompt=Transformers%20are%20a%20type%20of%20model%20architecture%20that&max_length=200&do_sample=true&temperature=0.7&top_k=50&top_p=0.9&eos_text=end&keep_prompt=true"

# export MODEL_NAME="flan-t5-base"
# export MODEL_NAME="flan-t5-xxl"

# cd ircot 
# export MODEL_NAME="qwen2.5-7B"
# nohup  uvicorn serve:app --host 0.0.0.0 --port 8010 --app-dir llm_server > llm_server.log 2>&1 &
# tail -f llm_server.log  

    """
    curl -X POST "http://127.0.0.1:8010/generate/" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请介绍一下东北大学",
    "max_length": 200,
    "do_sample": true,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "eos_text": "end",
    "keep_prompt": true
  }'
    """
