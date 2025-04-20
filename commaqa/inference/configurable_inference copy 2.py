import argparse
import json
import time
import os
import sys
from tqdm import tqdm
import _jsonnet

# Environment setup
os.chdir("/data0/lizhong/multi_hop_rag/ircot")
sys.path.append("/data0/lizhong/multi_hop_rag/ircot")
from commaqa.inference.utils import get_environment_variables
from commaqa.inference.participant_execution import ExecutionParticipant
from commaqa.inference.data_instances import StructuredDataInstance
from commaqa.inference.model_search import ModelController, BestFirstDecomposer
from commaqa.inference.dataset_readers import DatasetReader
from commaqa.inference.constants import MODEL_NAME_CLASS, READER_NAME_CLASS
from log.logging_config import logger

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="QA dataset processor")
    parser.add_argument("--input", help="Input QA file")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--config", required=True, help="Model config file")
    parser.add_argument("--reader", choices=READER_NAME_CLASS.keys(), help="Dataset reader")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--silent", action="store_true", help="Suppress output")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads")
    parser.add_argument("--openai-api-keyname", help="OpenAI API key env var name")
    parser.add_argument("--llm-server-key-suffix", default="", help="LLM server key suffix")
    parser.add_argument("--variable-replacements", default="", help="JSON string for jsonnet vars")
    return parser.parse_args()

def build_decomposer(config_map):
    """Initialize models and build decomposer"""
    model_map = {}
    for key, value in config_map["models"].items():
        class_name = value.pop("name")
        if class_name not in MODEL_NAME_CLASS:
            raise ValueError(f"Unknown model class: {class_name}")
        
        model = MODEL_NAME_CLASS[class_name](**value)
        config_map[key] = model.query
        model_map[key] = model

    # Special handling for ExecutionParticipant
    for model in model_map.values():
        if isinstance(model, ExecutionParticipant):
            model.set_model_lib(model_map)

    return BestFirstDecomposer(ModelController(config_map, StructuredDataInstance)), model_map

def load_config(config_file, var_replacements=""):
    """Load config with optional variable replacements"""
    if config_file.endswith(".jsonnet"):
        ext_vars = get_environment_variables()
        if var_replacements:
            from run import instatiate_config
            with open(config_file) as f:
                content = instatiate_config(f.read(), json.loads(var_replacements))
            temp_file = os.path.join(os.path.dirname(config_file), os.urandom(16).hex())
            with open(temp_file, "w") as f:
                f.write(content)
            config = json.loads(_jsonnet.evaluate_file(temp_file, ext_vars))
            os.remove(temp_file)
            return config
        return json.loads(_jsonnet.evaluate_file(config_file, ext_vars))
    
    if var_replacements:
        raise ValueError("Variable replacement only supported for jsonnet")
    return json.load(open(config_file))

def run_inference(args, reader, decomposer, override_answer=None):
    """Run inference on input data"""
    start = time.time()
    qid_answer_chains = []
    
    if args.threads > 1:
        import multiprocessing as mp
        with mp.Pool(args.threads) as pool:
            qid_answer_chains = pool.map(decomposer.return_qid_prediction, reader.read_examples(args.input))
    else:
        for example in tqdm(reader.read_examples(args.input), desc="Processing"):
            qid_answer_chains.append(decomposer.return_qid_prediction(
                example, override_answer_by=override_answer, 
                debug=args.debug, silent=args.silent))

    # Save results
    base_path = os.path.splitext(args.output)[0]
    with open(args.output, "w") as f:
        json.dump({x[0]: x[1] for x in qid_answer_chains}, f, indent=4)
    
    # Save metadata
    for suffix, content in [
        ("_time_taken.txt", str(round(time.time() - start))),
        ("_chains.txt", "\n".join(x[2] for x in qid_answer_chains)),
        ("_full_eval_path.txt", args.input),
        ("_variable_replacements.json", args.variable_replacements)
    ]:
        with open(base_path + suffix, "w") as f:
            f.write(content)

def main():
    args = parse_args()
    
    # API key handling
    if args.openai_api_keyname:
        os.environ["OPENAI_API_KEY"] = os.environ[args.openai_api_keyname]
        import openai
        openai.api_key = os.environ[args.openai_api_keyname]
    
    if args.llm_server_key_suffix:
        os.environ["LLM_SERVER_KEY_SUFFIX"] = args.llm_server_key_suffix

    # Initialize components
    config = load_config(args.config, args.variable_replacements)
    decomposer, models = build_decomposer(config)
    reader = READER_NAME_CLASS[config.get("reader", {}).get("name", args.reader)](**config.get("reader", {}).pop("config", {}))

    # Run appropriate mode
    if args.demo:
        # Demo mode implementation here
        pass
    else:
        run_inference(args, reader, decomposer, config.get("override_answer_by"))

if __name__ == "__main__":
    main()