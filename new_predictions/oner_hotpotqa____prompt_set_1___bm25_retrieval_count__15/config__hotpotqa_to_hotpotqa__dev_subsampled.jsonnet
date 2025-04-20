# Set dataset:
local dataset = "hotpotqa";
local retrieval_corpus_name = dataset;
local add_pinned_paras = if dataset == "iirc" then true else false;
local valid_qids = ["5abb14bd5542992ccd8e7f07", "5ac2ada5554299657fa2900d", "5a758ea55542992db9473680", "5ae0185b55429942ec259c1b", "5a8ed9f355429917b4a5bddd", "5abfb3435542990832d3a1c1", "5ab92dba554299131ca422a2", "5a835abe5542996488c2e426", "5a89c14f5542993b751ca98a", "5a90620755429933b8a20508", "5a7bbc50554299042af8f7d0", "5a8f44ab5542992414482a25", "5add363c5542990dbb2f7dc8", "5a7fc53555429969796c1b55", "5a790e7855429970f5fffe3d"];
local prompt_reader_args = {
    "filter_by_key_values": {
        "qid": valid_qids
    },
    "order_by_key": "qid",
    "estimated_generation_length": 300,
    "shuffle": false,
    "model_length_limit": 8000,
};

# (Potentially) Hyper-parameters:
# null means it's unused.
local llm_retrieval_count = null;
local llm_map_count = null;
local bm25_retrieval_count = 15;
local rc_context_type_ = null; # Choices: no, gold, gold_with_n_distractors
local distractor_count = null; # Choices: 1, 2, 3
local rc_context_type = (
    if rc_context_type_ == "gold_with_n_distractors"
    then "gold_with_" + distractor_count + "_distractors"  else rc_context_type_
);
local rc_qa_type = null; # Choices: direct, cot

{
    "start_state": "retrieve_and_reset_paragraphs",
    "end_state": "[EOQ]",
    "models": {
        "retrieve_and_reset_paragraphs": {
            "name": "retrieve_and_reset_paragraphs",
            "retrieval_type": "bm25",
            "retriever_host": std.extVar("RETRIEVER_HOST"),
            "retriever_port": std.extVar("RETRIEVER_PORT"),
            "retrieval_count": bm25_retrieval_count,
            "global_max_num_paras": 15,
            "query_source": "original_question",
            "source_corpus_name": retrieval_corpus_name,
            "document_type": "title_paragraph_text",
            "return_pids": true,
            "end_state": "[EOQ]",
        },
    },
    "reader": {
        "name": "multi_para_rc",
        "add_paras": false,
        "add_gold_paras": false,
        "add_pinned_paras": add_pinned_paras,
    },
    "prediction_type": "pids"
}