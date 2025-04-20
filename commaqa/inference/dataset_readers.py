import os
import json
import copy
from collections import Counter, defaultdict
import requests
import re

from diskcache import Cache
from tqdm import tqdm
import ftfy
import hashlib


def get_pid_for_title_paragraph_text(title: str, paragraph_text: str) -> str:
    title = ftfy.fix_text(title.strip())
    paragraph_text = ftfy.fix_text(paragraph_text.strip())

    if paragraph_text.startswith("Wikipedia Title: " + title + "\n"):
        paragraph_text = paragraph_text.replace("Wikipedia Title: " + title + "\n", "").strip()

    if paragraph_text.startswith("Wikipedia Title: " + title + " \n"):
        paragraph_text = paragraph_text.replace("Wikipedia Title: " + title + " \n", "").strip()

    if paragraph_text.startswith("Title: " + title + "\n"):
        paragraph_text = paragraph_text.replace("Title: " + title + "\n", "").strip()

    if paragraph_text.startswith("Title: " + title + " \n"):
        paragraph_text = paragraph_text.replace("Title: " + title + " \n", "").strip()

    title = "".join([i if ord(i) < 128 else " " for i in title]).lower()
    paragraph_text = "".join([i if ord(i) < 128 else " " for i in paragraph_text]).lower()

    title = re.sub(r" +", " ", title)
    paragraph_text = re.sub(r" +", " ", paragraph_text)

    # NOTE: This is more robust, but was done after V2 big exploration.
    # So uncomment it for rerunning evals for those experiments.
    title = re.sub(r" +", "", title)
    paragraph_text = re.sub(r" +", "", paragraph_text)

    # 生成唯一标识符，字符串MD5哈希值
    pid = "___".join(
        [
            "pid",
            hashlib.md5(title.encode("utf-8")).hexdigest(),
            hashlib.md5(paragraph_text.encode("utf-8")).hexdigest(),
        ]
    )

    return pid


class DatasetReader:
    def __init__(self, add_paras=False, add_gold_paras=False):
        self.add_paras = add_paras
        self.add_gold_paras = add_gold_paras

    def read_examples(self, file):
        return NotImplementedError("read_examples not implemented by " + self.__class__.__name__)


def format_drop_answer(answer_json):
    if answer_json["number"]:
        return answer_json["number"]
    if len(answer_json["spans"]):
        return answer_json["spans"]
    # only date possible
    date_json = answer_json["date"]
    if not (date_json["day"] or date_json["month"] or date_json["year"]):
        print("Number, Span or Date not set in {}".format(answer_json))
        return None
    return date_json["day"] + "-" + date_json["month"] + "-" + date_json["year"]


cache = Cache(os.path.expanduser("~/.cache/title_queries"))


def get_title_and_para(retriever_host, retriever_port, query_title):
    @cache.memoize()
    def _get_title_and_para(query_title):
        params = {
            "retrieval_method": "retrieve_from_elasticsearch",
            "query_text": query_title,
            "max_hits_count": 1,
            "document_type": "title",
        }
        url = retriever_host.rstrip("/") + ":" + str(retriever_port) + "/retrieve"
        result = requests.post(url, json=params)
        if result.ok:
            result = result.json()
            retrieval = result["retrieval"]
            if not retrieval:
                return None
            if query_title != retrieval[0]["title"]:
                print(f"WARNING: {query_title} != {retrieval[0]['title']}")
            return (retrieval[0]["title"], retrieval[0]["paragraph_text"])
        else:
            return None

    return _get_title_and_para(query_title)


class MultiParaRCReader(DatasetReader):
    def __init__(
        self,
        add_paras=False,
        add_gold_paras=False,
        add_pinned_paras=False,
        pin_position="no_op",
        remove_pinned_para_titles=False,
        max_num_words_per_para=None,
        retriever_host=None,
        retriever_port=None,
    ):
        super().__init__(add_paras, add_gold_paras)
        self.add_pinned_paras = add_pinned_paras
        self.pin_position = pin_position
        self.remove_pinned_para_titles = remove_pinned_para_titles
        self.max_num_words_per_para = max_num_words_per_para
        self.retriever_host = retriever_host
        self.retriever_port = retriever_port

        self.qid_to_external_paras = defaultdict(list)
        self.qid_to_external_titles = defaultdict(list)

    def read_examples(self, file):
        # First count total lines for accurate progress
        with open(file, "r") as input_fp:
            total_lines = sum(1 for _ in input_fp)

        with open(file, "r") as input_fp:
            for line_num, line in enumerate(tqdm(input_fp, total=total_lines, desc="Processing", unit="examples"), 1):
                if not line.strip():
                    continue

                try:
                    input_instance = json.loads(line)
                    qid = input_instance["question_id"]
                    query = question = input_instance["question_text"]
                    answers_objects = input_instance["answers_objects"]

                    # Format answers
                    formatted_answers = [
                        tuple(format_drop_answer(answers_object))
                        for answers_object in answers_objects
                    ]
                    answer = Counter(formatted_answers).most_common()[0][0]

                    output_instance = {
                        "qid": qid,
                        "query": query,
                        "answer": answer,
                        "question": question,
                    }

                    # Process paragraphs
                    title_paragraph_tuples = []

                    # Add pinned paragraphs if enabled
                    if self.add_pinned_paras:
                        title_paragraph_tuples.extend(
                            (p["title"], p["paragraph_text"])
                            for p in input_instance["pinned_contexts"]
                            if (p["title"], p["paragraph_text"]) not in title_paragraph_tuples
                        )

                    # Add regular or gold paragraphs
                    if self.add_paras:
                        assert not self.add_gold_paras, "Enable only one: add_paras or add_gold_paras"
                        title_paragraph_tuples.extend(
                            (p["title"], p["paragraph_text"])
                            for p in input_instance["contexts"]
                            if (p["title"], p["paragraph_text"]) not in title_paragraph_tuples
                        )
                    elif self.add_gold_paras:
                        title_paragraph_tuples.extend(
                            (p["title"], p["paragraph_text"])
                            for p in input_instance["contexts"]
                            if p["is_supporting"] and (p["title"], p["paragraph_text"]) not in title_paragraph_tuples
                        )

                    # Apply word limit if specified
                    if self.max_num_words_per_para is not None:
                        title_paragraph_tuples = [
                            (title, " ".join(text.split()[:self.max_num_words_per_para]))
                            for title, text in title_paragraph_tuples
                        ]

                    # Prepare output
                    output_instance.update({
                        "titles": [t[0] for t in title_paragraph_tuples],
                        "paras": [t[1] for t in title_paragraph_tuples],
                        "pids": [
                            get_pid_for_title_paragraph_text(t[0], t[1])
                            for t in title_paragraph_tuples
                        ],
                        "real_pids": [
                            p["id"] for p in input_instance["contexts"]
                            if p["is_supporting"] and "id" in p
                        ],
                        "metadata": {
                            "level": input_instance.get("level"),
                            "type": input_instance.get("type"),
                            "answer_type": input_instance.get("answer_type"),
                            "simplified_answer_type": input_instance.get("simplified_answer_type"),
                            "gold_titles": [p["title"] for p in input_instance["contexts"] if p["is_supporting"]],
                            "gold_paras": [p["paragraph_text"] for p in input_instance["contexts"] if p["is_supporting"]],
                            "gold_ids": [p.get("id") for p in input_instance["contexts"] if p["is_supporting"]],
                            "pin_position": self.pin_position
                        }
                    })

                    # Add pinned para metadata if enabled
                    if self.add_pinned_paras:
                        pinned = input_instance["pinned_contexts"][0]
                        output_instance["metadata"].update({
                            "pinned_para": pinned["paragraph_text"],
                            "pinned_title": pinned["title"]
                        })

                    # Add backup paras if they exist
                    if "paras" in output_instance:
                        output_instance.update({
                            "backup_paras": output_instance["paras"].copy(),
                            "backup_titles": output_instance["titles"].copy()
                        })

                    # Add valid titles if they exist
                    if "valid_titles" in input_instance:
                        output_instance["valid_titles"] = input_instance["valid_titles"]

                    yield output_instance

                except Exception as e:
                    print(f"Error processing line {line_num}: {str(e)}")
                    continue
