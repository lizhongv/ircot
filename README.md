## Installation

```bash
# 1. git
export https_proxy=http://agent.baidu.com:8891
git clone https://github.com/StonyBrookNLP/ircot.git
cd ircot

# 2. env
pip install uv 
uv venv ircot --python 3.10 && source ircot/bin/activate && uv pip install --upgrade pip
# deactivate 
# rm -rf ircot

# 3. install 
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt

python -m spacy download en_core_web_sm
```

## Prepare Data
```bash
# 1. download processed data 
# ./download/processed_data.sh
# processed_data/{dataset_name}/


# pip install modelscope
# modelscope login --token 3126ecd0-c40b-4aaf-9d54-0d68aec7e3e7
tar -czvf processed_data.tar.gz processed_data # 压缩
modelscope upload zl2272001/IRCoT_data ./processed_data.tar.gz processed_data.tar.gz --repo-type dataset # 上传
modelscope download --dataset zl2272001/IRCoT_data processed_data.tar.gz --local_dir ./ # 下载
tar -zxvf processed_data.tar.gz # 解压

# 2. download raw data
# ./download/raw_data.sh
# raw_data/{dataset_name}/


tar -czvf raw_data.tar.gz raw_data  # 压缩
modelscope upload zl2272001/IRCoT_data ./raw_data.tar.gz raw_data.tar.gz --repo-type dataset  # 上传
modelscope download --dataset zl2272001/IRCoT_data raw_data.tar.gz --local_dir ./ # 下载
tar -zxvf raw_data.tar.gz -C temp # 解压


# 3. download result
./download/official_eval.sh
tar -czvf official_evaluation.tar.gz official_evaluation
modelscope upload zl2272001/IRCoT_data ./official_evaluation.tar.gz official_evaluation.tar.gz --repo-type dataset


# 4. download predictions
./download/predictions.sh
tar -czvf predictions.tar.gz predictions
modelscope upload zl2272001/IRCoT_data ./predictions.tar.gz  predictions.tar.gz --repo-type dataset
```

## Elasticsearch  
```bash
# 1.download 
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz
tar -xzf elasticsearch-7.10.2-linux-x86_64.tar.gz  # 解压

# 2. install java
# cat /etc/os-release 
# java -version  

# 2.1 root用户安装jdk11
apt install openjdk-11-jdk # $JAVA_HOME

# 2.2 或者可以通过修改配置文件，直接使用自带的JDK，避免版本不匹配
vi elasticsearch-7.10.2/bin/elasticsearch-env
# https://www.cnblogs.com/zhuhuibiao/p/16446105.html

# 3. Add new user
groupadd elasticsearch
useradd -g elasticsearch elasticsearch
chown -R elasticsearch:elasticsearch elasticsearch-7.10.2
# cat /etc/passwd  

# Since the root directory cannot be authorized, move it to a non-root directory, such as the tmp directory.
ls -ld /tmp # check the permissions of the dir
mv ./elasticsearch-7.10.2 /tmp/

# 4. run elasticsearch
cd /tmp/elasticsearch-7.10.2
su elasticsearch  # switch users directly
su - elasticsearch  # require root permission to create "/home/elasticsearch"
whoami # elasticsearch

# ./bin/elasticsearch 
./bin/elasticsearch -d  # start the server
pkill -f elasticsearch # stop the server

# 5. check elash
curl http://localhost:9200 # base info
ps -ef | grep elasticsearch  # check process

# 6. run retriever
uvicorn serve:app --port 8000 --app-dir retriever_server

# 7. create index (downloaded raw_data and processed_data)
python retriever_server/build_index.py {dataset_name} 
# hotpotqa, iirc, 2wikimultihopqa, musique

# 8. check index
curl localhost:9200/_cat/indices  
# yellow open hotpotqa Z8EBBx6vQCCb5FMZhXk2Bg 1 1 5233329 0 2.1gb 2.1gb
```
## Instruction
```bash
./reproduce.sh oner codex hotpotqa

python runner.py $SYSTEM $MODEL $DATASET summarize --prompt_set 1
```

### hotpotqa
```bash
export SYSTEM=ircot
export MODEL=qwen2.5-7b
export DATASET=hotpotqa
python runner.py $SYSTEM $MODEL $DATASET write --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET predict --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET summarize --prompt_set 1

##### 1  write 
python runner.py ircot qwen2.5-7b hotpotqa write --prompt_set 1
// "ircot", "qwen2.5-7b", "hotpotqa", "write", "--prompt_set", "1"

python run.py write ircot_qwen2.5_7b_hotpotqa --instantiation_scheme ircot --prompt_set 1 --no_diff
// "write", "ircot_qwen2.5_7b_hotpotqa", "--instantiation_scheme", "ircot", "--prompt_set", "1", "--no_diff"

##### 2  predict 
python runner.py ircot qwen2.5-7b hotpotqa predict --prompt_set 1
// "ircot", "qwen2.5-7b", "hotpotqa", "predict", "--prompt_set", "1"

 python run.py predict ircot_qwen2.5_7b_hotpotqa --instantiation_scheme ircot --prompt_set 1 --evaluation_path processed_data/hotpotqa/dev_subsampled.jsonl --skip_if_exists --silent
// "predict", "ircot_qwen2.5_7b_hotpotqa", "--instantiation_scheme", "ircot", "--prompt_set", "1",  "--evaluation_path", "processed_data/hotpotqa/dev_subsampled.jsonl", "--skip_if_exists", "--silent"

python predict.py instantiated_configs/ircot_qwen2.5_7b_hotpotqa____prompt_set_1___bm25_retrieval_count__2___distractor_count__1.jsonnet processed_data/hotpotqa/dev_subsampled.jsonl --silent
// "instantiated_configs/ircot_qwen2.5_7b_hotpotqa____prompt_set_1___bm25_retrieval_count__2___distractor_count__1.jsonnet", "processed_data/hotpotqa/dev_subsampled.jsonl", "--silent"

RETRIEVER_HOST=http://localhost RETRIEVER_PORT=8000 LLM_SERVER_HOST=http://localhost LLM_SERVER_PORT=8010 python -m commaqa.inference.configurable_inference --config instantiated_configs/ircot_qwen2.5_7b_hotpotqa____prompt_set_1___bm25_retrieval_count__2___distractor_count__1.jsonnet --input processed_data/hotpotqa/dev_subsampled.jsonl --output new_predictions/ircot_qwen2.5_7b_hotpotqa____prompt_set_1___bm25_retrieval_count__2___distractor_count__1/prediction__hotpotqa_to_hotpotqa__dev_subsampled.json --silent
//  "--config", "instantiated_configs/ircot_qwen2.5_7b_hotpotqa____prompt_set_1___bm25_retrieval_count__2___distractor_count__1.jsonnet","--input","processed_data/hotpotqa/dev_subsampled.jsonl", "--output", "new_predictions/ircot_qwen2.5_7b_hotpotqa____prompt_set_1___bm25_retrieval_count__2___distractor_count__1/prediction__hotpotqa_to_hotpotqa__dev_subsampled.json","--silent"


```
 
 
## oner

### hotpotqa
```bash
export SYSTEM=oner 
export MODEL=none
export DATASET=hotpotqa
python runner.py $SYSTEM $MODEL $DATASET write --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET predict --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET summarize --prompt_set 1

python runner.py $SYSTEM $MODEL $DATASET write --prompt_set 1 --best
python runner.py $SYSTEM $MODEL $DATASET predict --prompt_set 1 --best --eval_test --official
python runner.py $SYSTEM $MODEL $DATASET summarize --prompt_set 1 --best --eval_test --official

# 1. write
python runner.py oner none hotpotqa write --prompt_set 1
# "oner", "none",  "hotpotqa",  "write", "--prompt_set", "1"
python run.py write oner_hotpotqa --instantiation_scheme oner --prompt_set 1 --no_diff
# "write", "oner_hotpotqa", "--instantiation_scheme", "oner", "--prompt_set", "1", "--no_diff"

# 2. predict
python runner.py oner none hotpotqa  predict --prompt_set 1
# "oner", "none",  "hotpotqa",  "predict", "--prompt_set", "1"

python run.py predict oner_hotpotqa --instantiation_scheme oner --prompt_set 1 --evaluation_path processed_data/hotpotqa/dev_subsampled.jsonl --skip_if_exists --silent
# "predict", "oner_hotpotqa", "--instantiation_scheme", "oner", "--prompt_set", "1", "--evaluation_path", "processed_data/hotpotqa/dev_subsampled.jsonl","--skip_if_exists", "--silent"

python predict.py instantiated_configs/oner_hotpotqa____prompt_set_1___bm25_retrieval_count__15.jsonnet processed_data/hotpotqa/dev_subsampled.jsonl --silent
# "instantiated_configs/oner_hotpotqa____prompt_set_1___bm25_retrieval_count__15.jsonnet", "processed_data/hotpotqa/dev_subsampled.jsonl", "--silent"

RETRIEVER_HOST=http://localhost RETRIEVER_PORT=8000 LLM_SERVER_HOST=http://localhost LLM_SERVER_PORT=8010 python -m commaqa.inference.configurable_inference --config instantiated_configs/oner_hotpotqa____prompt_set_1___bm25_retrieval_count__15.jsonnet --input processed_data/hotpotqa/dev_subsampled.jsonl --output new_predictions/oner_hotpotqa____prompt_set_1___bm25_retrieval_count__15/prediction__hotpotqa_to_hotpotqa__dev_subsampled.json --silent
# "--config", "instantiated_configs/oner_hotpotqa____prompt_set_1___bm25_retrieval_count__15.jsonnet", "--input", "processed_data/hotpotqa/dev_subsampled.jsonl", "--output", "new_predictions/oner_hotpotqa____prompt_set_1___bm25_retrieval_count__15/prediction__hotpotqa_to_hotpotqa__dev_subsampled.json", "--silent"

python evaluate.py instantiated_configs/oner_hotpotqa____prompt_set_1___bm25_retrieval_count__15.jsonnet processed_data/hotpotqa/dev_subsampled.jsonl
# "instantiated_configs/oner_hotpotqa____prompt_set_1___bm25_retrieval_count__15.jsonnet", "processed_data/hotpotqa/dev_subsampled.jsonl"

# 3. summary
python runner.py oner none hotpotqa summarize --prompt_set 1
```

### 2wikimultihopqa
```bash
export SYSTEM=oner 
export MODEL=none
export DATASET=2wikimultihopqa
python runner.py $SYSTEM $MODEL $DATASET write --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET predict --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET summarize --prompt_set 1

python runner.py $SYSTEM $MODEL $DATASET write --prompt_set 1 --best
python runner.py $SYSTEM $MODEL $DATASET predict --prompt_set 1 --best --eval_test --official
python runner.py $SYSTEM $MODEL $DATASET summarize --prompt_set 1 --best --eval_test --official
```
### musique
```bash
export SYSTEM=oner 
export MODEL=none
export DATASET=musique
python runner.py $SYSTEM $MODEL $DATASET write --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET predict --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET summarize --prompt_set 1

python runner.py $SYSTEM $MODEL $DATASET write --prompt_set 1 --best
python runner.py $SYSTEM $MODEL $DATASET predict --prompt_set 1 --best --eval_test --official
python runner.py $SYSTEM $MODEL $DATASET summarize --prompt_set 1 --best --eval_test --official
```

### iirc
```bash
export SYSTEM=oner 
export MODEL=none
export DATASET=iirc
python runner.py $SYSTEM $MODEL $DATASET write --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET predict --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET summarize --prompt_set 1

python runner.py $SYSTEM $MODEL $DATASET write --prompt_set 1 --best
python runner.py $SYSTEM $MODEL $DATASET predict --prompt_set 1 --best --eval_test --official
python runner.py $SYSTEM $MODEL $DATASET summarize --prompt_set 1 --best --eval_test --official


##### 1 write 
python runner.py oner none iirc write --prompt_set 1
# "oner","none","iirc","write","--prompt_set","1"

python run.py write oner_iirc --instantiation_scheme oner --prompt_set 1 --no_diff
# "write","oner_iirc","--instantiation_scheme","oner","--prompt_set","1","--no_diff"

##### 2 predict 
python runner.py oner none iirc predict --prompt_set 1
# "oner","none","iirc","predict","--prompt_set","1"

python run.py predict oner_iirc --instantiation_scheme oner --prompt_set 1 --evaluation_path processed_data/iirc/dev_subsampled.jsonl --skip_if_exists --silent
# "predict","oner_iirc","--instantiation_scheme","oner","--prompt_set","1","--evaluation_path","processed_data/iirc/dev_subsampled.jsonl","--skip_if_exists","--silent"

python predict.py instantiated_configs/oner_iirc____prompt_set_1___bm25_retrieval_count__15.jsonnet processed_data/iirc/dev_subsampled.jsonl --silent
# "instantiated_configs/oner_iirc____prompt_set_1___bm25_retrieval_count__15.jsonnet","processed_data/iirc/dev_subsampled.jsonl","--silent"

RETRIEVER_HOST=http://localhost RETRIEVER_PORT=8000 LLM_SERVER_HOST=http://localhost LLM_SERVER_PORT=8010 python -m commaqa.inference.configurable_inference --config instantiated_configs/oner_iirc____prompt_set_1___bm25_retrieval_count__15.jsonnet --input processed_data/iirc/dev_subsampled.jsonl --output new_predictions/oner_iirc____prompt_set_1___bm25_retrieval_count__15/prediction__iirc_to_iirc__dev_subsampled.json --silent
# "--config", "instantiated_configs/oner_iirc____prompt_set_1___bm25_retrieval_count__15.jsonnet", "--input", "processed_data/iirc/dev_subsampled.jsonl", "--output", "new_predictions/oner_iirc____prompt_set_1___bm25_retrieval_count__15/prediction__iirc_to_iirc__dev_subsampled.json", "--silent"

python evaluate.py instantiated_configs/oner_iirc____prompt_set_1___bm25_retrieval_count__15.jsonnet processed_data/iirc/dev_subsampled.jsonl
# "instantiated_configs/oner_iirc____prompt_set_1___bm25_retrieval_count__15.jsonnet", "processed_data/iirc/dev_subsampled.jsonl"

##### 5 summary
python runner.py oner none iirc summarize --prompt_set 1
```
