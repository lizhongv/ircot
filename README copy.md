# Installation
 安装依赖
 pip install pipreqs
 pipreqs . --force --ignore=ircot 
 

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
# pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

python -m spacy download en_core_web_sm
# pip install spacy==3.5.3 pydantic==1.10.2
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


部署生成器
```bash

cd ircot 
export MODEL_NAME="qwen2.5-7B"
nohup  uvicorn serve:app --host 0.0.0.0 --port 8010 --app-dir llm_server > llm_server.log 2>&1 &
tail -f llm_server.log  

  
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
```

部署检索器
```bash
cd ircot
nohup uvicorn serve:app --port 8000 --app-dir retriever_server > retriever_server.log 2>&1 &

curl "http://localhost:8000/"

curl -X POST  http://localhost:8000/retrieve \
    -H "Content-Type: application/json" \
    -d '{"retrieval_method": "retrieve_from_elasticsearch", "query_text": "Given that a certain scientist won the Nobel Prize in Physics for a discovery related to sub - atomic particles, and this scientist studied at a university in England, and the university is known for its strong research in quantum mechanics, which scientist is it?", "max_hits_count": 3, "max_buffer_count": 100, "document_type": "paragraph_text"}' 
```


## Instruction
```bash
./reproduce.sh oner codex hotpotqa

python runner.py $SYSTEM $MODEL $DATASET summarize --prompt_set 1
```

### hotpotqa
```bash
export SYSTEM=oner_qa
export MODEL=qwen2.5-7b
export DATASET=hotpotqa
python runner.py $SYSTEM $MODEL $DATASET write --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET predict --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET evaluate --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET summarize --prompt_set 1

export SYSTEM=nor_qa
export MODEL=qwen2.5-7b
export DATASET=hotpotqa
python runner.py $SYSTEM $MODEL $DATASET write --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET predict --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET evaluate --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET summarize --prompt_set 1

python runner.py $SYSTEM $MODEL $DATASET write --prompt_set 1 --best
python runner.py $SYSTEM $MODEL $DATASET predict --prompt_set 1 --best --eval_test --official
python runner.py $SYSTEM $MODEL $DATASET summarize --prompt_set 1 --best --eval_test --official

export SYSTEM=ircot_qa
export MODEL=qwen2.5-7b
export DATASET=hotpotqa
python runner.py $SYSTEM $MODEL $DATASET write --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET predict --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET evaluate --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET summarize --prompt_set 1

export SYSTEM=oner
export MODEL=qwen2.5-7b
export DATASET=hotpotqa
python runner.py $SYSTEM $MODEL $DATASET write --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET predict --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET evaluate --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET summarize --prompt_set 1

export SYSTEM=ircot
export MODEL=qwen2.5-7b
export DATASET=hotpotqa
python runner.py $SYSTEM $MODEL $DATASET write --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET predict --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET evaluate --prompt_set 1
python runner.py $SYSTEM $MODEL $DATASET summarize --prompt_set 1
```
