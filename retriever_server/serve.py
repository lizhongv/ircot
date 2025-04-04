import os
import sys
import json
from time import perf_counter
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from unified_retriever import UnifiedRetriever

retriever = UnifiedRetriever(host="http://localhost/", port=9200)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if True:
    from logging_.logging_config import logger, LYELLOW, RESET


app = FastAPI()


@app.get("/")
async def index():
    return {"message": "Hello! This is a retriever server."}


@app.post("/retrieve")
async def retrieve(arguments: Request):  # see the corresponding method in unified_retriever.py
    arguments = await arguments.json()
    retrieval_method = arguments.pop("retrieval_method")
    assert retrieval_method in ("retrieve_from_elasticsearch")
    start_time = perf_counter()
    retrieval = getattr(retriever, retrieval_method)(**arguments)
    end_time = perf_counter()
    time_in_seconds = round(end_time - start_time, 1)

    logger.debug(f"Retrieve texts: {LYELLOW}{json.dumps(retrieval, indent=4)}{RESET}")

    response_data = {"retrieval": retrieval, "time_in_seconds": time_in_seconds}
    return JSONResponse(content=response_data)  # JSONResponse source code: indent=4

    """
    curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"retrieval_method": "retrieve_from_elasticsearch", "query_text": "Given that a certain scientist won the Nobel Prize in Physics for a discovery related to sub - atomic particles, and this scientist studied at a university in England, and the university is known for its strong research in quantum mechanics, which scientist is it?", "max_hits_count": 3, "max_buffer_count": 100, "document_type": "paragraph_text"}' \
     http://localhost:8000/retrieve/
    """
