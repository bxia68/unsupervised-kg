import os
import asyncio
from httpx import AsyncClient
import json
from typing import List
from arq.connections import RedisSettings
from weaviate_wrapper import WeaviateWrapper, WeaviateText
from kg_runner import run_for_paragraph, get_model

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_SETTINGS = RedisSettings(
    host=REDIS_HOST,     
    port=REDIS_PORT,      
    password=REDIS_PASSWORD,  
)

# RESULT_ENDPOINT = os.getenv("REDIS_HOST", "http://cosmos0003.chtc.wisc.edu:9543/record_run")
RESULT_ENDPOINT = os.getenv("RESULT_ENDPOINT", "http://cosmos0001.chtc.wisc.edu:8060/print_json")

RUN_ID = os.getenv("RUN_ID", "rebel_2024-04-15_13:20:43.302554")
PIPELINE_ID = os.getenv("PIPELINE_ID", "0")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_TYPE = "rebel"
VERSION = 1
MODEL_ID = f"{MODEL_NAME}_v{VERSION}"

async def startup(ctx: dict):
    # initialize Weaviate and httpx sessions
    ctx["weaviate"] = WeaviateWrapper("http://cosmos0001.chtc.wisc.edu:8080", os.getenv("WEAVIATE_API_KEY"))
    ctx["httpx_client"] = AsyncClient()
    ctx["model"] = get_model(MODEL_TYPE, MODEL_NAME)
    print("WORKER: Ready to accept jobs.")
    
async def shutdown(ctx: dict):
    await ctx["httpx_client"].aclose()

async def store_results(ctx: dict, serialized_results: List[dict]):
    # post to Macrostrat endpoint if any triplets have been extracted in this batch
    if serialized_results:
        await ctx["httpx_client"].post(
            RESULT_ENDPOINT, 
            content=json.dumps({
                "run_id" : RUN_ID,
                "extraction_pipeline_id" : PIPELINE_ID,
                "model_id": MODEL_ID,
                "results": serialized_results
            })
        )

async def process_paragraphs(ctx: dict, paragraph_batch: List[str]):
    # pull paragraph text from Weaviate and extract triplets using the LLM
    output_list = []
    for paragraph_data in ctx["weaviate"].get_paragraphs_for_ids(paragraph_batch):
        output_kg = run_for_paragraph(paragraph_data.paragraph, paragraph_data.weaviate_id, ctx["model"])
        
        if output_kg and output_kg.relations:
            serialized_kg = {
                "text": {
                    "preprocessor_id": paragraph_data.preprocessor_id,
                    "paper_id": paragraph_data.paper_id,
                    "hashed_text": paragraph_data.preprocessor_id,
                    "weaviate_id": paragraph_data.preprocessor_id,
                    "paragraph_text": paragraph_data.paragraph
                },
                "relationships": []
            }

            for relationship in output_kg.get_json_representation():
                serialized_kg["relationships"].append({
                    "src": relationship["head"],
                    "relationship_type": relationship["type_key"],
                    "dst": relationship["tail"]
                })
                
            output_list.append(serialized_kg)
    
    # store results in Macrostrat endpoint
    await store_results(ctx, output_list)

class WorkerSettings:
    redis_settings = REDIS_SETTINGS
    functions = [process_paragraphs]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 10

async def main():
    # demo run of worker on 2 paragraphs
    ctx = {}
    await startup(ctx)
    await process_paragraphs(ctx, ["00000085-2145-4b37-b963-8c80d21b6964", "955616ce-f846-4665-9c64-d4709a34680d"])
    await shutdown(ctx)

if __name__ == "__main__":
    asyncio.run(main())