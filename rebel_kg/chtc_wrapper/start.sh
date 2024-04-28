export MODEL_NAME=dsarda/rebel_macrostrat_finetuned
# export REDIS_HOST=cosmos0001.chtc.wisc.edu
# conda run --no-capture-output -n unsupervised_kg python arq_worker.py
conda run --no-capture-output -n unsupervised_kg arq arq_worker.WorkerSettings