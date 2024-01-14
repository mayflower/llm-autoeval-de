#!/bin/bash

start=$(date +%s)

# Install dependencies
apt update
apt install -y screen vim git-lfs
screen

# Install common libraries
pip install -q git+https://github.com/huggingface/transformers.git git+https://github.com/casper-hansen/AutoAWQ.git  bitsandbytes requests accelerate sentencepiece pytablewriter einops protobuf 

if [ "$DEBUG" == "True" ]; then
    echo "Launch LLM AutoEval in debug mode"
fi

# Run evaluation
if [ "$BENCHMARK" == "openllm" ]; then
    git clone -b mmlu_de https://github.com/bjoernpl/lm-evaluation-harness-de.git
    cd lm-evaluation-harness-de
    git pull origin pull/3/head
    pip install -e ".[multilingual, sentencepiece]"

    pip install langdetect immutabledict

    benchmark="arc_de"
    python eval_de.py  --model hf-causal-experimental \
        --model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.8,use_accelerate=True,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks arc_challenge_de \
        --num_fewshot 25 \
        --batch_size auto \
        --output_path ./${benchmark}.json

    benchmark="hellaswag_de"
    python eval_de.py  --model hf-causal-experimental \
        --model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.8,use_accelerate=True,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks hellaswag_de \
        --num_fewshot 10 \
        --batch_size auto \
        --output_path ./${benchmark}.json

    benchmark="mmlu_de"
    python eval_de.py  --model hf-causal-experimental \
         --model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.8,use_accelerate=True,trust_remote_code=$TRUST_REMOTE_CODE \
         --tasks "MMLU-DE*" \
         --num_fewshot 5 \
         --batch_size auto \
         --verbosity DEBUG \
         --output_path ./${benchmark}.json
    
    benchmark="truthfulqa_de"
    python eval_de.py  --model hf-causal-experimental \
        --model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.8,use_accelerate=True,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks truthful_qa_de \
        --num_fewshot 0 \
        --batch_size auto \
        --output_path ./${benchmark}.json
    

    end=$(date +%s)
    echo "Elapsed Time: $(($end-$start)) seconds"
    
    python ../llm-autoeval/main.py . $(($end-$start))
else
    echo "Error: Invalid BENCHMARK value. Please set BENCHMARK to 'openllm'."
fi

if [ "$DEBUG" == "False" ]; then
    runpodctl remove pod $RUNPOD_POD_ID
fi
sleep infinity
