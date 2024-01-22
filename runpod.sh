#!/bin/bash

start=$(date +%s)

# Install dependencies
apt update
apt install -y screen vim git-lfs
screen

# Install common libraries
pip install -q git+https://github.com/huggingface/transformers.git git+https://github.com/casper-hansen/AutoAWQ.git  bitsandbytes requests accelerate sentencepiece pytablewriter einops protobuf flash-attn 

if [ "$DEBUG" == "True" ]; then
    echo "Launch LLM AutoEval in debug mode"
fi

# Run evaluation
if [ "$BENCHMARK" == "nous" ]; then
    git clone -b add-agieval https://github.com/dmahan93/lm-evaluation-harness
    cd lm-evaluation-harness
    pip install -e .

    benchmark="agieval"
    python main.py \
        --model hf-causal \
        --model_args pretrained=$MODEL,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks agieval_aqua_rat,agieval_logiqa_en,agieval_lsat_ar,agieval_lsat_lr,agieval_lsat_rc,agieval_sat_en,agieval_sat_en_without_passage,agieval_sat_math \
        --device cuda:0 \
        --batch_size auto \
        --output_path ./${benchmark}.json

    benchmark="gpt4all"
    python main.py \
        --model hf-causal \
        --model_args pretrained=$MODEL,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks hellaswag,openbookqa,winogrande,arc_easy,arc_challenge,boolq,piqa \
        --device cuda:0 \
        --batch_size auto \
        --output_path ./${benchmark}.json

    benchmark="truthfulqa"
    python main.py \
        --model hf-causal \
        --model_args pretrained=$MODEL,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks truthfulqa_mc \
        --device cuda:0 \
        --batch_size auto \
        --output_path ./${benchmark}.json

    benchmark="bigbench"
    python main.py \
        --model hf-causal \
        --model_args pretrained=$MODEL,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks bigbench_causal_judgement,bigbench_date_understanding,bigbench_disambiguation_qa,bigbench_geometric_shapes,bigbench_logical_deduction_five_objects,bigbench_logical_deduction_seven_objects,bigbench_logical_deduction_three_objects,bigbench_movie_recommendation,bigbench_navigate,bigbench_reasoning_about_colored_objects,bigbench_ruin_names,bigbench_salient_translation_error_detection,bigbench_snarks,bigbench_sports_understanding,bigbench_temporal_sequences,bigbench_tracking_shuffled_objects_five_objects,bigbench_tracking_shuffled_objects_seven_objects,bigbench_tracking_shuffled_objects_three_objects \
        --device cuda:0 \
        --batch_size auto \
        --output_path ./${benchmark}.json

    end=$(date +%s)
    echo "Elapsed Time: $(($end-$start)) seconds"
    
    python ../llm-autoeval/main.py . $(($end-$start))

elif [ "$BENCHMARK" == "openllm" ]; then
    git clone https://github.com/EleutherAI/lm-evaluation-harness
    cd lm-evaluation-harness
    pip install -e ".[vllm,promptsource]"
    pip install langdetect immutabledict

    benchmark="arc"
    lm_eval --model vllm \
        --model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks arc_challenge \
        --num_fewshot 25 \
        --batch_size auto \
        --output_path ./${benchmark}.json

    benchmark="hellaswag"
    lm_eval --model vllm \
        --model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks hellaswag \
        --num_fewshot 10 \
        --batch_size auto \
        --output_path ./${benchmark}.json

    # benchmark="mmlu"
    # lm_eval --model vllm \
    #     --model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=$TRUST_REMOTE_CODE \
    #     --tasks mmlu \
    #     --num_fewshot 5 \
    #     --batch_size auto \
    #     --verbosity DEBUG \
    #     --output_path ./${benchmark}.json
    
    benchmark="truthfulqa"
    lm_eval --model vllm \
        --model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks truthfulqa \
        --num_fewshot 0 \
        --batch_size auto \
        --output_path ./${benchmark}.json
    
    benchmark="winogrande"
    lm_eval --model vllm \
        --model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks winogrande \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path ./${benchmark}.json
    
    benchmark="gsm8k"
    lm_eval --model vllm \
        --model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path ./${benchmark}.json

    end=$(date +%s)
    echo "Elapsed Time: $(($end-$start)) seconds"
    
    python ../llm-autoeval/main.py . $(($end-$start))
    
elif [ "$BENCHMARK" == "openllm_de" ]; then
    git clone -b mmlu_de https://github.com/mayflower/lm-evaluation-harness-de.git
    cd lm-evaluation-harness-de
    pip install -e ".[multilingual, sentencepiece]"
    pip install langdetect immutabledict

    benchmark="arc_de"
    python eval_de.py  --model hf-causal-experimental \
        --model_args pretrained=${MODEL},dtype=auto,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks arc_challenge_de \
        --num_fewshot 25 \
        --batch_size auto \
        --output_path ./${benchmark}.json

    benchmark="hellaswag_de"
    python eval_de.py  --model hf-causal-experimental \
        --model_args pretrained=${MODEL},dtype=auto,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks hellaswag_de \
        --num_fewshot 10 \
        --batch_size auto \
        --output_path ./${benchmark}.json

    benchmark="mmlu_de"
    python eval_de.py  --model hf-causal-experimental \
         --model_args pretrained=${MODEL},dtype=auto,trust_remote_code=$TRUST_REMOTE_CODE \
         --tasks "MMLU-DE*" \
         --num_fewshot 5 \
         --batch_size auto \
         --output_path ./${benchmark}.json
    
    benchmark="truthfulqa_de"
    python eval_de.py  --model hf-causal-experimental \
        --model_args pretrained=${MODEL},dtype=auto,trust_remote_code=$TRUST_REMOTE_CODE \
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
