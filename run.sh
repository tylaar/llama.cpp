#!/bin/bash
if ! [[ "$1" =~ ^[0-9]{1,2}B$ ]]; then
    echo
    echo "Usage: run.sh 7B|13B|30B|65B"
    echo
    exit 1
fi

model=$1
shift
query=$*

./llama -m ~/hack/models/llama/$model/ggml-model-q4_0.bin -p "$query" -n 512 -n 512
