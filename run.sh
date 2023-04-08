#!/bin/bash
if ! [[ "$1" =~ ^[0-9]{1,2}B$ ]]; then
    echo
    echo "Usage: run.sh 7B|13B|30B|65B"
    echo
    exit 1
fi


./llama -m ~/hack/models/llama/$1/ggml-model-q4_0.bin -p $2 -n 512 -n 512
