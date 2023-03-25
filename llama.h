//
// Created by Yifeng Yu on 2023/3/24.
//

#ifndef LLAMA_CPP_LLAMA_H
#define LLAMA_CPP_LLAMA_H

#include "ggml.h"

#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include <fmt/core.h>

#define SPLIT_TYPE_COLUMN 0
#define SPLIT_TYPE_ROW    1

// determine number of model parts based on the dimension
static const std::map<int, int> LLAMA_N_PARTS = {
        { 4096, 1 },
        { 5120, 2 },
        { 6656, 4 },
        { 8192, 8 },
};

// default hparams (LLaMA 7B)
struct llama_hyper_params {
    int32_t n_vocab = 32000;
    int32_t n_ctx   = 512;   // this is provided as user input?
    int32_t n_embd  = 4096;
    int32_t n_mult  = 256;
    int32_t n_head  = 32;
    int32_t n_layer = 32;
    int32_t n_rot   = 64;
    int32_t f16     = 1;
};

// default loading parameters
class llama_load_ctx {
public:
    int n_ff;
    int n_parts;
    int split_type;
    ggml_type wtype;

    size_t determine_bpe(int32_t ftype, int ne[]);

    bool is_column_split_type() {
        return split_type == SPLIT_TYPE_COLUMN;
    }
};

class llama_layer {
public:
    // normalization
    struct ggml_tensor * attention_norm;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;

    // normalization
    struct ggml_tensor * ffn_norm;

    // ff
    struct ggml_tensor * w1;
    struct ggml_tensor * w2;
    struct ggml_tensor * w3;

    std::string ggml_tensor_shape(const ggml_tensor *t) const {
        return fmt::format("{}, {}, {}, {}", t->ne[0],
                           t->ne[1],
                           t->ne[2],
                           t->ne[3]);
    }

    void llama_layer_debug() const {
        std::cout << "layer tensor num attetion : " << ggml_tensor_shape(this->attention_norm)
                  << " Q:" << ggml_tensor_shape(this->wq)
                  << " K: " << ggml_tensor_shape(this->wk)
                  << " V:" << ggml_tensor_shape(this->wv)
                  << std::endl;
    }

};


class llama_model {
public:
    llama_hyper_params hparams;
    llama_load_ctx load_ctx;

    // First go through tok embedding
    struct ggml_tensor * tok_embeddings;

    // norm the input
    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<llama_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;

    // load the model's weights from a file
    bool load_model(const std::string & fname, gpt_vocab & vocab, int n_ctx);
    bool eval_model(const int n_threads, const int n_past, const std::vector<gpt_vocab::id> &embd_inp, std::vector<float>         & embd_w,
                    size_t                     & mem_per_token);
private:
    // checking if file magic number matched.
    bool verify_model_magic(std::ifstream& fin);
    // verify tensor shape and dimension.
    bool verify_tensor_shape_and_dim(ggml_tensor* tensor, std::string& name, int n_parts, int n_dims, int nelements, int ne[]);

    bool verify_tensor_one_dimension(ggml_tensor* tensor, std::string& name, int nelements, int ne[]);
    // verify tensor shape in column mode
    bool verify_tensor_shape_by_column(ggml_tensor *tensor, std::string& name, int n_parts, int nelements, int ne[]);
    // verify tensor shape in row mode
    bool verify_tensor_shape_by_row(ggml_tensor *tensor, std::string& name, int n_parts, int nelements, int ne[]);
    // load hparams for model metadata purpose.
    void load_model_hyper_params(std::ifstream &fin, int n_ctx);
    // load model's vocab
    void load_model_vocab(std::ifstream &fin, gpt_vocab& vocab);
    int load_model_tensor(std::ifstream &fin, int part_id);
    // build model ctx unit according to data type.
    bool build_model_ctx();
    // determine ggml type based on hyperparams.
    void determine_ggml_wtype();
    void determine_ggml_file_split(std::string& name);
};


#endif //LLAMA_CPP_LLAMA_H
