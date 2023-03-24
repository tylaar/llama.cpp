//
// Created by Yifeng Yu on 2023/3/24.
//

#ifndef LLAMA_CPP_LLAMA_H
#define LLAMA_CPP_LLAMA_H

// determine number of model parts based on the dimension
static const std::map<int, int> LLAMA_N_PARTS = {
        { 4096, 1 },
        { 5120, 2 },
        { 6656, 4 },
        { 8192, 8 },
};

// default hparams (LLaMA 7B)
struct llama_hparams {
    int32_t n_vocab = 32000;
    int32_t n_ctx   = 512;   // this is provided as user input?
    int32_t n_embd  = 4096;
    int32_t n_mult  = 256;
    int32_t n_head  = 32;
    int32_t n_layer = 32;
    int32_t n_rot   = 64;
    int32_t f16     = 1;
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
    llama_hparams hparams;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<llama_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};


#endif //LLAMA_CPP_LLAMA_H
