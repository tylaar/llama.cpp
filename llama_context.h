//
// Created by Yifeng Yu on 2023/4/5.
//

#ifndef LLAMA_CPP_LLAMA_CONTEXT_H
#define LLAMA_CPP_LLAMA_CONTEXT_H

#include "llama.h"

class llama_context {
public:
    std::mt19937 rng;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;
    bool has_evaluated_once = false;

    int64_t t_sample_us = 0;
    int64_t t_eval_us   = 0;
    int64_t t_p_eval_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_eval   = 0; // number of eval calls
    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)

    llama_model model;
    llama_vocab vocab;

    size_t mem_per_token = 0;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;
    bool logits_all = false;

    // input embedding (1-dimensional array: [n_embd])
    std::vector<float> embedding;

    // memory buffers used to evaluate the model
    // TODO: move in llama_state
    std::vector<uint8_t> buf_compute;
    std::vector<uint8_t> buf_scratch[LLAMA_MAX_SCRATCH_BUFFERS];

    int    buf_last = 0;
    size_t buf_max_size[LLAMA_MAX_SCRATCH_BUFFERS] = { 0 };

    static void sample_top_k(std::vector<std::pair<float, llama_vocab::id>> & logits_id, int top_k);

    llama_vocab::id llama_sample_top_p_top_k(
            const std::vector<llama_vocab::id> & last_n_tokens,
            int top_k,
            float top_p,
            float temp,
            float repeat_penalty);
    void use_buf(struct ggml_context * ctx, int i);
    size_t get_buf_max_mem(int i) const;

    int llama_n_vocab();
    int llama_n_ctx();
    int llama_n_embd();
    float * llama_get_logits();
    float * llama_get_embeddings();

    const char * llama_token_to_str(llama_token token);

    llama_token llama_token_bos();
    llama_token llama_token_eos();

    void llama_print_timings();
};


#endif //LLAMA_CPP_LLAMA_CONTEXT_H
