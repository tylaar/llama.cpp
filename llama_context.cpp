//
// Created by Yifeng Yu on 2023/4/5.
//

#include "llama_context.h"

void llama_context::sample_top_k(std::vector<std::pair<float, llama_vocab::id>> & logits_id, int top_k) {
    // find the top k tokens
    std::partial_sort(
            logits_id.begin(),
            logits_id.begin() + top_k, logits_id.end(),
            [](const std::pair<float, llama_vocab::id> & a, const std::pair<float, llama_vocab::id> & b) {
                return a.first > b.first;
            });

    logits_id.resize(top_k);
}

llama_vocab::id llama_context::llama_sample_top_p_top_k(
        const std::vector<llama_vocab::id> & last_n_tokens,
        int top_k,
        float top_p,
        float temp,
        float repeat_penalty) {

    const int n_logits = model.hparams.n_vocab;
    const auto * plogits = logits.data() + logits.size() - n_logits;

    std::vector<std::pair<float, llama_vocab::id>> logits_id;
    logits_id.reserve(n_logits);

    {
        const float scale = 1.0f/temp;
        for (int i = 0; i < n_logits; ++i) {
            // repetition penalty from ctrl paper (https://arxiv.org/abs/1909.05858)
            // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
            if (std::find(last_n_tokens.begin(), last_n_tokens.end(), i) != last_n_tokens.end()) {
                // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if (plogits[i] < 0.0f) {
                    logits_id.push_back(std::make_pair(plogits[i]*scale*repeat_penalty, i));
                } else {
                    logits_id.push_back(std::make_pair(plogits[i]*scale/repeat_penalty, i));
                }
            } else {
                logits_id.push_back(std::make_pair(plogits[i]*scale, i));
            }
        }
    }

    sample_top_k(logits_id, top_k);

    float maxl = -std::numeric_limits<float>::infinity();
    for (const auto & kv : logits_id) {
        maxl = Max(maxl, kv.first);
    }

    // compute probs for the top k tokens
    std::vector<float> probs;
    probs.reserve(logits_id.size());

    double sum = 0.0;
    for (const auto & kv : logits_id) {
        const float p = expf(kv.first - maxl);
        probs.push_back(p);
        sum += p;
    }

    // normalize the probs
    for (auto & p : probs) {
        p /= sum;
    }

    if (top_p < 1.0) {
        double cumsum = 0.0;
        for (int i = 0; i < (int) probs.size(); i++) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                probs.resize(i + 1);
                logits_id.resize(i + 1);
                break;
            }
        }

        cumsum = 1.0/cumsum;
        for (int i = 0; i < (int) probs.size(); i++) {
            probs[i] *= cumsum;
        }
    }

    //printf("\n");
    //for (int i = 0; i < (int) 10; i++) {
    //    printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
    //}
    //printf("\n\n");
    //exit(0);

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    return logits_id[idx].second;
}

void llama_context::llama_print_timings() {
    const int64_t t_end_us = ggml_time_us();

    const int32_t n_sample = Max(1, n_sample);
    const int32_t n_eval   = Max(1, n_eval);
    const int32_t n_p_eval = Max(1, n_p_eval);

    fprintf(stderr, "\n");
    fprintf(stderr, "%s:        load time = %8.2f ms\n", __func__, t_load_us / 1000.0);
    fprintf(stderr, "%s:      sample time = %8.2f ms / %5d runs   (%8.2f ms per run)\n",   __func__, 1e-3 * t_sample_us, n_sample, 1e-3 * t_sample_us / n_sample);
    fprintf(stderr, "%s: prompt eval time = %8.2f ms / %5d tokens (%8.2f ms per token)\n", __func__, 1e-3 * t_p_eval_us, n_p_eval, 1e-3 * t_p_eval_us / n_p_eval);
    fprintf(stderr, "%s:        eval time = %8.2f ms / %5d runs   (%8.2f ms per run)\n",   __func__, 1e-3 * t_eval_us,   n_eval,   1e-3 * t_eval_us   / n_eval);
    fprintf(stderr, "%s:       total time = %8.2f ms\n", __func__, (t_end_us - t_start_us)/1000.0);
}

void llama_context::use_buf(ggml_context* ctx, int i) {
#if defined(LLAMA_USE_SCRATCH)
    size_t last_size = 0;

    if (i == -1) {
        last_size = ggml_set_scratch(ctx, { 0, 0, nullptr, });
    } else {
        auto & buf = buf_scratch[i];
        last_size = ggml_set_scratch(ctx, { 0, buf.size(), buf.data(), });
    }

    if (buf_last >= 0) {
        buf_max_size[buf_last] = Max(buf_max_size[buf_last], last_size);
    }

    buf_last = i;
#else
    (void) i;
        (void) ctx;
#endif
}

size_t llama_context::get_buf_max_mem(int i) const {
#if defined(LLAMA_USE_SCRATCH)
    return buf_max_size[i];
#else
    (void) i;
        return 0;
#endif

}

int llama_context::llama_n_vocab() {
    return vocab.id_to_token.size();
}

int llama_context::llama_n_ctx() {
    return model.hparams.n_ctx;
}

int llama_context::llama_n_embd() {
    return model.hparams.n_embd;
}

float * llama_context::llama_get_logits() {
    return logits.data();
}

float * llama_context::llama_get_embeddings() {
    return embedding.data();
}

const char * llama_context::llama_token_to_str(llama_token token) {
    if (token >= llama_n_vocab()) {
        return nullptr;
    }

    return vocab.id_to_token[token].tok.c_str();
}

llama_token llama_context::llama_token_bos() {
    return 1;
}

llama_token llama_context::llama_token_eos() {
    return 2;
}
