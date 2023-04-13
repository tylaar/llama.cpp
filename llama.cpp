//
// Created by Yifeng Yu on 2023/3/25.
//

#include "llama.h"
#include "llama_context.h"
#include <fmt/core.h>

llama_context *llama_model::init_from_file(const std::string &path_model, llama_context_params &params) {
    llama_context *ctx = new llama_context();

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }

    ctx->rng = std::mt19937(params.seed);
    ctx->logits_all = params.logits_all;

    ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;

    if (!load_model(path_model, *ctx, params.n_ctx, params.n_parts, memory_type,
                    params.vocab_only)) {
        fprintf(stderr, "%s: failed to load model\n", __func__);
        llama_free(ctx);
        return nullptr;
    }

    if (params.use_mlock) {
        char *err;
        if (!ggml_mlock(ctx->model.ctx,
                        ctx->model.mm_addr,
                        ctx->model.mm_length,
                        &err)) {
            fprintf(stderr, "%s\n", err);
            free(err);
            llama_free(ctx);
            return nullptr;
        }
    }

    // reserve memory for context buffers
    {
        if (!ctx->model.kv_self.kv_cache_init(ctx->model.hparams, memory_type, ctx->model.hparams.n_ctx)) {
            fprintf(stderr, "%s: kv_cache_init() failed for self-attention cache\n", __func__);
            llama_free(ctx);
            return nullptr;
        }

        {
            const size_t memory_size = ggml_nbytes(ctx->model.kv_self.k) + ggml_nbytes(ctx->model.kv_self.v);
            fprintf(stderr, "%s: kv self size  = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
        }

        const auto &hparams = ctx->model.hparams;

        // resized during inference
        if (params.logits_all) {
            ctx->logits.reserve(hparams.n_ctx * hparams.n_vocab);
        } else {
            ctx->logits.reserve(hparams.n_ctx);
        }

        if (params.embedding) {
            ctx->embedding.resize(hparams.n_embd);
        }

        ctx->buf_compute.resize(MEM_REQ_EVAL.at(ctx->model.type));

        ctx->buf_scratch[0].resize(MEM_REQ_SCRATCH0.at(ctx->model.type));
        ctx->buf_scratch[1].resize(MEM_REQ_SCRATCH1.at(ctx->model.type));
    }

    return ctx;
}
