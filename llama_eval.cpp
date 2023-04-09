//
// Created by Yifeng Yu on 2023/4/5.
//


// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
// The GPT-J model requires about 16MB of memory per input token.
//

#include "llama.h"
#include "llama_context.h"

int llama_model::eval(
        const llama_token *tokens,
        int n_tokens,
        int n_past,
        int n_threads) {
    if (!eval_internal(tokens, n_tokens, n_past, n_threads)) {
        fprintf(stderr, "%s: failed to eval\n", __func__);
        return 1;
    }
    // get a more accurate load time, upon first eval
    if (!lctx->has_evaluated_once) {
        lctx->t_load_us = ggml_time_us() - lctx->t_start_us;
        lctx->has_evaluated_once = true;
    }
    return 0;
}

bool llama_model::eval_internal(
        const llama_token *tokens,
        const int n_tokens,
        const int n_past,
        const int n_threads) {
    const int64_t t_start_us = ggml_time_us();

    const int N = n_tokens;
    const auto &hparams = this->hparams;

    auto &kv_self = this->kv_self;

    LLAMA_ASSERT(!!kv_self.ctx);

    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_vocab = hparams.n_vocab;

    auto &mem_per_token = lctx->mem_per_token;
    auto &buf_compute = lctx->buf_compute;

    struct ggml_init_params params = {
            /*.mem_size   =*/ buf_compute.size(),
            /*.mem_buffer =*/ buf_compute.data(),
            /*.no_alloc   =*/ false,
    };

    struct ggml_context *ctx0 = ggml_init(params);

    // for big prompts, if BLAS is enabled, it is better to use only one thread
    // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
    ggml_cgraph gf = {};
    gf.n_threads = N >= 32 && ggml_cpu_has_blas() ? 1 : n_threads;

    struct ggml_tensor *embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, tokens, N * ggml_element_size(embd));

    struct ggml_tensor *inpL = ggml_get_rows(ctx0, tok_embeddings, embd);

    long long total_sa_time = 0;
    long long total_ffn_time = 0;
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor *inpSA = inpL;
        struct ggml_tensor *cur;

        lctx->use_buf(ctx0, 0);

        // norm
        cur = eval_norm(ctx0, inpL, layers[il].attention_norm);

        // self-attention

        auto sa_time = ggml_time_us();
        cur = eval_self_attention(&gf, ctx0, cur, il, n_past, N);
        total_sa_time += sa_time;

        lctx->use_buf(ctx0, 1);

        struct ggml_tensor *inpFF = ggml_add(ctx0, cur, inpSA);

        auto ffn_time = ggml_time_us();
        // feed-forward network
        {
            // norm
            cur = eval_norm(ctx0, inpFF, layers[il].ffn_norm);

            struct ggml_tensor *tmp = ggml_mul_mat(ctx0,
                                                   layers[il].w3,
                                                   cur);

            cur = ggml_mul_mat(ctx0,
                               layers[il].w1,
                               cur);

            // SILU activation
            cur = ggml_silu(ctx0, cur);

            cur = ggml_mul(ctx0, cur, tmp);

            cur = ggml_mul_mat(ctx0,
                               layers[il].w2,
                               cur);
        }

        total_ffn_time += ffn_time;
        cur = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        inpL = cur;
    }

    std::cerr << "avg self-attention time" << total_sa_time / n_layer / (1000.0) <<  "ms and fnn time " << total_ffn_time / n_layer / (1000.0) << "ms" << std::endl;
    std::cerr << "total self-attention time" << total_sa_time / 1000.0 <<  "ms and fnn time " << total_ffn_time / 1000.0 << "ms" << std::endl;
    lctx->use_buf(ctx0, 0);

    // used at the end to optionally extract the embeddings
    struct ggml_tensor *embeddings = NULL;

    // norm
    inpL = eval_norm(ctx0, inpL, norm);

    // lm_head
    auto fmm_time = ggml_time_us();
    inpL = ggml_mul_mat(ctx0, output, inpL);
    std::cerr << "final mul_mat for layer cost " << ggml_time_us() - fmm_time << std::endl;

    lctx->use_buf(ctx0, -1);

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute(ctx0, &gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // extract logits
    {
        auto &logits_out = lctx->logits;

        if (lctx->logits_all) {
            logits_out.resize(n_vocab * N);
            memcpy(logits_out.data(), (float *) ggml_get_data(inpL), sizeof(float) * n_vocab * N);
        } else {
            // return result for just the last token
            logits_out.resize(n_vocab);
            memcpy(logits_out.data(), (float *) ggml_get_data(inpL) + (n_vocab * (N - 1)), sizeof(float) * n_vocab);
        }
    }

    // extract embeddings
    if (lctx->embedding.size()) {
        auto &embedding_out = lctx->embedding;

        embedding_out.resize(n_embd);
        memcpy(embedding_out.data(), (float *) ggml_get_data(embeddings) + (n_embd * (N - 1)), sizeof(float) * n_embd);
    }

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0) / N;
    }

#if 0
    printf("\n%s: used_mem = %.3f MB, scratch -- %.3f MB %.3f MB\n", __func__,
            ggml_used_mem(ctx0)/1024.0/1024.0,
            lctx.get_buf_max_mem(0)/1024.0/1024.0,
            lctx.get_buf_max_mem(1)/1024.0/1024.0);
#endif

    ggml_free(ctx0);

    // measure the performance only for the single-token evals
    if (N == 1) {
        lctx->t_eval_us += ggml_time_us() - t_start_us;
        lctx->n_eval++;
    } else if (N > 1) {
        lctx->t_p_eval_us += ggml_time_us() - t_start_us;
        lctx->n_p_eval += N;
    }

    return true;
}

ggml_tensor *llama_model::eval_self_attention(ggml_cgraph *gf, ggml_context *ctx0, ggml_tensor *cur, int il, int n_past, int N) {
    const int n_embd = hparams.n_embd;
    const int n_ctx = hparams.n_ctx;
    const int n_head = hparams.n_head;
    const int n_rot = hparams.n_embd / hparams.n_head;

    // self-attention

    struct ggml_tensor *Qcur = ggml_mul_mat(ctx0, layers[il].wq, cur);
    struct ggml_tensor *Kcur = ggml_mul_mat(ctx0, layers[il].wk, cur);
    struct ggml_tensor *Vcur = ggml_mul_mat(ctx0, layers[il].wv, cur);

    // store key and value to memory
    if (N >= 1) {
        struct ggml_tensor *k = ggml_view_1d(ctx0, kv_self.k, N * n_embd, (ggml_element_size(kv_self.k) * n_embd) * (il * n_ctx + n_past));
        struct ggml_tensor *v = ggml_view_1d(ctx0, kv_self.v, N * n_embd, (ggml_element_size(kv_self.v) * n_embd) * (il * n_ctx + n_past));

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
    }

    // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
    struct ggml_tensor *Q =
            ggml_permute(ctx0,
                         ggml_rope(ctx0,
                                   ggml_cpy(ctx0,
                                            Qcur,
                                            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd / n_head, n_head, N)),
                                   n_past, n_rot, 0),
                         0, 2, 1, 3);

    // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
    struct ggml_tensor *K =
            ggml_permute(ctx0,
                         ggml_rope(ctx0,
                                   ggml_reshape_3d(ctx0,
                                                   ggml_view_1d(ctx0, kv_self.k, (n_past + N) * n_embd,
                                                                il * n_ctx * ggml_element_size(kv_self.k) * n_embd),
                                                   n_embd / n_head, n_head, n_past + N),
                                   n_past, n_rot, 1),
                         0, 2, 1, 3);

    // K * Q
    struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);

    // KQ_scaled = KQ / sqrt(n_embd/n_head)
    struct ggml_tensor *KQ_scaled =
            ggml_scale(ctx0,
                       KQ,
                       ggml_new_f32(ctx0, 1.0f / sqrtf(float(n_embd) / n_head)));

    // KQ_masked = mask_past(KQ_scaled)
    struct ggml_tensor *KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

    // KQ = soft_max(KQ_masked)
    struct ggml_tensor *KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

    // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
    struct ggml_tensor *V_trans =
            ggml_cpy(ctx0,
                     ggml_permute(ctx0,
                                  ggml_reshape_3d(ctx0,
                                                  ggml_view_1d(ctx0, kv_self.v, (n_past + N) * n_embd,
                                                               il * n_ctx * ggml_element_size(kv_self.v) * n_embd),
                                                  n_embd / n_head, n_head, n_past + N),
                                  1, 2, 0, 3),
                     ggml_new_tensor_3d(ctx0, kv_self.v->type, n_past + N, n_embd / n_head, n_head));

    // KQV = transpose(V) * KQ_soft_max
    struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

    // KQV_merged = KQV.permute(0, 2, 1, 3)
    struct ggml_tensor *KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

    // cur = KQV_merged.contiguous().view(n_embd, N)
    cur = ggml_cpy(ctx0,
                   KQV_merged,
                   ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

    // projection (no bias)
    cur = ggml_mul_mat(ctx0,
                       layers[il].wo,
                       cur);
    return cur;
}

ggml_tensor *llama_model::eval_norm(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *norm) {
    cur = ggml_rms_norm(ctx0, cur);

    // inpL = norm*inpL
    cur = ggml_mul(ctx0,
                   ggml_repeat(ctx0, norm, cur),
                   cur);
    return cur;
}