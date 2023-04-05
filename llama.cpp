//
// Created by Yifeng Yu on 2023/3/25.
//

#include "llama.h"
#include "llama_context.h"
#include <fmt/core.h>
#include <fstream>
bool llama_model::build_ggml_ctx() {
    //auto &ctx = this->ctx;
    size_t ctx_size = 0;
    {

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        ctx_size += n_embd * n_vocab * ggml_type_sizef(load_ctx.wtype); // tok_embeddings

        ctx_size += n_embd * ggml_type_sizef(GGML_TYPE_F32); // norm

        ctx_size += n_embd * n_vocab * ggml_type_sizef(load_ctx.wtype); // output

        ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // attention_norm

        ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(load_ctx.wtype)); // wq
        ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(load_ctx.wtype)); // wk
        ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(load_ctx.wtype)); // wv
        ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(load_ctx.wtype)); // wo

        ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ffn_norm

        ctx_size += n_layer * (load_ctx.n_ff * n_embd * ggml_type_sizef(load_ctx.wtype)); // w1
        ctx_size += n_layer * (load_ctx.n_ff * n_embd * ggml_type_sizef(load_ctx.wtype)); // w2
        ctx_size += n_layer * (load_ctx.n_ff * n_embd * ggml_type_sizef(load_ctx.wtype)); // w3

        ctx_size += n_ctx * n_layer * n_embd * ggml_type_sizef(GGML_TYPE_F32); // memory_k
        ctx_size += n_ctx * n_layer * n_embd * ggml_type_sizef(GGML_TYPE_F32); // memory_v

        ctx_size += (5 + 10 * n_layer) * 256; // object overhead

        fprintf(stderr, "%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size / (1024.0 * 1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
                /*.mem_size   =*/ ctx_size,
                /*.mem_buffer =*/ NULL,
        };

        this->ctx = ggml_init(params);
        if (!this->ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }
    return true;
}

void llama_model::determine_ggml_file_split(std::string &name) {
    // split_type = 0: split by columns
    // split_type = 1: split by rows
    int split_type = 0;

    // split_type = 0:
    // regex:
    //   - tok_embeddings.*
    //   - layers.*.attention.wo.weight
    //   - layers.*.feed_forward.w2.weight

    // split_type = 1:
    // regex:
    //   - output.*
    //   - layers.*.attention.wq.weight
    //   - layers.*.attention.wk.weight
    //   - layers.*.attention.wv.weight
    //   - layers.*.feed_forward.w1.weight
    //   - layers.*.feed_forward.w3.weight
    if (name.find("tok_embeddings") != std::string::npos) {
        this->load_ctx.split_type = SPLIT_TYPE_COLUMN;
    } else if (name.find("layers") != std::string::npos) {
        if (name.find("attention.wo.weight") != std::string::npos) {
            this->load_ctx.split_type = SPLIT_TYPE_COLUMN;
        } else if (name.find("feed_forward.w2.weight") != std::string::npos) {
            this->load_ctx.split_type = SPLIT_TYPE_COLUMN;
        } else {
            this->load_ctx.split_type = SPLIT_TYPE_ROW;
        }
    } else if (name.find("output") != std::string::npos) {
        this->load_ctx.split_type = SPLIT_TYPE_ROW;
    }
}



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


size_t llama_load_ctx::determine_bpe(int32_t ftype, int ne[]) {
    size_t bpe;
    switch (ftype) {
        case 0:
            bpe = ggml_type_size(GGML_TYPE_F32);
            break;
        case 1:
            bpe = ggml_type_size(GGML_TYPE_F16);
            break;
        case 2:
            bpe = ggml_type_size(GGML_TYPE_Q4_0);
            assert(ne[0] % 64 == 0);
            break;
        case 3:
            bpe = ggml_type_size(GGML_TYPE_Q4_1);
            assert(ne[0] % 64 == 0);
            break;
        default: {
            return -1;
        }
    }
    return bpe;
}

int llama_model::load_model_tensor(std::ifstream &fin, int part_id) {
    int32_t n_dims;
    int32_t length;
    int32_t ftype;
    size_t total_size = 0;

    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
    fin.read(reinterpret_cast<char *>(&length), sizeof(length));
    fin.read(reinterpret_cast<char *>(&ftype), sizeof(ftype));

    if (fin.eof()) {
        return -1;
    }

    int32_t nelements = 1;
    int32_t ne[2] = {1, 1};
    for (int i = 0; i < n_dims; ++i) {
        fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        nelements *= ne[i];
    }

    std::string name(length, 0);
    fin.read(&name[0], length);

    if (this->tensors.find(name.data()) == this->tensors.end()) {
        fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
        return false;
    }

    determine_ggml_file_split(name);
    auto tensor = this->tensors[name.data()];
    if (!verify_tensor_shape_and_dim(tensor, name, load_ctx.n_parts, n_dims, nelements, ne)) {
        return false;
    }

    size_t bpe = load_ctx.determine_bpe(ftype, ne);
    if (bpe == -1ul) {
        fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
        return false;
    }

    if (n_dims == 1 || load_ctx.n_parts == 1) {
        if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
            fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                    __func__, name.data(), ggml_nbytes(tensor), nelements * bpe);
            return false;
        }

        if (part_id == 0) {
            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
        } else {
            fin.seekg(ggml_nbytes(tensor), std::ios::cur);
        }

        total_size += ggml_nbytes(tensor);
    } else {
        if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor) / (load_ctx.n_parts)) {
            fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                    __func__, name.data(), ggml_nbytes(tensor) / load_ctx.n_parts, nelements * bpe);
            return false;
        }

        if (load_ctx.is_column_split_type()) {
            const int np0 = ne[0];

            const size_t row_size = (tensor->ne[0] / ggml_blck_size(tensor->type)) * ggml_type_size(tensor->type);
            assert(row_size == tensor->nb[1]);

            for (int i1 = 0; i1 < ne[1]; ++i1) {
                const size_t offset_row = i1 * row_size;
                const size_t offset = offset_row + ((part_id * np0) / ggml_blck_size(tensor->type)) *
                                                   ggml_type_size(tensor->type);
                fin.read(reinterpret_cast<char *>(tensor->data) + offset, row_size / load_ctx.n_parts);
            }
        } else {
            const int np1 = ne[1];
            const size_t row_size = (tensor->ne[0] / ggml_blck_size(tensor->type)) * ggml_type_size(tensor->type);

            for (int i1 = 0; i1 < ne[1]; ++i1) {
                const size_t offset_row = (i1 + part_id * np1) * row_size;
                fin.read(reinterpret_cast<char *>(tensor->data) + offset_row, row_size);
            }
        }

        total_size += ggml_nbytes(tensor) / load_ctx.n_parts;
    }

    fprintf(stderr, "%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1],
            ftype == 0 ? "float" : "f16", ggml_nbytes(tensor) / 1024.0 / 1024.0);

    return total_size;
}
