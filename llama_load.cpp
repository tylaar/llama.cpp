//
// Created by Yifeng Yu on 2023/4/5.
//

#include <iostream>
#include <fstream>
#include <fmt/core.h>
#include "llama.h"
#include "llama_context.h"
#include "llama_loader.h"
#include "llama_memory_mapper.h"

bool llama_model::load_model(const std::string &fname,
                             llama_context &ctx,
                             int n_ctx,
                             int n_parts,
                             ggml_type memory_type,
                             bool vocab_only) {
    fprintf(stderr, "%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    ctx.t_start_us = ggml_time_us();

    auto &model = ctx.model;
    auto &vocab = ctx.vocab;

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    auto loader = new llama_loader(model, vocab, fin, fname);
    std::vector<char> f_buf(1024 * 1024);
    fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());

    fin.seekg(0, fin.end);
    const size_t file_size = fin.tellg();
    fin.seekg(0);

    // verify magic
    if (!loader->verify_model_magic()) {
        return false;
    }

    int n_ff = loader->load_model_hyper_params(n_ctx, n_parts);

    loader->load_model_vocab();

    if (vocab_only) {
        return true;
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    // wtype is for per-layer weights, while vtype is for other weights
    ggml_type wtype, vtype;
    std::pair<ggml_type, ggml_type> types;
    try {
        loader->determine_ggml_type();
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        fin.close();
        return false;
    }

    // map model into memory
    loader->mmap_memory();

    // calculate ctx size.
    size_t ctx_size = loader->calculate_ctx_size();
    // print memory requirements
    loader->print_memory_loaded(memory_type, ctx_size);

    // create the ggml context
    {
        ctx.model.buf.resize(ctx_size);

        struct ggml_init_params params = {
                /*.mem_size   =*/ model.buf.size(),
                /*.mem_buffer =*/ model.buf.data(),
                /*.no_alloc   =*/ true,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    loader->prepare_layer_memory(n_ff);

    //std::vector<uint8_t> tmp;

    std::cerr << __func__  << " loading tensors from " << fname.c_str() << std::endl;

    // load weights
    size_t total_size = 0;
    try {
        total_size = loader->load_layer_weight();
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        fin.close();
        return false;
    }

    fin.close();

    fprintf(stderr, "%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0, model.n_loaded);
    if (model.n_loaded == 0) {
        fprintf(stderr, "%s: WARN no tensors loaded from model file - assuming empty model for testing\n", __func__);
    } else if (model.n_loaded != (int) model.tensors.size()) {
        fprintf(stderr, "%s: ERROR not all tensors loaded from model file - expected %zu, got %d\n", __func__, model.tensors.size(),
                model.n_loaded);
        return false;
    }

    // loading time will be recalculate after the first eval, so
    // we take page faults deferred by mmap() into consideration
    ctx.t_load_us = ggml_time_us() - ctx.t_start_us;

    return true;
}


void llama_model::llama_free(llama_context *ctx) {
    ctx->model.kv_self.kv_cache_free();

    if (ctx->model.ctx) {
        ggml_free(ctx->model.ctx);
    }

    if (ctx->model.mm_addr) {
        llama_memory_mapper::munmap_file(ctx->model.mm_addr, ctx->model.mm_length);
    }

    delete ctx;
}
