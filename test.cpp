#include "ggml.h"
#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <fmt/core.h>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <unistd.h>

static const size_t MB = 1024*1024;

struct test_hparams {
    int32_t n_vocab = 10;
    int32_t n_ctx = 10;
    int32_t n_embd = 4;
    int32_t n_head = 2;
    int32_t n_layer = 1;
    int32_t n_rot = 2;
    int32_t f16 = 0;
};


class llama_kv_cache {
public:
    struct ggml_tensor * k;
    struct ggml_tensor * v;

    struct ggml_context * ctx;

    std::vector<uint8_t> buf;

    int n; // number of tokens currently in the cache
    bool kv_cache_init(
            const struct test_hparams& hparams,
            ggml_type   wtype,
            int   n_ctx) {
        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;

        const int n_mem      = n_layer*n_ctx;
        const int n_elements = n_embd*n_mem;

        buf.resize(2u*n_elements*ggml_type_size(wtype) + 2u*MB);

        struct ggml_init_params params;
        params.mem_size   = buf.size();
        params.mem_buffer = buf.data();
        params.no_alloc   = false;

        ctx = ggml_init(params);

        if (!ctx) {
            fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
            return false;
        }

        k = ggml_new_tensor_1d(ctx, wtype, n_elements);
        v = ggml_new_tensor_1d(ctx, wtype, n_elements);

        return true;
    }

    void kv_cache_free() {
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
    }
};


struct test_layer {
    //struct ggml_tensor * c_attn_k_v_w;
    //struct ggml_tensor * c_attn_k_v_b;
    ggml_tensor * c_attn_q_w;
    ggml_tensor * c_attn_q_b;

    ggml_tensor * c_attn_k_w;
    ggml_tensor * c_attn_k_b;

    ggml_tensor * c_attn_v_w;
    ggml_tensor * c_attn_v_b;

    ggml_tensor * c_l_norm_w;
    ggml_tensor * c_l_norm_b;

    ggml_tensor * c_p_l_norm_w;
    ggml_tensor * c_p_l_norm_b;

    ggml_tensor * c_l_dense_w;
    ggml_tensor * c_l_dense_b;

    // ff
    struct ggml_tensor * c_mlp_h_to_4h_w;
    struct ggml_tensor * c_mlp_h_to_4h_b;

    struct ggml_tensor * c_mlp_4h_to_h_w;
    struct ggml_tensor * c_mlp_4h_to_h_b;
};

struct test_model {
    test_hparams hparams;
    struct ggml_tensor *embed_in_wte;
    struct ggml_tensor *embed_out_wte;

    std::vector<test_layer> layers;

    struct ggml_tensor *final_norm_w;
    struct ggml_tensor *final_norm_b;

    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    struct ggml_context * ctx;
    // key + value cache for the self attention
    // TODO: move to llama_state
    struct llama_kv_cache kv_self;

    std::map<std::string, struct ggml_tensor *> tensors;
};

gpt_vocab build_vocab() {

    gpt_vocab* vocab = new gpt_vocab();
    vocab->id_to_token[0] = "a";
    vocab->token_to_id["a"] = 0;

    vocab->id_to_token[1] = "b";
    vocab->token_to_id["b"] = 1;

    vocab->id_to_token[2] = "c";
    vocab->token_to_id["c"] = 2;

    vocab->id_to_token[3] = "d";
    vocab->token_to_id["d"] = 3;

    vocab->id_to_token[4] = "e";
    vocab->token_to_id["e"] = 4;

    vocab->id_to_token[5] = "f";
    vocab->token_to_id["f"] = 5;

    vocab->id_to_token[6] = "g";
    vocab->token_to_id["g"] = 6;

    vocab->id_to_token[7] = "h";
    vocab->token_to_id["h"] = 7;

    vocab->id_to_token[8] = "i";
    vocab->token_to_id["i"] = 8;

    vocab->id_to_token[9] = "j";
    vocab->token_to_id["j"] = 9;
    return *vocab;
}

bool load_model(const std::string & fname, test_model & model, gpt_vocab & vocab) {
    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    auto &ctx = model.ctx;
    size_t ctx_size = 0;


    // load hparams
    {
        auto & hparams = model.hparams;

        fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        fin.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fin.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fin.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *) &hparams.n_rot, sizeof(hparams.n_rot));
        fin.read((char *) &hparams.f16,     sizeof(hparams.f16));

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: f16     = %d\n", __func__, hparams.f16);
    }

    ggml_type wtype = GGML_TYPE_COUNT;
    switch (model.hparams.f16) {
        case 0: wtype = GGML_TYPE_F32;  break;
        case 1: wtype = GGML_TYPE_F16;  break;
        case 2: wtype = GGML_TYPE_Q4_0; break;
        case 3: wtype = GGML_TYPE_Q4_1; break;
        default:
        {
            fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                    __func__, fname.c_str(), model.hparams.f16);
            return false;
        }
    }

    const auto & hparams = model.hparams;

    if (!model.kv_self.kv_cache_init(model.hparams, GGML_TYPE_F32, model.hparams.n_ctx)) {
        fprintf(stderr, "%s: kv_cache_init() failed for self-attention cache\n", __func__);
        return false;
    }

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_vocab = hparams.n_vocab;
    const int n_rot = hparams.n_rot;

    ctx_size += n_embd*n_vocab*ggml_type_sizef(wtype); // embed_in_wte
    ctx_size += n_embd*n_vocab* ggml_type_sizef(wtype); // embed_out_wte
    ctx_size += n_embd * ggml_type_sizef(wtype); // final norm weight
    ctx_size += n_embd * ggml_type_sizef(wtype);
    {
        ctx_size += n_layer*(3*n_embd*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // attn_k_v_w TODO: in qkv in total, not isolated yet.
        ctx_size += n_layer*(3*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // attn_k_v_b;

        ctx_size += n_layer*(n_embd* ggml_type_sizef(GGML_TYPE_F32)); // layer_norm_w;
        ctx_size += n_layer*(n_embd* ggml_type_sizef(GGML_TYPE_F32)); // layer_norm_b;

        ctx_size += n_layer*(n_embd* ggml_type_sizef(GGML_TYPE_F32)); // post_layer_norm_w;
        ctx_size += n_layer*(n_embd* ggml_type_sizef(GGML_TYPE_F32)); // post_layer_norm_b;

        ctx_size += n_layer*(n_embd* n_embd* ggml_type_size(GGML_TYPE_F32)); // output_dense_weight
        ctx_size += n_layer*(n_embd * ggml_type_sizef(GGML_TYPE_F32)); // output_dense_bias

        // ff part
        ctx_size += n_layer*(n_embd*n_embd*4 * ggml_type_size(GGML_TYPE_F32));
        ctx_size += n_layer*(n_embd*4 * ggml_type_size(GGML_TYPE_F32));
        ctx_size += n_layer*(n_embd*n_embd*4 * ggml_type_size(GGML_TYPE_F32));
        ctx_size += n_layer*(n_embd * ggml_type_size(GGML_TYPE_F32));

        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_k TODO for caching??
        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_v TODO for caching??

    }
    ctx_size += n_embd * ggml_type_sizef(GGML_TYPE_F32);
    ctx_size += (5 + 10*n_layer)*256*2; // object overhead

    printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));

    // create the ggml context
    {
        struct ggml_init_params params = {
                .mem_size   = ctx_size,
                .mem_buffer = NULL,
                .no_alloc   = false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;
        const int n_rot = hparams.n_rot;

        model.layers.resize(n_layer);

        model.embed_in_wte = ggml_new_tensor_2d(ctx, wtype,  n_embd, n_vocab);
        model.embed_out_wte = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);
        model.final_norm_w = ggml_new_tensor_1d(ctx, wtype, n_embd);
        model.final_norm_b = ggml_new_tensor_1d(ctx, wtype, n_embd);
        model.tensors["embd_in"] = model.embed_in_wte;
        model.tensors["embd_out"] = model.embed_out_wte;
        model.tensors["final_norm_w"] = model.final_norm_w;
        model.tensors["final_norm_b"] = model.final_norm_b;
        for (int i = 0 ; i < n_layer; i++) {
            auto & layer = model.layers[i];

            layer.c_attn_q_w = ggml_new_tensor_2d(ctx, wtype,n_embd, n_embd);
            layer.c_attn_q_b = ggml_new_tensor_1d(ctx, wtype,n_embd);
            model.tensors["c_l_" + std::to_string(i) + "_attn_q_w"] = layer.c_attn_q_w;
            model.tensors["c_l_" + std::to_string(i) + "_attn_q_b"] = layer.c_attn_q_b;

            layer.c_attn_k_w = ggml_new_tensor_2d(ctx, wtype,n_embd, n_embd);
            layer.c_attn_k_b = ggml_new_tensor_1d(ctx, wtype,n_embd);
            model.tensors["c_l_" + std::to_string(i) + "_attn_k_w"] = layer.c_attn_k_w;
            model.tensors["c_l_" + std::to_string(i) + "_attn_k_b"] = layer.c_attn_k_b;

            layer.c_attn_v_w = ggml_new_tensor_2d(ctx, wtype,n_embd, n_embd);
            layer.c_attn_v_b = ggml_new_tensor_1d(ctx, wtype,n_embd);
            model.tensors["c_l_" + std::to_string(i) + "_attn_v_w"] = layer.c_attn_v_w;
            model.tensors["c_l_" + std::to_string(i) + "_attn_v_b"] = layer.c_attn_v_b;

            layer.c_l_norm_w = ggml_new_tensor_1d(ctx, wtype,n_embd);
            model.tensors["c_l_" + std::to_string(i) + "_norm_w"] = layer.c_l_norm_w;
            layer.c_l_norm_b = ggml_new_tensor_1d(ctx, wtype,n_embd);
            model.tensors["c_l_" + std::to_string(i) + "_norm_b"] = layer.c_l_norm_b;
            layer.c_p_l_norm_w = ggml_new_tensor_1d(ctx, wtype,n_embd);
            model.tensors["c_l_" + std::to_string(i) + "_p_norm_w"] = layer.c_p_l_norm_w;
            layer.c_p_l_norm_b = ggml_new_tensor_1d(ctx, wtype,n_embd);
            model.tensors["c_l_" + std::to_string(i) + "_p_norm_b"] = layer.c_p_l_norm_b;

            layer.c_l_dense_w = ggml_new_tensor_2d(ctx, wtype,n_embd, n_embd);
            layer.c_l_dense_b = ggml_new_tensor_1d(ctx, wtype,n_embd);
            model.tensors["c_l_" + std::to_string(i) + "_dense_w"] = layer.c_l_dense_w;
            model.tensors["c_l_" + std::to_string(i) + "_dense_b"] = layer.c_l_dense_b;

            layer.c_mlp_h_to_4h_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd*4);
            layer.c_mlp_h_to_4h_b = ggml_new_tensor_1d(ctx, wtype, n_embd*4);
            model.tensors["c_l_" + std::to_string(i) + "_mlp_h_4h_w"] = layer.c_mlp_h_to_4h_w;
            model.tensors["c_l_" + std::to_string(i) + "_mlp_h_4h_b"] = layer.c_mlp_h_to_4h_b;
            layer.c_mlp_4h_to_h_w = ggml_new_tensor_2d(ctx, wtype, n_embd*4, n_embd);
            layer.c_mlp_4h_to_h_b = ggml_new_tensor_1d(ctx, wtype, n_embd);
            model.tensors["c_l_" + std::to_string(i) + "_mlp_4h_h_w"] = layer.c_mlp_4h_to_h_w;
            model.tensors["c_l_" + std::to_string(i) + "_mlp_4h_h_b"] = layer.c_mlp_4h_to_h_b;
        }
    }

    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;

        const int n_mem      = n_layer*n_ctx;
        const int n_elements = n_embd*n_mem;

        model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);

        const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

        printf("%s: memory_size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }

    // loading
    {
        int n_tensors = 0;
        size_t total_size = 0;

        printf("%s: ", __func__);

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

            if (fin.eof()) {
                break;
            }

            int64_t nelements = 1;
            int64_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                int32_t ne_cur;
                fin.read(reinterpret_cast<char *>(&ne_cur), sizeof(ne_cur));
                ne[i] = ne_cur;
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            std::cout << "trying to load tensor " << name << std::endl;
            if (model.tensors.find(name.data()) == model.tensors.end()) {
                std::cerr << fmt::format("{}: unknown tensor '{}' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements) {
                std::cerr << fmt::format("{}: tensor '{}' has wrong size in model file\n", __func__, name.data());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                std::cerr << fmt::format("{}: tensor '{}' has wrong shape in model file: got [{}, {}], expected [{}, {}]\n",
                                         __func__, name.data(), ne[0], ne[1], tensor->ne[0], tensor->ne[1]);
                return false;
            }

            size_t bpe = 0;

            switch (ftype) {
                case 0: bpe = ggml_type_size(GGML_TYPE_F32);  break;
                case 1: bpe = ggml_type_size(GGML_TYPE_F16);  break;
                case 2: bpe = ggml_type_size(GGML_TYPE_Q4_0); assert(ne[0] % 64 == 0); break;
                case 3: bpe = ggml_type_size(GGML_TYPE_Q4_1); assert(ne[0] % 64 == 0); break;
                default:
                {
                    std::cerr << fmt::format("{}: unknown ftype {} in model file\n", __func__, ftype);
                    return false;
                }
            }

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                std::cerr << fmt::format("{}: tensor '{}' has wrong size in model file: got {}, expected {}\n",
                                         __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            //printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 8 == 0) {
                printf(".");
                fflush(stdout);
            }
        }

        printf(" done\n");

        printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size/1024.0/1024.0, n_tensors);
    }

    fin.close();

    return true;
}

std::string default_prompts = "Hello, I am";

struct ggml_tensor* eval(
        const test_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
        size_t                     & mem_per_token) {
    const int N = embd_inp.size();

    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_vocab;
    const int n_rot   = hparams.n_rot;
    const int n_hidden = n_embd * 3;
    const int head_size = n_embd/n_head;

    static size_t buf_size = 256u*1024*1024;
    static void * buf = malloc(buf_size);

    if (mem_per_token > 0 && mem_per_token*N > buf_size) {
        const size_t buf_size_new = 1.1*(mem_per_token*N); // add 10% to account for ggml object overhead
        //printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

        // reallocate
        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr) {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return nullptr;
        }
    }

    struct ggml_init_params params = {
            .mem_size   = buf_size,
            .mem_buffer = buf,
            .no_alloc   = false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = { .n_threads = n_threads }; //TODO: thread 1 first for safety.

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, embd_inp.data(), N*ggml_element_size(embd));
    ggml_tensor *alpha = ggml_new_f32(ctx0, 0.1250);
    // embed_in_wte
    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.embed_in_wte, embd);
    //ggml_build_forward_expand(&gf, inpL);
    ggml_tensor* inpSA_debug;
    ggml_tensor* final_norm_debug;
    ggml_tensor* out_debug;
    ggml_tensor* logits;
    ggml_tensor* q_rot_debug;
    ggml_tensor* k_rot_debug;
    ggml_tensor* l0_dense;
    ggml_tensor* l1_dense;
    ggml_tensor* l1_q_rot_debug;
    ggml_tensor* l1_k_rot_debug;
/*
    ggml_tensor* inp_normed_debug;
    ggml_tensor* q_debug;
    ggml_tensor* k_debug;
    ggml_tensor* v_debug;

    ggml_tensor* q_reshape_debug;
    ggml_tensor* k_reshape_debug;
    ggml_tensor* v_reshape_debug;

    ggml_tensor* qk_debug;
    ggml_tensor* qk_scaled_debug;
    ggml_tensor* qk_causal_masked_debug;
    ggml_tensor* qk_softmax_debug;
    ggml_tensor* qkv_output_debug;
    ggml_tensor *qkv_permuted_debug;
    */
    ggml_tensor* qkv_merged_debug;
    ggml_tensor* qkv_densed_debug;

    ggml_tensor* qkv_copy;
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor *inpSA = inpL;
        struct ggml_tensor * cur;
        inpSA_debug = inpSA;
        {
            cur = ggml_norm(ctx0, inpL);

            // cur = ln_input_norm_w*cur + ln_input_norm_b
            cur = ggml_add(ctx0,
                           ggml_mul(ctx0,
                                    cur,
                                    ggml_repeat(ctx0,  model.layers[il].c_l_norm_w, cur)),
                           ggml_repeat(ctx0, model.layers[il].c_l_norm_b, cur));
        }

        auto qkv_post_normed = ggml_norm(ctx0, inpL);

        // cur = ln_input_norm_w*cur + ln_input_norm_b
        qkv_post_normed = ggml_add(ctx0,
                                   ggml_mul(ctx0,
                                            qkv_post_normed,
                                            ggml_repeat(ctx0,  model.layers[il].c_p_l_norm_w, qkv_post_normed)),
                                   ggml_repeat(ctx0, model.layers[il].c_p_l_norm_b, qkv_post_normed));


        //inp_normed_debug = cur;

        // self-attention
        {
            auto qw = model.layers[il].c_attn_q_w;
            auto qb = model.layers[il].c_attn_q_b;

            debug_print_tensor_lite(qw);
            auto kw = model.layers[il].c_attn_k_w;
            auto kb = model.layers[il].c_attn_k_b;
            debug_print_tensor_lite(kw);
            auto vw = model.layers[il].c_attn_v_w;
            auto vb = model.layers[il].c_attn_v_b;
            debug_print_tensor_lite(vw);
            auto q_m = ggml_add(ctx0,
                                ggml_mul_mat(ctx0, qw, cur),
                                ggml_repeat(ctx0, qb, cur));
            auto k_m = ggml_add(ctx0,
                                ggml_mul_mat(ctx0, kw, cur),
                                ggml_repeat(ctx0, kb, cur));
            auto v_m = ggml_add(ctx0,
                                ggml_mul_mat(ctx0, vw, cur),
                                ggml_repeat(ctx0, vb, cur));
            // TODO: viewing in 3d with slicing problematic.

            if (N >= 1) {
                struct ggml_tensor *k = ggml_view_1d(ctx0, model.kv_self.k, N * n_embd, (ggml_element_size(model.kv_self.k) * n_embd) * (il * n_ctx + n_past));
                struct ggml_tensor *v = ggml_view_1d(ctx0, model.kv_self.v, N * n_embd, (ggml_element_size(model.kv_self.v) * n_embd) * (il * n_ctx + n_past));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, k_m, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, v_m, v));

            }
            // q_reshape_debug = ggml_reshape_3d(ctx0, q_cur, n_embd / n_head, n_head, N);
            auto q = ggml_permute(ctx0,
                                  ggml_rope(ctx0,
                                            ggml_reshape_3d(ctx0,
                                                            q_m,
                                                            n_embd / n_head, n_head, N),
                                            n_past, n_rot, 0),
                                  0, 2, 1, 3);

            // k_reshape_debug = ggml_reshape_3d(ctx0, k_cur, n_embd / n_head, n_head, N);
            auto k = ggml_permute(ctx0,
                                  ggml_rope(ctx0,
                                            ggml_reshape_3d(ctx0,
                                                            ggml_view_1d(ctx0, model.kv_self.k, (n_past + N) * n_embd,
                                                                         il * n_ctx * ggml_element_size(model.kv_self.k) * n_embd),
                                                            n_embd / n_head, n_head, N),
                                            n_past, n_rot, 1),
                             0, 2, 1, 3);

/*

            auto q_rot = ggml_rope(ctx0,
                                   q, 0, n_rot, 0);
            auto k_rot = ggml_rope(ctx0,
                                   k, 0, n_rot, 1);

*/

            q_rot_debug = q;
            k_rot_debug = k;
            if (il == 1) {
                l1_k_rot_debug = k;
                l1_q_rot_debug = q;
            }
            auto v_cur = ggml_add(ctx0,
                                  ggml_mul_mat(ctx0, vw, cur),
                                  ggml_repeat(ctx0, vb, cur));

            auto v_permuted = ggml_permute(ctx0,
                                           ggml_reshape_3d(ctx0,
                                                           ggml_view_1d(ctx0, model.kv_self.v, (n_past + N) * n_embd,
                                                                        il * n_ctx * ggml_element_size(model.kv_self.v) * n_embd),
                                                           n_embd / n_head, n_head, n_past + N),
                                      1, 2, 0, 3);
            auto v = ggml_cpy(ctx0,
                              v_permuted,
                              ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, v_permuted->ne[0], v_permuted->ne[1], v_permuted->ne[2]));
            //q_debug = q;
            //k_debug = k;
            //v_debug = v;

            auto qk = ggml_mul_mat(ctx0, k, q);
            //qk_debug = qk;
            auto qk_scaled = ggml_scale(ctx0, qk, alpha);
            //qk_scaled_debug = qk_scaled;
            auto qk_causal_masked = ggml_diag_mask_inf(ctx0, qk_scaled, 0);
            //qk_causal_masked_debug = qk_causal_masked;
            auto qk_softmax = ggml_soft_max(ctx0, qk_causal_masked);
            //qk_softmax_debug = qk_softmax;

            auto qkv_output = ggml_mul_mat(ctx0, qk_softmax, v);
            //qkv_output_debug = qkv_output;

            auto qkv_permuted =
                    ggml_permute(ctx0,
                                 ggml_permute(ctx0, qkv_output, 1, 0, 2, 3),
                                 0, 2, 1, 3);
            //qkv_permuted_debug = qkv_permuted;
            auto qkv_merged = ggml_cpy(ctx0,
                           qkv_permuted,
                           ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            qkv_merged_debug = qkv_merged;
            auto qkv_densed = ggml_add(ctx0,
                                       ggml_mul_mat(ctx0, model.layers[il].c_l_dense_w, qkv_merged),
                                       ggml_repeat(ctx0, model.layers[il].c_l_dense_b, qkv_merged));
            qkv_densed_debug = qkv_densed;

            //lctx->use_buf(ctx0, 1);

            auto qkv_h_4h = ggml_mul_mat(ctx0, model.layers[il].c_mlp_h_to_4h_w, qkv_post_normed);
            qkv_h_4h = ggml_add(ctx0,
                                qkv_h_4h,
                                ggml_repeat(ctx0,  model.layers[il].c_mlp_h_to_4h_b, qkv_h_4h));

            auto gelu_ed = ggml_gelu(ctx0, qkv_h_4h);

            auto qkv_4h_h = ggml_mul_mat(ctx0, model.layers[il].c_mlp_4h_to_h_w, gelu_ed);
            qkv_4h_h = ggml_add(ctx0,
                                qkv_4h_h,
                                ggml_repeat(ctx0, model.layers[il].c_mlp_4h_to_h_b, qkv_4h_h));

            auto first_part = ggml_add(ctx0,qkv_densed, inpSA);
            //f_debug = first_part;
            auto second_part = qkv_4h_h;
            //s_debug = second_part;
            auto qkv_res = ggml_add(ctx0,
                                    first_part, second_part
            );
            qkv_copy = qkv_res;

            inpL = qkv_copy;
            if (il==0) {
                l0_dense = qkv_res;
            }
            if (il==1) {
                l1_dense = qkv_res;
            }
            //ggml_build_forward_expand(&gf, qkv_t);
            //ggml_build_forward_expand(&gf, qkv_t_reshaped);
            ggml_build_forward_expand(&gf, inpSA);
            ggml_build_forward_expand(&gf, q);
            ggml_build_forward_expand(&gf, k);
            ggml_build_forward_expand(&gf, v);
            /*ggml_build_forward_expand(&gf, q_rot);
            ggml_build_forward_expand(&gf, k_rot);
            */
            ggml_build_forward_expand(&gf, qk);
            ggml_build_forward_expand(&gf, qkv_res);
            ggml_build_forward_expand(&gf, qk_scaled);
            ggml_build_forward_expand(&gf, qk_causal_masked);
            ggml_build_forward_expand(&gf, qk_softmax);
            ggml_build_forward_expand(&gf, qkv_output);
            ggml_build_forward_expand(&gf, qkv_merged);
            ggml_build_forward_expand(&gf, qkv_densed);
            ggml_build_forward_expand(&gf, first_part);
            ggml_build_forward_expand(&gf, second_part);

            // Below are for debugging purpose
            ggml_build_forward_expand(&gf, qkv_densed_debug);
            ggml_build_forward_expand(&gf, qkv_merged_debug);
/*
            ggml_build_forward_expand(&gf, inp_normed_debug);
            ggml_build_forward_expand(&gf, q_reshape_debug);
            ggml_build_forward_expand(&gf, k_reshape_debug);
            ggml_build_forward_expand(&gf, v_reshape_debug);
            ggml_build_forward_expand(&gf, qk_debug);
            ggml_build_forward_expand(&gf, qk_scaled_debug);
            ggml_build_forward_expand(&gf, qk_causal_masked_debug);
            ggml_build_forward_expand(&gf, qk_softmax_debug);
            ggml_build_forward_expand(&gf, qkv_output_debug);
            ggml_build_forward_expand(&gf, qkv_merged_debug);
            ggml_build_forward_expand(&gf, qkv_densed_debug);
*/

            std::cout << "current q nelements:" << ggml_nelements(q) <<  " k nelem: " << ggml_nelements(k) << " v nelems:" << ggml_nelements(v) << std::endl;
        }

    }

    {
        // final norm
        auto final_normed = ggml_norm(ctx0, inpL);

        // cur = ln_input_norm_w*cur + ln_input_norm_b
        final_normed = ggml_add(ctx0,
                       ggml_mul(ctx0,
                                final_normed,
                                ggml_repeat(ctx0,  model.final_norm_w, final_normed)),
                       ggml_repeat(ctx0, model.final_norm_b, final_normed));
        final_norm_debug = final_normed;

        ggml_build_forward_expand(&gf, final_normed);
        logits = ggml_mul_mat(ctx0,  model.embed_out_wte, final_normed);
        out_debug = logits;
        ggml_build_forward_expand(&gf, logits);
    }
    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);
    //debug_print_tensor(gf.nodes[6]);
    //debug_print_tensor(gf.nodes[7]);
    std::cout << "==========input_normed_printing=============" << std::endl;
    debug_print_tensor_lite(inpSA_debug);
    std::cout << "==========input_normed_printend=============" << std::endl;
    std::cout << "==========l_0_hidden=============" << std::endl;
    debug_print_tensor_lite(l0_dense);
    std::cout << "==========l_1_hidden=============" << std::endl;
    debug_print_tensor_lite(l1_dense);
/*
    debug_print_tensor_lite(q_debug);
    std::cout << "==========query_reshape_printing=============" << std::endl;
    debug_print_tensor_lite(q_reshape_debug);
    std::cout << "==========query_reshape_printend=============" << std::endl;
    std::cout << "==========key_reshape_printing=============" << std::endl;
    debug_print_tensor_lite(k_reshape_debug);
    std::cout << "==========key_reshape_printend=============" << std::endl;
    std::cout << "==========value_reshape_printing=============" << std::endl;
    debug_print_tensor_lite(v_reshape_debug);
    std::cout << "==========value_reshape_printend=============" << std::endl;
    std::cout << "**********staring printing permuted****************" << std::endl;
    debug_print_tensor_lite(q_debug);
    debug_print_tensor_lite(k_debug);
    debug_print_tensor_lite(v_debug);
    std::cout << "**********ending printing permuted****************" << std::endl;
    std::cout << "==========qk_malmut_printing=============" << std::endl;
    debug_print_tensor_lite(qkv_copy);
    std::cout << "==========qk_malmut_printend=============" << std::endl;

    std::cout << "==========qk_malmut_scaled_printing=============" << std::endl;
    debug_print_tensor_lite(qk_scaled_debug);
    std::cout << "==========qk_malmut_scaled_printend=============" << std::endl;
    std::cout << "==========qk_malmut_masked_printing=============" << std::endl;
    debug_print_tensor_lite(qk_causal_masked_debug);
    std::cout << "==========qk_malmut_masked_printend=============" << std::endl;

    std::cout << "==========qk_softmax_printing=============" << std::endl;
    debug_print_tensor_lite(qk_softmax_debug);
    std::cout << "==========qk_softmax_printend=============" << std::endl;
    std::cout << "==========qkv_output_printing=============" << std::endl;
    debug_print_tensor_lite(qkv_output_debug);
    std::cout << "==========qkv_output_printend=============" << std::endl;
    std::cout << "==========qkv_permuted_printing=============" << std::endl;
    debug_print_tensor_lite(qkv_permuted_debug);
    std::cout << "==========qkv_permuted_printend=============" << std::endl;

    std::cout << "==========qkv_merged_printing=============" << std::endl;
    debug_print_tensor_lite(qkv_merged_debug);
    std::cout << "==========qkv_merged_printend=============" << std::endl;
    */
    std::cout << "==============rotated_query=============" << std::endl;
    debug_print_tensor_lite(l1_q_rot_debug);
    std::cout << "==============rotated_key=============" << std::endl;
    debug_print_tensor_lite(l1_k_rot_debug);

    std::cout << "==========qkv_densed_printing=============" << std::endl;
    debug_print_tensor_lite(qkv_copy);
    std::cout << "==========qkv_densed_printend=============" << std::endl;
    std::cout << "==========qkv_after_mlp_printing=============" << std::endl;
    debug_print_tensor_lite(out_debug);
    std::cout << "==========qkv_after_mlp_printend=============" << std::endl;


    //debug_print_graph_filter_type(&gf, GGML_OP_ADD);
    //debug_print_graph_filter_type(&gf, GGML_OP_RESHAPE);
    //debug_print_graph_filter_type(&gf, GGML_OP_VIEW);
    //debug_print_graph_filter_type(&gf, GGML_OP_PERMUTE);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result for just the last token
    //embd_w.resize(n_vocab);
    //memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);

    ggml_free(ctx0);

    return out_debug;
}

int main() {

    std::string fname = "/Users/yifengyu/hack/models/test/test.bin";
    test_model model;
    auto vocab = build_vocab();

    // load the model
    {
        if (!load_model(fname, model, vocab)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, fname.c_str());
            return 1;
        }
    }

    int n_past = 0;
    std::vector<float> logits_group;
    std::vector<ggml_tensor*> logits;

    std::vector<gpt_vocab::id> embd_inp = {12092, 13, 309, 717};
    size_t mem_per_token = 0;

    for (int i = 0 ; i < 10 ; i++) {
        auto res = eval(model, 1, n_past, embd_inp, mem_per_token);
        n_past += embd_inp.size();
        logits.push_back(res);
    }
}