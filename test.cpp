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

struct test_hparams {
    int32_t n_vocab = 10;
    int32_t n_ctx = 10;
    int32_t n_embd = 4;
    int32_t n_head = 2;
    int32_t n_layer = 1;
    int32_t n_rot = 2;
    int32_t f16 = 0;
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
};

struct test_model {
    test_hparams hparams;
    struct ggml_tensor *embed_in_wte;
    struct ggml_tensor *embed_out_wte;

    std::vector<test_layer> layers;
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    struct ggml_context * ctx;
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

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_vocab = hparams.n_vocab;
    const int n_rot = hparams.n_rot;

    ctx_size += n_embd*n_vocab*ggml_type_sizef(wtype); // embed_in_wte
    ctx_size += n_embd*n_vocab* ggml_type_sizef(wtype); // embed_out_wte

    {
        ctx_size += n_layer*(3*n_embd*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // attn_k_v_w;
        ctx_size += n_layer*(3*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // attn_k_v_b;

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
        model.tensors["embd_in"] = model.embed_in_wte;
        model.tensors["embd_out"] = model.embed_out_wte;
        for (int i = 0 ; i < n_layer; i++) {
            auto & layer = model.layers[i];

            layer.c_attn_q_w = ggml_new_tensor_2d(ctx, wtype,n_embd, n_embd);
            layer.c_attn_q_b = ggml_new_tensor_1d(ctx, wtype,n_embd);
            model.tensors["c_attn_q_w"] = layer.c_attn_q_w;
            model.tensors["c_attn_q_b"] = layer.c_attn_q_b;

            layer.c_attn_k_w = ggml_new_tensor_2d(ctx, wtype,n_embd, n_embd);
            layer.c_attn_k_b = ggml_new_tensor_1d(ctx, wtype,n_embd);
            model.tensors["c_attn_k_w"] = layer.c_attn_k_w;
            model.tensors["c_attn_k_b"] = layer.c_attn_k_b;

            layer.c_attn_v_w = ggml_new_tensor_2d(ctx, wtype,n_embd, n_embd);
            layer.c_attn_v_b = ggml_new_tensor_1d(ctx, wtype,n_embd);
            model.tensors["c_attn_v_w"] = layer.c_attn_v_w;
            model.tensors["c_attn_v_b"] = layer.c_attn_v_b;
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

bool eval(
        const test_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
        std::vector<float>         & embd_w,
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
            return false;
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

    // embed_in_wte
    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.embed_in_wte, embd);

    ggml_tensor* qkv_debug;
    ggml_tensor* qkv_t_debug;
    ggml_tensor* qkv_t_reshaped_debug;

    ggml_tensor* q_debug;
    ggml_tensor* k_debug;
    ggml_tensor* v_debug;

    ggml_tensor* q_reshape_debug;
    ggml_tensor* k_reshape_debug;
    ggml_tensor* v_reshape_debug;

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur = inpL;

/* TODO Pending to test norm.
        {
            cur = ggml_norm(ctx0, inpL);

            // cur = ln_input_norm_w*cur + ln_input_norm_b
            cur = ggml_add(ctx0,
                           ggml_mul(ctx0,
                                    ggml_repeat(ctx0, model.layers[il].ln_input_norm_w, cur),
                                    cur),
                           ggml_repeat(ctx0, model.layers[il].ln_input_norm_b, cur));
        }

*/
        // Notice here we bypass out the cur pointer to inpSA for possible residual possibility
        struct ggml_tensor * inpSA = cur;

        // self-attention
        {
            /*
            debug_print_tensor(model.layers[il].c_attn_k_v_w);
            debug_print_tensor(model.layers[il].c_attn_k_v_b);
            auto qkv = ggml_mul_mat(ctx0,  inpSA,  model.layers[il].c_attn_k_v_w);
            auto qkv_b = ggml_repeat(ctx0, model.layers[il].c_attn_k_v_b, ggml_transpose(ctx0, qkv));
            qkv = ggml_add(ctx0,  qkv, ggml_transpose(ctx0, qkv_b));
            ggml_build_forward_expand(&gf, qkv);

            auto jump_type_size = ggml_element_size(qkv);
            // TODO: so far this is still working, but could be problematic
            auto jump_unit_size = qkv->ne[0] * qkv->ne[1] / 3;
            auto offset_unit = jump_type_size * jump_unit_size;
            std::cout << "jump unit_size:" << jump_unit_size << " and jump type size: " << jump_type_size << " and offset unit: "<< offset_unit << std::endl;

            // TODO: this reshape is causing copy
            auto qkv_t = ggml_new_tensor_2d(ctx0, qkv->type, n_embd * 3, n_embd);
            qkv_t = ggml_cpy(ctx0,
                             ggml_transpose(ctx0, qkv),
                             qkv_t);
            auto qkv_t_reshaped = ggml_reshape_3d(ctx0, qkv_t,6, 2, 4);
            //qkv_t_reshaped_debug = qkv_t_reshaped;
            //TODO: only for debugging.
            qkv_t_debug = qkv_t;
            qkv_debug = qkv;
            qkv_t_reshaped_debug = qkv_t_reshaped;

            */

            auto qw = model.layers[il].c_attn_q_w;
            auto qb = model.layers[il].c_attn_q_b;

            debug_print_tensor(qw);
            auto kw = model.layers[il].c_attn_k_w;
            auto kb = model.layers[il].c_attn_k_b;
            debug_print_tensor(kw);
            auto vw = model.layers[il].c_attn_v_w;
            auto vb = model.layers[il].c_attn_v_b;
            debug_print_tensor(vw);
            // TODO: viewing in 3d with slicing problematic.
            int q_type_size = ggml_type_sizef(qw->type);
            auto q = ggml_add(ctx0,
                              ggml_mul_mat(ctx0, qw, inpSA),
                              ggml_repeat(ctx0, qb, qw));

            auto k_type_size = ggml_type_sizef(kw->type);
            auto k = ggml_add(ctx0,
                              ggml_mul_mat(ctx0, kw, inpSA),
                              ggml_repeat(ctx0, kb, kw));

            auto v_type_size = ggml_type_sizef(vw->type);
            auto v = ggml_add(ctx0,
                              ggml_mul_mat(ctx0, vw, inpSA),
                              ggml_repeat(ctx0, vb, vw));

            q_reshape_debug = ggml_reshape_3d(ctx0, q, 2, 4, 2);
            q = ggml_permute(ctx0,
                             q_reshape_debug,
                             2, 1, 0, 3);
            k_reshape_debug = ggml_reshape_3d(ctx0, k, 2, 2, 4);
            k = ggml_permute(ctx0,
                             k_reshape_debug,
                             0, 2, 1, 3);
            v_reshape_debug = ggml_reshape_3d(ctx0, v, 2, 2, 4);
            v = ggml_permute(ctx0,
                             v_reshape_debug,
                             0, 2, 1, 3);
            q_debug = q;
            k_debug = k;
            v_debug = v;

            //auto q_permuted = ggml_permute(ctx0,q, 0, 2,1, 3);
            //auto k_permuted = ggml_permute(ctx0,k,0, 2, 1, 3);
            //auto v_permuted = ggml_permute(ctx0,v,0, 2, 1, 3);

            // TODO: directly slicing, ugly, and problematic.
            // auto new_qkv = ggml_reshape_3d(ctx0, qkv, 3*n_embd/n_head, n_head, N);
            // auto sum = ggml_sum(ctx0, new_qkv);
            //debug_print_tensor(q);
            //debug_print_tensor(k);
            //debug_print_tensor(v);

            //ggml_build_forward_expand(&gf, qkv_t);
            //ggml_build_forward_expand(&gf, qkv_t_reshaped);
            ggml_build_forward_expand(&gf, q_reshape_debug);
            ggml_build_forward_expand(&gf, k_reshape_debug);
            ggml_build_forward_expand(&gf, v_reshape_debug);

            ggml_build_forward_expand(&gf, q);
            ggml_build_forward_expand(&gf, k);
            ggml_build_forward_expand(&gf, v);
            //ggml_build_forward_expand(&gf, q_permuted);
            //ggml_build_forward_expand(&gf, k_permuted);
            //ggml_build_forward_expand(&gf, v_permuted);

            std::cout << "current q nelements:" << ggml_nelements(q) <<  " k nelem: " << ggml_nelements(k) << " v nelems:" << ggml_nelements(v) << std::endl;
        }

    }

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);
    //debug_print_tensor(gf.nodes[5]);
    //debug_print_tensor(gf.nodes[6]);
    //debug_print_tensor(gf.nodes[7]);
/*
    std::cout << "==========qkv_debug_printing=============" << std::endl;
    debug_print_tensor(qkv_debug);
    std::cout << "==========qkv_debug_printend=============" << std::endl;
    std::cout << "==========qkv_t_debug_printing=============" << std::endl;
    debug_print_tensor(qkv_t_debug);
    std::cout << "==========qkv_t_debug_printend=============" << std::endl;

    std::cout << "==========qkv_t_reshape_debug_printing=============" << std::endl;
    debug_print_tensor(qkv_t_reshaped_debug);
    std::cout << "==========qkv_t_reshape_debug_printend=============" << std::endl;


*/
    std::cout << "==========query_reshape_printing=============" << std::endl;
    debug_print_tensor(q_reshape_debug);
    std::cout << "==========query_reshape_printend=============" << std::endl;
/*
    std::cout << "==========key_reshape_printing=============" << std::endl;
    debug_print_tensor(k_reshape_debug);
    std::cout << "==========key_reshape_printend=============" << std::endl;
    std::cout << "==========value_reshape_printing=============" << std::endl;
    debug_print_tensor(v_reshape_debug);
    std::cout << "==========value_reshape_printend=============" << std::endl;
*/
    debug_print_tensor(q_debug);
    debug_print_tensor(k_debug);
    debug_print_tensor(v_debug);

    //debug_print_graph_filter_type(&gf, GGML_OP_ADD);
    debug_print_graph_filter_type(&gf, GGML_OP_RESHAPE);
    debug_print_graph_filter_type(&gf, GGML_OP_VIEW);
    //debug_print_graph_filter_type(&gf, GGML_OP_PERMUTE);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result for just the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }
    //printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return true;
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
    std::vector<float> logits;

    std::vector<gpt_vocab::id> embd_inp = {2,4,6,8};
    size_t mem_per_token = 0;

    eval(model, 1, n_past, embd_inp, logits, mem_per_token);

}