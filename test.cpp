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
    struct ggml_tensor * c_attn_k_v_w;
    struct ggml_tensor * c_attn_k_v_b;
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

        model.embed_in_wte = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);
        model.embed_out_wte = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);
        model.tensors["embd_in"] = model.embed_in_wte;
        model.tensors["embd_out"] = model.embed_out_wte;
        for (int i = 0 ; i < n_layer; i++) {
            auto & layer = model.layers[i];

            layer.c_attn_k_v_w = ggml_new_tensor_2d(ctx, wtype, n_embd, 3*n_embd);
            layer.c_attn_k_v_b = ggml_new_tensor_1d(ctx, wtype, 3*n_embd);
            model.tensors["c_attn_k_v_w"] = layer.c_attn_k_v_w;
            model.tensors["c_attn_k_v_b"] = layer.c_attn_k_v_b;
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
                                         __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
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

    const int d_key = n_embd/n_head;

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

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur = inpL;

/*
        // norm
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
            auto qkv = ggml_mul_mat(ctx0, model.layers[il].c_attn_k_v_w, inpSA);
            auto qkv_b = ggml_repeat(ctx0, model.layers[il].c_attn_k_v_b, qkv);
            qkv = ggml_add(ctx0,  qkv, qkv_b);

            // TODO: directly slicing, ugly, and problematic.
            auto new_qkv = ggml_reshape_3d(ctx0, qkv, n_embd*3/n_head, n_head, N);
            // auto sum = ggml_sum(ctx0, new_qkv);
            auto jump_type_size = ggml_element_size(new_qkv);
            auto jump_unit_size = new_qkv->ne[0] * new_qkv->ne[1] * new_qkv->ne[2] / 3;
            auto offset_unit = jump_type_size * jump_unit_size;
            std::cout << "jump unit_size:" << jump_unit_size << " and jump type size: " << jump_type_size << " and offset unit: "<< offset_unit << std::endl;
            auto q = ggml_view_3d(ctx0, new_qkv,
                                  n_embd/n_head, n_head, N,
                                  n_ctx*(new_qkv->ne[0])/3, n_ctx*(new_qkv->ne[0])/3*n_head,
                                  0);
            auto k = ggml_view_3d(ctx0, new_qkv,
                                  n_embd/n_head, n_head, N,
                                  n_ctx*(new_qkv->ne[0])/3, n_ctx*(new_qkv->ne[0])/3*n_head,
                                  1 * offset_unit);
            auto v = ggml_view_3d(ctx0, new_qkv,
                                  n_embd/n_head, n_head, N,
                                  n_ctx*(new_qkv->ne[0])/3, n_ctx*(new_qkv->ne[0])/3*n_head,
                                  2 * offset_unit);
            std::cout << "current q nelements:" << ggml_nelements(q) <<  " k nelem: " << ggml_nelements(k) << " v nelems:" << ggml_nelements(v) << std::endl;
            assert(ggml_nelements(new_qkv) != (ggml_nelements(q) + ggml_nelements(k) + ggml_element_size(v)));

            struct ggml_tensor * Qcur = q;
            struct ggml_tensor * Kcur = k;

            // store key and value to memory
            {
                struct ggml_tensor * Vcur = ggml_transpose(ctx0, v);

                struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_2d(ctx0, model.memory_v, N, n_embd,
                                                      (   n_ctx)*ggml_element_size(model.memory_v),
                                                      (il*n_ctx)*ggml_element_size(model.memory_v)*n_embd + n_past*ggml_element_size(model.memory_v));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            struct ggml_tensor * Q =
                    ggml_permute(ctx0,
                                 Qcur,
                                 0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            struct ggml_tensor * K =
                    ggml_permute(ctx0,
                                 ggml_reshape_3d(ctx0,
                                                 ggml_view_1d(ctx0, model.memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_k)*n_embd),
                                                 n_embd/n_head, n_head, n_past + N),
                                 0, 2, 1, 3);

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor * KQ_scaled =
                    ggml_scale(ctx0,
                               KQ,
                               ggml_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
                    );

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_tensor * V =
                    ggml_view_3d(ctx0, model.memory_v,
                                 n_past + N, n_embd/n_head, n_head,
                                 n_ctx*ggml_element_size(model.memory_v),
                                 n_ctx*ggml_element_size(model.memory_v)*n_embd/n_head,
                                 il*n_ctx*ggml_element_size(model.memory_v)*n_embd);

            // KQV = transpose(V) * KQ_soft_max
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                           KQV_merged,
                           ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
        }

        struct ggml_tensor * inpFF = cur;

/*
        // feed-forward network
        // this is independent of the self-attention result, so it could be done in parallel to the self-attention
        {
            // note here we pass inpSA instead of cur
            cur = ggml_mul_mat(ctx0,
                               model.layers[il].c_mlp_h_to_4h_w,
                               inpSA);

            cur = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].c_mlp_h_to_4h_b, cur),
                           cur);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // projection
            // cur = proj_w*cur + proj_b
            cur = ggml_mul_mat(ctx0,
                               model.layers[il].c_mlp_4h_to_h_w,
                               cur);

            cur = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].c_mlp_4h_to_h_b, cur),
                           cur);
        }
*/

        // self-attention + FF
        cur  = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        inpL = ggml_add(ctx0, cur, inpL);
    }

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);

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