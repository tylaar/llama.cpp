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

const std::string model_prefix = "gpt_neox.";
const std::string layer_prefix = "gpt_neox.layers.";
// default hparams (GPT-J 6B)
struct pythia_hparams {
    int32_t n_vocab = 50304;
    int32_t n_ctx   = 2048;
    int32_t n_embd  = 512;
    int32_t n_head  = 8;
    int32_t n_layer = 6;
    int32_t n_rot   = 8;
    int32_t f16     = 1;
};

struct pythia_layer {
    // normalization
    struct ggml_tensor * ln_input_norm_w; // input_norm_weight
    struct ggml_tensor * ln_input_norm_b; // input_norm_bias

    struct ggml_tensor * ln_post_norm_w; // post_norm_weight
    struct ggml_tensor * ln_post_norm_b; // post_norm_bias

    // attention
    struct ggml_tensor * c_attn_b; // attn_bias_mask
    struct ggml_tensor * c_attn_k_v_w; // attn_unit_k_v_w
    struct ggml_tensor * c_attn_k_v_b; // attn_unit_k_v_b

    struct ggml_tensor * c_attn_dense_w; // attn_dense_weight
    struct ggml_tensor * c_attn_dense_b; // attn_dense_bias

    // ff
    struct ggml_tensor * c_mlp_h_to_4h_w;
    struct ggml_tensor * c_mlp_h_to_4h_b;

    struct ggml_tensor * c_mlp_4h_to_h_w;
    struct ggml_tensor * c_mlp_4h_to_h_b;

    // rot inv.feq
    struct ggml_tensor * c_rot_inv_freq; // rot_freq
};

struct pythia_model {
    pythia_hparams hparams;

    struct ggml_tensor * embed_in_wte; // position embedding
    struct ggml_tensor * embed_out_wte;

    // normalization
    struct ggml_tensor * final_layer_norm_w; // final_layer_w
    struct ggml_tensor * final_layer_norm_b; // final_layer_b

    std::vector<pythia_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// load the model's weights from a file
bool pythia_model_load(const std::string & fname, pythia_model & model, gpt_vocab & vocab) {
    printf("%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

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

    // load hparams
    {
        auto & hparams = model.hparams;

        fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        fin.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fin.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fin.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *) &hparams.n_rot,   sizeof(hparams.n_rot));
        fin.read((char *) &hparams.f16,     sizeof(hparams.f16));

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: n_rot   = %d\n", __func__, hparams.n_rot);
        printf("%s: f16     = %d\n", __func__, hparams.f16);
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        fin.read((char *) &n_vocab, sizeof(n_vocab));

        if (n_vocab != model.hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
            return false;
        }

        std::string word;
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            word.resize(len);
            fin.read((char *) word.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
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

    const ggml_type wtype2 = GGML_TYPE_F32;

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;
        const int n_rot = hparams.n_rot;

        ctx_size += n_embd*n_vocab*ggml_type_sizef(wtype); // embed_in_wte
        ctx_size += n_embd*n_vocab* ggml_type_sizef(wtype); // embed_out_wte

        //ctx_size += n_embd*n_vocab*ggml_type_sizef(wtype);         // final_layer_norm_w
        //ctx_size +=        n_vocab*ggml_type_sizef(GGML_TYPE_F32); // final_layer_norm_b
        {
            ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_input_norm_w
            ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_input_norm_b

            ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_post_norm_w
            ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_post_norm_b

            ctx_size += n_layer*(4*n_embd*4*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // attn_b
            ctx_size += n_layer*(n_rot*ggml_type_sizef(GGML_TYPE_F32)); // rot_inv_freq

            ctx_size += n_layer*(3*n_embd*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // attn_k_v_w;
            ctx_size += n_layer*(3*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // attn_k_v_b;

            // ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // c_attn_dense_w

            ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // attn_dense_w
            ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // attn_dense_b

            ctx_size += n_layer*(4*n_embd*n_embd*ggml_type_sizef(wtype));         // c_mlp_h_to_4h_w
            ctx_size += n_layer*(       4*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_mlp_h_to_4h_b

            ctx_size += n_layer*(4*n_embd*n_embd*ggml_type_sizef(wtype));         // c_mlp_4h_to_h_w
            ctx_size += n_layer*(         n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_mlp_4h_to_h_b

            ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_post_norm_g
            ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_post_norm_b

            ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_k TODO for caching??
            ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_v TODO for caching??
        }

        ctx_size += n_embd* ggml_type_sizef(GGML_TYPE_F32); // output_final_layer_norm_w;
        ctx_size += n_embd* ggml_type_sizef(GGML_TYPE_F32); // output_final_layer_norm_b;
        ctx_size += (5 + 10*n_layer)*256*2; // object overhead

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }


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
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;
        const int n_rot = hparams.n_rot;

        model.layers.resize(n_layer);

        model.embed_in_wte    = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);
        model.embed_out_wte = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);

/*
        model.ln_f_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.ln_f_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
*/

        model.final_layer_norm_w  = ggml_new_tensor_1d(ctx, wtype, n_embd);
        model.final_layer_norm_b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        // map by name
        model.tensors[model_prefix + "embed_in.weight"] = model.embed_in_wte;

        // TODO: this naming shall be a gpt_neox bug.
        model.tensors["embed_out.weight"] = model.embed_out_wte;

        model.tensors[model_prefix + "final_layer_norm.weight"] = model.final_layer_norm_w;
        model.tensors[model_prefix + "final_layer_norm.bias"]   = model.final_layer_norm_b;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            layer.ln_input_norm_w          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_input_norm_b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.ln_post_norm_w          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_post_norm_b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            //layer.c_attn_q_proj_w = ggml_new_tensor_2d(ctx, wtype,           n_embd,   n_embd);
            //layer.c_attn_k_proj_w = ggml_new_tensor_2d(ctx, wtype,           n_embd,   n_embd);
            //layer.c_attn_v_proj_w = ggml_new_tensor_2d(ctx, wtype,           n_embd,   n_embd);
            layer.c_attn_b = ggml_new_tensor_2d(ctx, wtype, 4*n_embd, 4*n_embd);
            layer.c_rot_inv_freq = ggml_new_tensor_1d(ctx, wtype, n_rot);

            layer.c_attn_k_v_w = ggml_new_tensor_2d(ctx, wtype, n_embd, 3*n_embd);
            layer.c_attn_k_v_b = ggml_new_tensor_1d(ctx, wtype, 3*n_embd);

            layer.c_attn_dense_w   = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.c_attn_dense_b   = ggml_new_tensor_1d(ctx, wtype, n_embd);

            layer.c_mlp_h_to_4h_w      = ggml_new_tensor_2d(ctx, wtype, n_embd, 4 * n_embd);
            layer.c_mlp_h_to_4h_b      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * n_embd);

            layer.c_mlp_4h_to_h_w    = ggml_new_tensor_2d(ctx, wtype, 4 * n_embd, n_embd);
            layer.c_mlp_4h_to_h_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            // map by name
            model.tensors[layer_prefix + std::to_string(i) + ".input_layernorm.weight"]          = layer.ln_input_norm_w;
            model.tensors[layer_prefix + std::to_string(i) + ".input_layernorm.bias"]            = layer.ln_input_norm_b;
            model.tensors[layer_prefix + std::to_string(i) + ".post_attention_layernorm.weight"]          = layer.ln_post_norm_w;
            model.tensors[layer_prefix + std::to_string(i) + ".post_attention_layernorm.bias"]            = layer.ln_post_norm_b;

            model.tensors[layer_prefix + std::to_string(i) + ".attention.bias"]          = layer.c_attn_b;
            model.tensors[layer_prefix + std::to_string(i) + ".attention.rotary_emb.inv_freq"]    = layer.c_rot_inv_freq;
            model.tensors[layer_prefix + std::to_string(i) + ".attention.query_key_value.weight"]      = layer.c_attn_k_v_w;
            model.tensors[layer_prefix + std::to_string(i) + ".attention.query_key_value.bias"]        = layer.c_attn_k_v_b;

            model.tensors[layer_prefix + std::to_string(i) + ".attention.dense.weight"] = layer.c_attn_dense_w;
            model.tensors[layer_prefix + std::to_string(i) + ".attention.dense.bias"] = layer.c_attn_dense_b;

            model.tensors[layer_prefix + std::to_string(i) + ".mlp.dense_h_to_4h.weight"]     = layer.c_mlp_h_to_4h_w;
            model.tensors[layer_prefix + std::to_string(i) + ".mlp.dense_h_to_4h.bias"]       = layer.c_mlp_h_to_4h_b;

            model.tensors[layer_prefix + std::to_string(i) + ".mlp.dense_4h_to_h.weight"]    = layer.c_mlp_4h_to_h_w;
            model.tensors[layer_prefix + std::to_string(i) + ".mlp.dense_4h_to_h.bias"]      = layer.c_mlp_4h_to_h_b;
        }
    }

    // key + value memory
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

    // load weights
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

            if (0) {
                static const char * ftype_str[] = { "f32", "f16", "q4_0", "q4_1", };
                printf("%24s - [%5lld, %5lld], type = %6s, %6.2f MB, %9zu bytes\n", name.data(), ne[0], ne[1], ftype_str[ftype], ggml_nbytes(tensor)/1024.0/1024.0, ggml_nbytes(tensor));
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
bool pythia_eval(
        const pythia_model & model,
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
    struct ggml_cgraph gf = { .n_threads = n_threads };

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, embd_inp.data(), N*ggml_element_size(embd));

    // embed_in_wte
    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.embed_in_wte, embd);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur;

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

        struct ggml_tensor * inpSA = cur;

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_rope(
                    ctx0,
                    ggml_reshape_3d(ctx0,
                                    ggml_mul_mat(ctx0, model.layers[il].c_attn_k_v_w, cur),
                                    n_embd/n_head, n_head, N),
                    n_past, n_rot, 0);
            struct ggml_tensor * Kcur = ggml_rope(
                    ctx0,
                    ggml_reshape_3d(ctx0,
                                    ggml_mul_mat(ctx0, model.layers[il].c_attn_k_v_w, cur),
                                    n_embd/n_head, n_head, N),
                    n_past, n_rot, 0);

            // store key and value to memory
            {
                struct ggml_tensor * Vcur = ggml_transpose(ctx0, ggml_mul_mat(ctx0, model.layers[il].c_attn_k_v_w, cur));

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

            // projection (no bias)
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].c_attn_dense_w,
                    cur);
        }

        struct ggml_tensor * inpFF = cur;

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

        // self-attention + FF
        cur  = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        inpL = ggml_add(ctx0, cur, inpL);
    }

    // norm
    {
        inpL = ggml_norm(ctx0, inpL);

        /*// inpL = ln_f_g*inpL + ln_f_b
        inpL = ggml_add(ctx0,
                ggml_mul(ctx0,
                    ggml_repeat(ctx0, model.ln_f_g, inpL),
                    inpL),
                ggml_repeat(ctx0, model.ln_f_b, inpL));*/
    }

    // lm_head
    {
        inpL = ggml_mul_mat(ctx0, model.final_layer_norm_w, inpL);

        inpL = ggml_add(ctx0,
                ggml_repeat(ctx0, model.final_layer_norm_b, inpL),
                inpL);
    }

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

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

int main(int argc, char ** argv) {
    const int64_t t_main_start_us = ggml_time_us();

    gpt_params params;
    params.model = "/Users/yifengyu/hack/models/pythia-70m/ggml-model.bin";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    printf("%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.prompt.empty()) {
        if( !isatty(STDIN_FILENO) ){
            std::string line;
            while( std::getline(std::cin, line) ){
                params.prompt = params.prompt + "\n" + line;
            }
        } else {
            params.prompt = gpt_random_prompt(rng);
        }
    }

    int64_t t_load_us = 0;

    gpt_vocab vocab;
    pythia_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!pythia_model_load(params.model, model, vocab)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    int n_past = 0;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());

    printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
    printf("\n");

    std::vector<gpt_vocab::id> embd;

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    pythia_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);

    for (int i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!pythia_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) {
                printf("Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if (i >= embd_inp.size()) {
            // sample next token
            const int   top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            {
                const int64_t t_start_sample_us = ggml_time_us();

                id = gpt_sample_top_k_top_p(vocab, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);
        } else {
            // if here, it means we are still processing the input prompt
            for (int k = i; k < embd_inp.size(); k++) {
                embd.push_back(embd_inp[k]);
                if (embd.size() > params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        // display text
        for (auto id : embd) {
            printf("%s", vocab.id_to_token[id].c_str());
        }
        fflush(stdout);

        // end of text token
        if (embd.back() == 50256) {
            break;
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}
