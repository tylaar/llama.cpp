//
// Created by Yifeng Yu on 2023/3/25.
//

#include "llama.h"
#include <fmt/core.h>
#include <fstream>

bool llama_model::verify_model_magic(std::ifstream &fin) {
    uint32_t magic;
    fin.read((char *) &magic, sizeof(magic));
    if (magic != 0x67676d6c) {
        return false;
    }
    return true;
}

bool llama_model::verify_tensor_one_dimension(ggml_tensor *tensor, std::string &name, int nelements, int ne[]) {
    if (ggml_nelements(tensor) != nelements) {
        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
        return false;
    }

    if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
        fprintf(stderr,
                "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
        return false;
    }
    return true;
}

bool llama_model::verify_tensor_shape_by_column(ggml_tensor *tensor, std::string &name, int n_parts, int nelements,
                                                int ne[]) {
    if (ggml_nelements(tensor) / n_parts != nelements) {
        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
        return false;
    }

    if (tensor->ne[0] / load_ctx.n_parts != ne[0] || tensor->ne[1] != ne[1]) {
        fprintf(stderr,
                "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                __func__, name.data(), tensor->ne[0] / load_ctx.n_parts, tensor->ne[1], ne[0], ne[1]);
        return false;
    }

    return true;
}

bool
llama_model::verify_tensor_shape_by_row(ggml_tensor *tensor, std::string &name, int n_parts, int nelements, int ne[]) {
    if (ggml_nelements(tensor) / n_parts != nelements) {
        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
        return false;
    }

    if (tensor->ne[0] != ne[0] || tensor->ne[1] / load_ctx.n_parts != ne[1]) {
        fprintf(stderr,
                "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                __func__, name.data(), tensor->ne[0], tensor->ne[1] / n_parts, ne[0], ne[1]);
        return false;
    }
    return true;
}

bool llama_model::verify_tensor_shape_and_dim(ggml_tensor *tensor, std::string &name, int n_parts, int n_dims, int nelements,
                                         int ne[]) {
    if (n_dims == 1) {
        if (!verify_tensor_one_dimension(tensor, name, nelements, ne)) {
            return false;
        }
    } else {
        if (load_ctx.is_column_split_type() && !verify_tensor_shape_by_column(tensor, name, n_parts, nelements, ne)) {
            return false;
        } else if (!load_ctx.is_column_split_type() &&
                   !verify_tensor_shape_by_row(tensor, name, n_parts, nelements, ne)) {
            return false;
        }
    }
    return true;
}

void llama_model::load_model_hyper_params(std::ifstream &fin, int n_ctx) {
    // load hparams
    int n_ff = 0;
    int n_parts = 0;

    fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
    //fin.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
    fin.read((char *) &hparams.n_embd, sizeof(hparams.n_embd));
    fin.read((char *) &hparams.n_mult, sizeof(hparams.n_mult));
    fin.read((char *) &hparams.n_head, sizeof(hparams.n_head));
    fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
    fin.read((char *) &hparams.n_rot, sizeof(hparams.n_rot));
    fin.read((char *) &hparams.f16, sizeof(hparams.f16));

    hparams.n_ctx = n_ctx;

    n_ff = ((2 * (4 * hparams.n_embd) / 3 + hparams.n_mult - 1) / hparams.n_mult) * hparams.n_mult;
    n_parts = LLAMA_N_PARTS.at(hparams.n_embd);

    fprintf(stderr, "%s: n_vocab = %d\n", __func__, hparams.n_vocab);
    fprintf(stderr, "%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
    fprintf(stderr, "%s: n_embd  = %d\n", __func__, hparams.n_embd);
    fprintf(stderr, "%s: n_mult  = %d\n", __func__, hparams.n_mult);
    fprintf(stderr, "%s: n_head  = %d\n", __func__, hparams.n_head);
    fprintf(stderr, "%s: n_layer = %d\n", __func__, hparams.n_layer);
    fprintf(stderr, "%s: n_rot   = %d\n", __func__, hparams.n_rot);
    fprintf(stderr, "%s: f16     = %d\n", __func__, hparams.f16);
    fprintf(stderr, "%s: n_ff    = %d\n", __func__, n_ff);
    fprintf(stderr, "%s: n_parts = %d\n", __func__, n_parts);

    this->load_ctx.n_parts = n_parts;
    this->load_ctx.n_ff = n_ff;
}

void llama_model::load_model_vocab(std::ifstream &fin, gpt_vocab &vocab) {
    std::string word;
    for (int i = 0; i < this->hparams.n_vocab; i++) {
        uint32_t len;
        fin.read((char *) &len, sizeof(len));

        word.resize(len);
        fin.read((char *) word.data(), len);

        vocab.token_to_id[word] = i;
        vocab.id_to_token[i] = word;
    }
}

void llama_model::determine_ggml_wtype() {
    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    load_ctx.wtype = GGML_TYPE_COUNT;
    switch (hparams.f16) {
        case 0:
            load_ctx.wtype = GGML_TYPE_F32;
            break;
        case 1:
            load_ctx.wtype = GGML_TYPE_F16;
            break;
        case 2:
            load_ctx.wtype = GGML_TYPE_Q4_0;
            break;
        case 3:
            load_ctx.wtype = GGML_TYPE_Q4_1;
            break;
        default: {
        }
    }
}

bool llama_model::build_model_ctx() {
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

bool llama_model::load_model(const std::string &fname, gpt_vocab &vocab, int n_ctx) {
    fprintf(stderr, "%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    std::vector<char> f_buf(1024 * 1024);

    auto fin = std::ifstream(fname, std::ios::binary);
    fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    if (!verify_model_magic(fin)) {
        fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
        return false;
    }

    load_model_hyper_params(fin, n_ctx);
    load_model_vocab(fin, vocab);

    determine_ggml_wtype();

    if (!build_model_ctx()) {
        fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                __func__, fname.c_str(), this->hparams.f16);
        return false;
    }

    // prepare memory for the weights
    {
        const auto &hparams = this->hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        this->layers.resize(n_layer);

        this->tok_embeddings = ggml_new_tensor_2d(ctx, load_ctx.wtype, n_embd, n_vocab);

        this->norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        this->output = ggml_new_tensor_2d(ctx, load_ctx.wtype, n_embd, n_vocab);

        // map by name
        this->tensors["tok_embeddings.weight"] = this->tok_embeddings;

        this->tensors["norm.weight"] = this->norm;
        this->tensors["output.weight"] = this->output;

        for (int i = 0; i < n_layer; ++i) {
            auto &layer = this->layers[i];

            layer.attention_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.wq = ggml_new_tensor_2d(ctx, load_ctx.wtype, n_embd, n_embd);
            layer.wk = ggml_new_tensor_2d(ctx, load_ctx.wtype, n_embd, n_embd);
            layer.wv = ggml_new_tensor_2d(ctx, load_ctx.wtype, n_embd, n_embd);
            layer.wo = ggml_new_tensor_2d(ctx, load_ctx.wtype, n_embd, n_embd);

            layer.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.w1 = ggml_new_tensor_2d(ctx, load_ctx.wtype, n_embd, load_ctx.n_ff);
            layer.w2 = ggml_new_tensor_2d(ctx, load_ctx.wtype, load_ctx.n_ff, n_embd);
            layer.w3 = ggml_new_tensor_2d(ctx, load_ctx.wtype, n_embd, load_ctx.n_ff);

            // map by name
            this->tensors["layers." + std::to_string(i) + ".attention_norm.weight"] = layer.attention_norm;

            this->tensors["layers." + std::to_string(i) + ".attention.wq.weight"] = layer.wq;
            this->tensors["layers." + std::to_string(i) + ".attention.wk.weight"] = layer.wk;
            this->tensors["layers." + std::to_string(i) + ".attention.wv.weight"] = layer.wv;
            this->tensors["layers." + std::to_string(i) + ".attention.wo.weight"] = layer.wo;

            this->tensors["layers." + std::to_string(i) + ".ffn_norm.weight"] = layer.ffn_norm;

            this->tensors["layers." + std::to_string(i) + ".feed_forward.w1.weight"] = layer.w1;
            this->tensors["layers." + std::to_string(i) + ".feed_forward.w2.weight"] = layer.w2;
            this->tensors["layers." + std::to_string(i) + ".feed_forward.w3.weight"] = layer.w3;
        }
    }

    // key + value memory
    {
        const auto &hparams = this->hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;

        const int n_mem = n_layer * n_ctx;
        const int n_elements = n_embd * n_mem;

        this->memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        this->memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);

        const size_t memory_size = ggml_nbytes(this->memory_k) + ggml_nbytes(this->memory_v);

        fprintf(stderr, "%s: memory_size = %8.2f MB, n_mem = %d\n", __func__, memory_size / 1024.0 / 1024.0, n_mem);
    }

    const size_t file_offset = fin.tellg();

    fin.close();

    std::vector<uint8_t> tmp;

    for (int i = 0; i < load_ctx.n_parts; ++i) {
        const int part_id = i;
        //const int part_id = n_parts - i - 1;

        std::string fname_part = fname;
        if (i > 0) {
            fname_part += "." + std::to_string(i);
        }

        fprintf(stderr, "%s: loading model part %d/%d from '%s'\n", __func__, i + 1, load_ctx.n_parts,
                fname_part.c_str());

        fin = std::ifstream(fname_part, std::ios::binary);
        fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());
        fin.seekg(file_offset);

        // load weights
        {
            int n_tensors = 0;
            size_t total_size = 0;

            fprintf(stderr, "%s: ", __func__);

            while (true) {
                auto partial_size = load_model_tensor(fin, part_id);
                if (partial_size == -1) {
                    break;
                } else {
                    total_size += partial_size;
                }
                //
                if (++n_tensors % 8 == 0) {
                    fprintf(stderr, ".");
                    fflush(stderr);
                }
            }

            fprintf(stderr, " done\n");
        }

        fin.close();
    }
    return true;
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
bool llama_model::eval_model(
        //const llama_model* model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> &embd_inp,
        std::vector<float> &embd_w,
        size_t &mem_per_token) {
    const int N = embd_inp.size();

    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx = hparams.n_ctx;
    const int n_head = hparams.n_head;
    const int n_vocab = hparams.n_vocab;
    const int n_rot = hparams.n_embd / hparams.n_head;

    const int d_key = n_embd / n_head;

    // TODO: check if this size scales with n_ctx linearly and remove constant. somehow I feel it wasn't the case
    // static size_t buf_size = hparams.n_ctx*1024*1024;
    static size_t buf_size = 512u * 1024 * 1024;
    static void *buf = malloc(buf_size);

    if (mem_per_token > 0 && mem_per_token * N > buf_size) {
        const size_t buf_size_new = 1.1 * (mem_per_token * N); // add 10% to account for ggml object overhead
        //fprintf(stderr, "\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

        // reallocate
        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr) {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return false;
        }
    }

    struct ggml_init_params params = {
            /*.mem_size   =*/ buf_size,
            /*.mem_buffer =*/ buf,
    };

    struct ggml_context *ctx0 = ggml_init(params);
    ggml_cgraph gf = {};
    gf.n_threads = n_threads;

    struct ggml_tensor *embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, embd_inp.data(), N * ggml_element_size(embd));

    struct ggml_tensor *inpL = ggml_get_rows(ctx0, tok_embeddings, embd);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor *inpSA = inpL;

        struct ggml_tensor *cur;
        // norm

        cur = eval_norm(ctx0, inpL, layers[il].attention_norm);
        cur = eval_self_attention(&gf, ctx0, cur, il, n_past, N);

        struct ggml_tensor *inpFF = ggml_add(ctx0, cur, inpSA);

        // feed-forward network
        {
            // norm
            cur = eval_norm(ctx0, inpFF, layers[il].ffn_norm);

            struct ggml_tensor *tmp = ggml_mul_mat(ctx0, layers[il].w3,
                                                   cur);

            cur = ggml_mul_mat(ctx0, layers[il].w1, cur);

            // SILU activation
            cur = ggml_silu(ctx0, cur);
            cur = ggml_mul(ctx0, cur, tmp);
            cur = ggml_mul_mat(ctx0, layers[il].w2, cur);
        }

        cur = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        inpL = cur;
    }

    // norm

    inpL = eval_norm(ctx0, inpL, norm);

    // lm_head
    {
        inpL = ggml_mul_mat(ctx0, output, inpL);
    }

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

    // return result for just the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab * (N - 1)), sizeof(float) * n_vocab);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0) / N;
    }
    //fprintf(stderr, "used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return true;
}

ggml_tensor *
llama_model::eval_self_attention(ggml_cgraph *gf, ggml_context *ctx0, ggml_tensor *cur, int il, int n_past, int N) {
    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx = hparams.n_ctx;
    const int n_head = hparams.n_head;
    const int n_vocab = hparams.n_vocab;
    const int n_rot = hparams.n_embd / hparams.n_head;

    // self-attention

    struct ggml_tensor *Qcur = ggml_mul_mat(ctx0, layers[il].wq, cur);
    struct ggml_tensor *Kcur = ggml_mul_mat(ctx0, layers[il].wk, cur);
    struct ggml_tensor *Vcur = ggml_mul_mat(ctx0, layers[il].wv, cur);

    // store key and value to memory
    if (N >= 1) {
        struct ggml_tensor *k = ggml_view_1d(ctx0, memory_k, N * n_embd,
                                             (ggml_element_size(memory_k) * n_embd) * (il * n_ctx + n_past));
        struct ggml_tensor *v = ggml_view_1d(ctx0, memory_v, N * n_embd,
                                             (ggml_element_size(memory_v) * n_embd) * (il * n_ctx + n_past));

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
                                                   ggml_view_1d(ctx0, memory_k, (n_past + N) * n_embd,
                                                                il * n_ctx * ggml_element_size(memory_k) * n_embd),
                                                   n_embd / n_head, n_head, n_past + N),
                                   n_past, n_rot, 1),
                         0, 2, 1, 3);

    // K * Q
    struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);

    // KQ_scaled = KQ / sqrt(n_embd/n_head)
    struct ggml_tensor *KQ_scaled =
            ggml_scale(ctx0,
                       KQ,
                       ggml_new_f32(ctx0, 1.0f / sqrt(float(n_embd) / n_head))
            );

    // KQ_masked = mask_past(KQ_scaled)
    struct ggml_tensor *KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

    // KQ = soft_max(KQ_masked)
    struct ggml_tensor *KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

    // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
    struct ggml_tensor *V_trans =
            ggml_permute(ctx0,
                         ggml_reshape_3d(ctx0,
                                         ggml_view_1d(ctx0, memory_v, (n_past + N) * n_embd,
                                                      il * n_ctx * ggml_element_size(memory_v) * n_embd),
                                         n_embd / n_head, n_head, n_past + N),
                         1, 2, 0, 3);

    // KQV = transpose(V) * KQ_soft_max
    struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

    // KQV_merged = KQV.permute(0, 2, 1, 3)
    struct ggml_tensor *KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

    // cur = KQV_merged.contiguous().view(n_embd, N)
    cur = ggml_cpy(ctx0,
                   KQV_merged,
                   ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

    // projection (no bias)
    cur = ggml_mul_mat(ctx0, layers[il].wo, cur);
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