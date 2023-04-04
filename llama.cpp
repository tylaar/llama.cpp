//
// Created by Yifeng Yu on 2023/3/25.
//

#include "llama.h"
#include <fmt/core.h>
#include <fstream>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

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


static void *mmap_file(const char *fname, uint64_t *mm_length) {
#if defined(_WIN32) && !defined(_POSIX_MAPPED_FILES)
    HANDLE hFile = CreateFileA(fname,
                               GENERIC_READ,
                               FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               NULL,
                               OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL | FILE_ATTRIBUTE_NOT_CONTENT_INDEXED,
                               NULL);
    if (hFile == INVALID_HANDLE_VALUE) return 0;
    LARGE_INTEGER fileSize;
    fileSize.QuadPart = -1;
    GetFileSizeEx(hFile, &fileSize);
    int64_t length = fileSize.QuadPart;
    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(hFile);
    if (!hMapping) return 0;
    void *addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMapping);
    if (!addr) return 0;
#else
    int fd = open(fname, O_RDONLY);
    if (fd == -1) return 0;
    int64_t length = lseek(fd, 0, SEEK_END);
    void *addr = mmap(NULL, length, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) return 0;
#endif
    *mm_length = length;
    return addr;
}

static void munmap_file(void *addr, size_t length) {
#if defined(_WIN32) && !defined(_POSIX_MAPPED_FILES)
    UnmapViewOfFile(addr);
#else
    munmap(addr, length);
#endif
}


llama_context *llama_model::init_from_file(const std::string &path_model, llama_context_params &params) {
    llama_context *ctx = new llama_context;

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

void llama_model::llama_free(struct llama_context *ctx) {
    ctx->model.kv_self.kv_cache_free();

    if (ctx->model.ctx) {
        ggml_free(ctx->model.ctx);
    }

    if (ctx->model.mm_addr) {
        munmap_file(ctx->model.mm_addr, ctx->model.mm_length);
    }

    delete ctx;
}


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

    std::vector<char> f_buf(1024 * 1024);
    fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());

    fin.seekg(0, fin.end);
    const size_t file_size = fin.tellg();
    fin.seekg(0);

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic == LLAMA_FILE_MAGIC_UNVERSIONED) {
            fprintf(stderr,
                    "%s: invalid model file '%s' (too old, regenerate your model files or convert them with convert-unversioned-ggml-to-ggml.py!)\n",
                    __func__, fname.c_str());
            return false;
        }
        if (magic != LLAMA_FILE_MAGIC) {
            fprintf(stderr,
                    "%s: invalid model file (bad magic [got %#x want %#x])\n"
                    "\tyou most likely need to regenerate your ggml files\n"
                    "\tthe benefit is you'll get 10-100x faster load times\n"
                    "\tsee https://github.com/ggerganov/llama.cpp/issues/91\n"
                    "\tuse convert-pth-to-ggml.py to regenerate from original pth\n"
                    "\tuse migrate-ggml-2023-03-30-pr613.py if you deleted originals\n",
                    fname.c_str(), magic, LLAMA_FILE_MAGIC);
            return false;
        }

        uint32_t format_version;
        fin.read((char *) &format_version, sizeof(format_version));

        if (format_version != LLAMA_FILE_VERSION) {
            fprintf(stderr, "%s: invalid model file '%s' (unsupported format version)" ", expected %d)\n",
                    __func__, fname.c_str(), LLAMA_FILE_VERSION);
            return false;
        }
    }

    int n_ff = 0;

    // load hparams
    {
        auto &hparams = model.hparams;

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

        if (n_parts < 1) {
            n_parts = LLAMA_N_PARTS.at(hparams.n_embd);
        }

        // temp warning to tell the user to use "--n_parts"
        if (hparams.f16 == 4 && n_parts != 1) {
            fprintf(stderr, "%s: GPTQ model detected - are you sure n_parts should be %d? we normally expect it to be 1\n", __func__, n_parts);
            fprintf(stderr, "%s: use '--n_parts 1' if necessary\n", __func__);
        }

        if (hparams.n_layer == 32) {
            model.type = e_model::MODEL_7B;
        }

        if (hparams.n_layer == 40) {
            model.type = e_model::MODEL_13B;
        }

        if (hparams.n_layer == 60) {
            model.type = e_model::MODEL_30B;
        }

        if (hparams.n_layer == 80) {
            model.type = e_model::MODEL_65B;
        }

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
        fprintf(stderr, "%s: type    = %d\n", __func__, model.type);
    }

    // load vocab
    {
        std::string word;
        vocab.id_to_token.resize(model.hparams.n_vocab);
        std::vector<char> tmp(64);

        for (int i = 0; i < model.hparams.n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            word.resize(len);
            if (len > 0) {
                tmp.resize(len);
                fin.read(tmp.data(), len);
                word.assign(tmp.data(), len);
            } else {
                word.clear();
            }

            float score;
            fin.read((char *) &score, sizeof(score));

            vocab.token_to_id[word] = i;

            auto &tok_score = vocab.id_to_token[i];
            tok_score.tok = word;
            tok_score.score = score;
        }
    }

    if (vocab_only) {
        return true;
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    // wtype is for per-layer weights, while vtype is for other weights
    ggml_type wtype, vtype;
    switch (model.hparams.f16) {
        case 0:
            wtype = vtype = GGML_TYPE_F32;
            break;
        case 1:
            wtype = vtype = GGML_TYPE_F16;
            break;
        case 2:
            wtype = vtype = GGML_TYPE_Q4_0;
            break;
        case 3:
            wtype = vtype = GGML_TYPE_Q4_1;
            break;
        case 4:
            wtype = GGML_TYPE_Q4_1;
            vtype = GGML_TYPE_F16;
            break;
        default: {
            fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                    __func__, fname.c_str(), model.hparams.f16);
            return false;
        }
    }

    // map model into memory
    char *mm_addr = NULL;
    model.mm_addr = mmap_file(fname.c_str(), &model.mm_length);
    if (model.mm_addr == NULL) {
        fprintf(stderr, "%s: failed to mmap '%s'\n", __func__, fname.c_str());
        return false;
    }
    mm_addr = (char *) model.mm_addr;
    fprintf(stderr, "%s: ggml map size = %6.2f MB\n", __func__, model.mm_length / (1024.0 * 1024.0));

    size_t ctx_size = 0;
    {
        const auto &hparams = model.hparams;
        const int n_layer = hparams.n_layer;
        ctx_size += (5 + 10 * n_layer) * 256; // object overhead
        fprintf(stderr, "%s: ggml ctx size = %6.2f KB\n", __func__, ctx_size / 1024.0);
    }

    // print memory requirements
    {
        const size_t scale = memory_type == GGML_TYPE_F32 ? 2 : 1;

        // this is the total memory required to run the inference
        const size_t mem_required =
                ctx_size +
                model.mm_length +
                MEM_REQ_SCRATCH0.at(model.type) +
                MEM_REQ_SCRATCH1.at(model.type) +
                MEM_REQ_EVAL.at(model.type);

        // this is the memory required by one llama_state
        const size_t mem_required_state =
                scale * MEM_REQ_KV_SELF.at(model.type);

        fprintf(stderr, "%s: mem required  = %7.2f MB (+ %7.2f MB per state)\n", __func__,
                mem_required / 1024.0 / 1024.0, mem_required_state / 1024.0 / 1024.0);
    }

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

    // prepare memory for the weights
    {
        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.tok_embeddings = ggml_new_tensor_2d(model.ctx, vtype, n_embd, n_vocab);

        model.norm = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, n_embd);
        model.output = ggml_new_tensor_2d(model.ctx, vtype, n_embd, n_vocab);

        // map by name
        model.tensors["tok_embeddings.weight"] = model.tok_embeddings;

        model.tensors["norm.weight"] = model.norm;
        model.tensors["output.weight"] = model.output;

        for (int i = 0; i < n_layer; ++i) {
            auto &layer = model.layers[i];

            layer.attention_norm = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, n_embd);

            layer.wq = ggml_new_tensor_2d(model.ctx, wtype, n_embd, n_embd);
            layer.wk = ggml_new_tensor_2d(model.ctx, wtype, n_embd, n_embd);
            layer.wv = ggml_new_tensor_2d(model.ctx, wtype, n_embd, n_embd);
            layer.wo = ggml_new_tensor_2d(model.ctx, wtype, n_embd, n_embd);

            layer.ffn_norm = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, n_embd);

            layer.w1 = ggml_new_tensor_2d(model.ctx, wtype, n_embd, n_ff);
            layer.w2 = ggml_new_tensor_2d(model.ctx, wtype, n_ff, n_embd);
            layer.w3 = ggml_new_tensor_2d(model.ctx, wtype, n_embd, n_ff);

            // map by name
            model.tensors["layers." + std::to_string(i) + ".attention_norm.weight"] = layer.attention_norm;

            model.tensors["layers." + std::to_string(i) + ".attention.wq.weight"] = layer.wq;
            model.tensors["layers." + std::to_string(i) + ".attention.wk.weight"] = layer.wk;
            model.tensors["layers." + std::to_string(i) + ".attention.wv.weight"] = layer.wv;
            model.tensors["layers." + std::to_string(i) + ".attention.wo.weight"] = layer.wo;

            model.tensors["layers." + std::to_string(i) + ".ffn_norm.weight"] = layer.ffn_norm;

            model.tensors["layers." + std::to_string(i) + ".feed_forward.w1.weight"] = layer.w1;
            model.tensors["layers." + std::to_string(i) + ".feed_forward.w2.weight"] = layer.w2;
            model.tensors["layers." + std::to_string(i) + ".feed_forward.w3.weight"] = layer.w3;
        }
    }

    std::vector<uint8_t> tmp;

    fprintf(stderr, "%s: loading tensors from '%s'\n", __func__, fname.c_str());

    // load weights
    {
        size_t total_size = 0;
        model.n_loaded = 0;

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ftype), sizeof(ftype));

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = {1, 1};
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];

            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return false;
            }
            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                return false;
            }
            if (0) {
                static const char *ftype_str[] = {"f32", "f16", "q4_0", "q4_1",};
                fprintf(stderr, "%24s - [%5d, %5d], type = %6s\n", name.data(), ne[0], ne[1], ftype_str[ftype]);
            }

            switch (ftype) {
                case 0:  // f32
                case 1:  // f16
                    break;
                case 2:  // q4_0
                case 3:  // q4_1
                    assert(ne[0] % 64 == 0);
                    break;
                default:
                    fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                    return false;
            };

            // load the tensor data into memory without copying or reading it
            size_t offset = fin.tellg();
            size_t tensor_data_size = ggml_nbytes(tensor);
            offset = (offset + 31) & -32;
            tensor->data = mm_addr + offset;
            fin.seekg(offset + tensor_data_size);
            total_size += tensor_data_size;
            model.n_loaded++;

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
    }

    // loading time will be recalculate after the first eval, so
    // we take page faults deferred by mmap() into consideration
    ctx.t_load_us = ggml_time_us() - ctx.t_start_us;

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

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor *inpSA = inpL;
        struct ggml_tensor *cur;

        lctx->use_buf(ctx0, 0);

        // norm
        cur = eval_norm(ctx0, inpL, layers[il].attention_norm);

        // self-attention

        cur = eval_self_attention(&gf, ctx0, cur, il, n_past, N);


        lctx->use_buf(ctx0, 1);

        struct ggml_tensor *inpFF = ggml_add(ctx0, cur, inpSA);

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

        cur = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        inpL = cur;
    }

    lctx->use_buf(ctx0, 0);

    // used at the end to optionally extract the embeddings
    struct ggml_tensor *embeddings = NULL;

    // norm
    inpL = eval_norm(ctx0, inpL, norm);

    // lm_head
    inpL = ggml_mul_mat(ctx0, output, inpL);

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