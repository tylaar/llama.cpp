//
// Created by Yifeng Yu on 2023/4/5.
//

#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#include "llama.h"



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


bool llama_model::verify_model_magic(std::ifstream &fin, std::string const& fname) {
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

    return true;
}


int llama_model::load_model_hyper_params(std::ifstream &fin, llama_context& ctx, int n_ctx, int n_parts) {
    //auto &vocab = ctx.vocab;
    // load hparams
    int n_ff = 0;
    // load hparams
    {
        auto &hparams = ctx.model.hparams;

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
            ctx.model.type = e_model::MODEL_7B;
        }

        if (hparams.n_layer == 40) {
            ctx.model.type = e_model::MODEL_13B;
        }

        if (hparams.n_layer == 60) {
            ctx.model.type = e_model::MODEL_30B;
        }

        if (hparams.n_layer == 80) {
            ctx.model.type = e_model::MODEL_65B;
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
        fprintf(stderr, "%s: type    = %d\n", __func__, ctx.model.type);
    }
    return n_ff;
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

void llama_model::load_model_vocab(std::ifstream &fin, llama_context &ctx) {
    auto& vocab = ctx.vocab;
    // load vocab
    {
        std::string word;
        vocab.id_to_token.resize(ctx.model.hparams.n_vocab);
        std::vector<char> tmp(64);

        for (int i = 0; i < ctx.model.hparams.n_vocab; i++) {
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

}

std::pair<ggml_type, ggml_type> llama_model::determine_ggml_type(llama_context& ctx) {
    ggml_type wtype, vtype;
    switch (ctx.model.hparams.f16) {
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
            throw std::invalid_argument("F16 ggml type type no recognizable");
        }
    }
    return {wtype, vtype};
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
    if (!verify_model_magic(fin, fname)) {
        return false;
    }

    int n_ff = load_model_hyper_params(fin, ctx, n_ctx, n_parts);

    load_model_vocab(fin, ctx);

    if (vocab_only) {
        return true;
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    // wtype is for per-layer weights, while vtype is for other weights
    ggml_type wtype, vtype;
    std::pair<ggml_type, ggml_type> types;
    try {
        types = determine_ggml_type(ctx);
    } catch (std::invalid_argument e) {
        std::cerr << e.what() << std::endl;
        return false;
    }

    wtype = types.first;
    vtype = types.second;

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

