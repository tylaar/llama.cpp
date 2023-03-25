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

std::pair<int, int> llama_model::load_model_hyper_params(std::ifstream &fin, int n_ctx) {
    // load hparams
    int n_ff = 0;
    int n_parts = 0;

    auto &hparams = this->hparams;

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

    return {n_ff, n_parts};
}

void llama_model::load_model_vocab(std::ifstream &fin, gpt_vocab& vocab) {
    std::string word;
    for (int i = 0; i < this->hparams.n_vocab; i++) {
        uint32_t len;
        fin.read((char *) &len, sizeof(len));

        word.resize(len);
        fin.read((char *) word.data(), len);

        vocab.token_to_id[word] = i;
        vocab.id_to_token[i] = word;

        //if (i < 30000) {
        //    fprintf(stderr, "%s: vocab[%d] = '%s'\n", __func__, i, word.c_str());
        //}
    }
}

ggml_type llama_model::determine_ggml_wtype() {
    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = GGML_TYPE_COUNT;
    switch (hparams.f16) {
        case 0:
            return GGML_TYPE_F32;
        case 1:
            return GGML_TYPE_F16;
        case 2:
            return GGML_TYPE_Q4_0;
        case 3:
            return GGML_TYPE_Q4_1;
        default: {
            return wtype;
        }
    }
}
bool llama_model::build_model_ctx(ggml_type wtype, int n_ff, int n_parts) {
    //auto &ctx = this->ctx;
    size_t ctx_size = 0;
    {
        const auto &hparams = this->hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        ctx_size += n_embd * n_vocab * ggml_type_sizef(wtype); // tok_embeddings

        ctx_size += n_embd * ggml_type_sizef(GGML_TYPE_F32); // norm

        ctx_size += n_embd * n_vocab * ggml_type_sizef(wtype); // output

        ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // attention_norm

        ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // wq
        ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // wk
        ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // wv
        ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // wo

        ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ffn_norm

        ctx_size += n_layer * (n_ff * n_embd * ggml_type_sizef(wtype)); // w1
        ctx_size += n_layer * (n_ff * n_embd * ggml_type_sizef(wtype)); // w2
        ctx_size += n_layer * (n_ff * n_embd * ggml_type_sizef(wtype)); // w3

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

    auto hparams_count = load_model_hyper_params(fin, n_ctx);
    int n_ff = hparams_count.first ;
    int n_parts = hparams_count.second;

    load_model_vocab(fin, vocab);
    auto wtype = determine_ggml_wtype();

    if (!build_model_ctx(wtype, n_ff, n_parts)) {
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

        this->tok_embeddings = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);

        this->norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        this->output = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);

        // map by name
        this->tensors["tok_embeddings.weight"] = this->tok_embeddings;

        this->tensors["norm.weight"] = this->norm;
        this->tensors["output.weight"] = this->output;

        for (int i = 0; i < n_layer; ++i) {
            auto &layer = this->layers[i];

            layer.attention_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.wq = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wk = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wv = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wo = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);

            layer.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.w1 = ggml_new_tensor_2d(ctx, wtype, n_embd, n_ff);
            layer.w2 = ggml_new_tensor_2d(ctx, wtype, n_ff, n_embd);
            layer.w3 = ggml_new_tensor_2d(ctx, wtype, n_embd, n_ff);

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

    for (int i = 0; i < n_parts; ++i) {
        const int part_id = i;
        //const int part_id = n_parts - i - 1;

        std::string fname_part = fname;
        if (i > 0) {
            fname_part += "." + std::to_string(i);
        }

        fprintf(stderr, "%s: loading model part %d/%d from '%s'\n", __func__, i + 1, n_parts, fname_part.c_str());

        fin = std::ifstream(fname_part, std::ios::binary);
        fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());
        fin.seekg(file_offset);

        // load weights
        {
            int n_tensors = 0;
            size_t total_size = 0;

            fprintf(stderr, "%s: ", __func__);

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

                if (this->tensors.find(name.data()) == this->tensors.end()) {
                    fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                    return false;
                }

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
                    split_type = 0;
                } else if (name.find("layers") != std::string::npos) {
                    if (name.find("attention.wo.weight") != std::string::npos) {
                        split_type = 0;
                    } else if (name.find("feed_forward.w2.weight") != std::string::npos) {
                        split_type = 0;
                    } else {
                        split_type = 1;
                    }
                } else if (name.find("output") != std::string::npos) {
                    split_type = 1;
                }

                auto tensor = this->tensors[name.data()];

                if (n_dims == 1) {
                    if (ggml_nelements(tensor) != nelements) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                        return false;
                    }
                } else {
                    if (ggml_nelements(tensor) / n_parts != nelements) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                        return false;
                    }
                }

                if (n_dims == 1) {
                    if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                        fprintf(stderr,
                                "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                                __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                        return false;
                    }
                } else {
                    if (split_type == 0) {
                        if (tensor->ne[0] / n_parts != ne[0] || tensor->ne[1] != ne[1]) {
                            fprintf(stderr,
                                    "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                                    __func__, name.data(), tensor->ne[0] / n_parts, tensor->ne[1], ne[0], ne[1]);
                            return false;
                        }
                    } else {
                        if (tensor->ne[0] != ne[0] || tensor->ne[1] / n_parts != ne[1]) {
                            fprintf(stderr,
                                    "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                                    __func__, name.data(), tensor->ne[0], tensor->ne[1] / n_parts, ne[0], ne[1]);
                            return false;
                        }
                    }
                }

                if (0) {
                    static const char *ftype_str[] = {"f32", "f16", "q4_0", "q4_1",};
                    fprintf(stderr, "%24s - [%5d, %5d], type = %6s, split = %d\n", name.data(), ne[0], ne[1],
                            ftype_str[ftype], split_type);
                }

                size_t bpe = 0;

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
                        fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                        return false;
                    }
                };

                if (n_dims == 1 || n_parts == 1) {
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
                    if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor) / n_parts) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                                __func__, name.data(), ggml_nbytes(tensor) / n_parts, nelements * bpe);
                        return false;
                    }

                    if (split_type == 0) {
                        const int np0 = ne[0];

                        const size_t row_size =
                                (tensor->ne[0] / ggml_blck_size(tensor->type)) * ggml_type_size(tensor->type);
                        assert(row_size == tensor->nb[1]);

                        for (int i1 = 0; i1 < ne[1]; ++i1) {
                            const size_t offset_row = i1 * row_size;
                            const size_t offset = offset_row + ((part_id * np0) / ggml_blck_size(tensor->type)) *
                                                               ggml_type_size(tensor->type);
                            fin.read(reinterpret_cast<char *>(tensor->data) + offset, row_size / n_parts);
                        }
                    } else {
                        const int np1 = ne[1];

                        const size_t row_size =
                                (tensor->ne[0] / ggml_blck_size(tensor->type)) * ggml_type_size(tensor->type);

                        for (int i1 = 0; i1 < ne[1]; ++i1) {
                            const size_t offset_row = (i1 + part_id * np1) * row_size;
                            fin.read(reinterpret_cast<char *>(tensor->data) + offset_row, row_size);
                        }
                    }

                    total_size += ggml_nbytes(tensor) / n_parts;
                }

                //fprintf(stderr, "%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
                if (++n_tensors % 8 == 0) {
                    fprintf(stderr, ".");
                    fflush(stderr);
                }
            }

            fprintf(stderr, " done\n");

            fprintf(stderr, "%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0,
                    n_tensors);
        }
        fin.close();
    }
    return true;
}
