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


bool llama_loader::verify_model_magic() {
    uint32_t magic;
    fin.read((char *) &magic, sizeof(magic));
    if (magic == LLAMA_FILE_MAGIC_UNVERSIONED) {
        std::cerr << fmt::format(
                "%s: invalid model file '%s' (too old, regenerate your model files or convert them with convert-unversioned-ggml-to-ggml.py!)\n",
                __func__, fname.c_str());
        return false;
    }
    if (magic != LLAMA_FILE_MAGIC) {
        std::cerr << fmt::format(
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

void llama_loader::mmap_memory() {
    mm_addr = (char *) llama_memory_mapper::mmap_file(fname.c_str(), &model.mm_length);
    if (mm_addr == NULL) {
        throw std::invalid_argument(fmt::format("%s: failed to mmap '%s'\n", __func__, fname.c_str()));
    }
    model.mm_addr = mm_addr;
    std::cout << __func__  << fmt::format(" ggml map size = %6.2f MB\n", model.mm_length / (1024.0 * 1024.0));
}

int llama_loader::load_model_hyper_params(int n_ctx, int n_parts) {
    //auto &vocab = ctx.vocab;
    // load hparams
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
            std::cerr << fmt::format( "%s: GPTQ model detected - are you sure n_parts should be %d? we normally expect it to be 1\n", __func__, n_parts);
            std::cerr << fmt::format( "%s: use '--n_parts 1' if necessary\n", __func__);
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

        std::cerr << fmt::format( "%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        std::cerr << fmt::format( "%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        std::cerr << fmt::format( "%s: n_embd  = %d\n", __func__, hparams.n_embd);
        std::cerr << fmt::format( "%s: n_mult  = %d\n", __func__, hparams.n_mult);
        std::cerr << fmt::format( "%s: n_head  = %d\n", __func__, hparams.n_head);
        std::cerr << fmt::format( "%s: n_layer = %d\n", __func__, hparams.n_layer);
        std::cerr << fmt::format( "%s: n_rot   = %d\n", __func__, hparams.n_rot);
        std::cerr << fmt::format( "%s: f16     = %d\n", __func__, hparams.f16);
        std::cerr << fmt::format( "%s: n_ff    = %d\n", __func__, n_ff);
        std::cerr << fmt::format( "%s: n_parts = %d\n", __func__, n_parts);
        std::cerr << fmt::format( "%s: type    = %d\n", __func__, model.type);
    }
    return n_ff;
}

size_t llama_loader::calculate_ctx_size() {
    size_t ctx_size = 0;
    const auto &hparams = model.hparams;
    const int n_layer = hparams.n_layer;
    ctx_size += (5 + 10 * n_layer) * 256; // object overhead
    std::cerr << fmt::format( "%s: ggml ctx size = %6.2f KB\n", __func__, ctx_size / 1024.0);
    return ctx_size;
}


void llama_loader::print_memory_loaded(ggml_type memory_type, size_t ctx_size) {
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

    std::cerr << fmt::format( "%s: mem required  = %7.2f MB (+ %7.2f MB per state)\n", __func__,
            mem_required / 1024.0 / 1024.0, mem_required_state / 1024.0 / 1024.0);
}


void llama_loader::load_model_vocab() {
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

}

void llama_loader::determine_ggml_type() {
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
            throw std::invalid_argument("F16 ggml type type no recognizable");
        }
    }
}


void llama_loader::prepare_layer_memory(int n_ff) {
    // prepare memory for the weights

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

size_t llama_loader::load_layer_weight() {
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
            throw std::invalid_argument(fmt::format("%s: unknown tensor '%s' in model file", __func__, name.data()));
        }

        auto tensor = model.tensors[name.data()];

        if (ggml_nelements(tensor) != nelements) {
            throw std::invalid_argument(fmt::format("%s: tensor '%s' has wrong size in model file", __func__, name.data()));
        }
        if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
            throw std::invalid_argument(fmt::format("%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                                                    __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]));
        }
        if (0) {
            static const char *ftype_str[] = {"f32", "f16", "q4_0", "q4_1",};
            std::cerr << fmt::format( "%24s - [%5d, %5d], type = %6s\n", name.data(), ne[0], ne[1], ftype_str[ftype]);
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
    return total_size;
}
