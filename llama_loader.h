//
// Created by Yifeng Yu on 2023/4/5.
//

#ifndef LLAMA_CPP_LLAMA_LOADER_H
#define LLAMA_CPP_LLAMA_LOADER_H

#include "llama.h"

class llama_loader {
private:
    llama_model& model;
    llama_vocab& vocab;
    std::ifstream& fin;
    std::string const& fname;
    ggml_type wtype;
    ggml_type vtype;
    char* mm_addr;

public:
    llama_loader(llama_model& m, llama_vocab& v, std::ifstream& fin, std::string const& name): model(m), vocab(v), fin(fin), fname(name) {}
    bool verify_model_magic();
    int load_model_hyper_params(int n_ctx, int n_parts);
    size_t calculate_ctx_size();
    void print_memory_loaded(ggml_type memory_type, size_t ctx_size);
    void load_model_vocab();
    void prepare_layer_memory(int n_ff);
    void determine_ggml_type();
    void mmap_memory();
    size_t load_layer_weight();
};


#endif //LLAMA_CPP_LLAMA_LOADER_H
