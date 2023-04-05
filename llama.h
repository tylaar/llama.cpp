//
// Created by Yifeng Yu on 2023/3/24.
//

#ifndef LLAMA_CPP_LLAMA_H
#define LLAMA_CPP_LLAMA_H

#include "ggml.h"

#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include <queue>

#define SPLIT_TYPE_COLUMN 0
#define SPLIT_TYPE_ROW    1

#define Min(X, Y) ((Y) > (X) ? (X) : (Y))
#define Max(X, Y) ((Y) < (X) ? (X) : (Y))

#define LLAMA_FILE_VERSION 1
#define LLAMA_FILE_MAGIC 0x67676a74 // 'ggjt' in hex
#define LLAMA_FILE_MAGIC_UNVERSIONED 0x67676d6c // pre-versioned files

#define LLAMA_USE_SCRATCH
#define LLAMA_MAX_SCRATCH_BUFFERS 16

#define LLAMA_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "LLAMA_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)


static const size_t MB = 1024*1024;

typedef int llama_token;

typedef struct llama_token_data {
    llama_token id;  // token id

    float p;     // probability of the token
    float plog;  // log probability of the token

} llama_token_data;

// available llama models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_7B,
    MODEL_13B,
    MODEL_30B,
    MODEL_65B,
};


static const std::map<e_model, size_t> MEM_REQ_SCRATCH0 = {
        { MODEL_7B,    512ull*MB },
        { MODEL_13B,   512ull*MB },
        { MODEL_30B,   512ull*MB },
        { MODEL_65B,   512ull*MB },
};

static const std::map<e_model, size_t> MEM_REQ_SCRATCH1 = {
        { MODEL_7B,    512ull*MB },
        { MODEL_13B,   512ull*MB },
        { MODEL_30B,   512ull*MB },
        { MODEL_65B,   512ull*MB },
};

// 2*n_embd*n_ctx*n_layer*sizeof(float16)
static const std::map<e_model, size_t> MEM_REQ_KV_SELF = {
        { MODEL_7B,   1026ull*MB },
        { MODEL_13B,  1608ull*MB },
        { MODEL_30B,  3124ull*MB },
        { MODEL_65B,  5120ull*MB },
};

// this is mostly needed for temporary mul_mat buffers to dequantize the data
// not actually needed if BLAS is disabled
static const std::map<e_model, size_t> MEM_REQ_EVAL = {
        { MODEL_7B,   768ull*MB },
        { MODEL_13B, 1024ull*MB },
        { MODEL_30B, 1280ull*MB },
        { MODEL_65B, 1536ull*MB },
};

// determine number of model parts based on the dimension
static const std::map<int, int> LLAMA_N_PARTS = {
        { 4096, 1 },
        { 5120, 2 },
        { 6656, 4 },
        { 8192, 8 },
};

// default hparams (LLaMA 7B)
struct llama_hyper_params {
    int32_t n_vocab = 32000;
    int32_t n_ctx   = 512;   // this is provided as user input?
    int32_t n_embd  = 4096;
    int32_t n_mult  = 256;
    int32_t n_head  = 32;
    int32_t n_layer = 32;
    int32_t n_rot   = 64;
    int32_t f16     = 1;
};

// default loading parameters
class llama_load_ctx {
public:
    int n_ff;
    int n_parts;
    int split_type;
    ggml_type wtype;

    size_t determine_bpe(int32_t ftype, int ne[]);

    bool is_column_split_type() {
        return split_type == SPLIT_TYPE_COLUMN;
    }
};

struct llama_context_params {
    int n_ctx;   // text context
    int n_parts; // -1 for default
    int seed;    // RNG seed, 0 for random

    bool f16_kv;     // use fp16 for KV cache
    bool logits_all; // the llama_eval() call computes all logits, not just the last one
    bool vocab_only; // only load the vocabulary, no weights
    bool use_mlock;  // force system to keep model in RAM
    bool embedding;  // embedding mode only

    // called with a progress value between 0 and 1, pass NULL to disable
    //llama_progress_callback progress_callback;
    // context pointer passed to the progress callback
    //void * progress_callback_user_data;
};

struct llama_vocab {
    using id    = int32_t;
    using token = std::string;

    struct token_score {
        token tok;
        float score;
    };

    std::map<token, id> token_to_id;
    std::vector<token_score> id_to_token;
};

class llama_model;

class llama_layer {
public:
    // normalization
    struct ggml_tensor * attention_norm;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;

    // normalization
    struct ggml_tensor * ffn_norm;

    // ff
    struct ggml_tensor * w1;
    struct ggml_tensor * w2;
    struct ggml_tensor * w3;

};

/*
struct llama_model {
    e_model type = MODEL_UNKNOWN;

    llama_hyper_params hparams;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<llama_layer> layers;

    // context
    struct ggml_context * ctx;

    // key + value cache for the self attention
    // TODO: move to llama_state
    struct llama_kv_cache kv_self;

};

*/

class llama_kv_cache {
public:
    struct ggml_tensor * k;
    struct ggml_tensor * v;

    struct ggml_context * ctx;

    std::vector<uint8_t> buf;

    int n; // number of tokens currently in the cache
    bool kv_cache_init(
            const struct llama_hyper_params & hparams,
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


struct llama_context;

class llama_model {
public:
    // the model memory buffer
    std::vector<uint8_t> buf;

    // model memory mapped file
    void * mm_addr = NULL;
    uint64_t mm_length = 0;

    // tensors
    int n_loaded;
    std::map<std::string, struct ggml_tensor *> tensors;

    e_model type = MODEL_UNKNOWN;
    llama_hyper_params hparams;
    llama_load_ctx load_ctx;
    llama_context* lctx;

    // First go through tok embedding
    struct ggml_tensor * tok_embeddings;

    // norm the input
    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<llama_layer> layers;

    // key + value cache for the self attention
    // TODO: move to llama_state
    struct llama_kv_cache kv_self;

    //
    struct ggml_context * ctx;

    static llama_context* init_from_file(const std::string& path_model, llama_context_params& cparams);
    // load the model's weights from a file
    static bool load_model(const std::string & fname,
                    llama_context & lctx,
                    int n_ctx,
                    int n_parts,
                    ggml_type memory_type,
                    bool vocab_only);

    int eval(const llama_token *tokens, int n_tokens, int n_past, int n_threads);

    static void llama_free(struct llama_context * ctx);
private:

        /**
         * Model loading part
         */
    // checking if file magic number matched.
    static bool verify_model_magic(std::ifstream& fin, std::string const& fname);
    // load hparams for model metadata purpose.
    static int load_model_hyper_params(std::ifstream &fin, llama_context& ctx, int n_ctx, int n_parts);
    // load model's vocab
    static void load_model_vocab(std::ifstream &fin, llama_context& ctx);
    // determine ggml type based on hyperparams.
    static std::pair<ggml_type, ggml_type> determine_ggml_type(llama_context& ctx);

    // verify tensor shape and dimension.
    bool verify_tensor_shape_and_dim(ggml_tensor* tensor, std::string& name, int n_parts, int n_dims, int nelements, int ne[]);

    bool verify_tensor_one_dimension(ggml_tensor* tensor, std::string& name, int nelements, int ne[]);
    // verify tensor shape in column mode
    bool verify_tensor_shape_by_column(ggml_tensor *tensor, std::string& name, int n_parts, int nelements, int ne[]);
    // verify tensor shape in row mode
    bool verify_tensor_shape_by_row(ggml_tensor *tensor, std::string& name, int n_parts, int nelements, int ne[]);

    int load_model_tensor(std::ifstream &fin, int part_id);
    // build model ctx unit according to data type.
    bool build_model_ctx();

    void determine_ggml_file_split(std::string& name);

    bool eval_internal(const llama_token * tokens,
                       const int   n_tokens,
                       const int   n_past,
                       const int   n_threads);
    /**
     * Evaluation part
     */
     inline ggml_tensor* eval_self_attention(ggml_cgraph* gf, ggml_context *ctx0, ggml_tensor *cur, int il, int n_past, int N);
     inline ggml_tensor* eval_norm(ggml_context *ctx0, ggml_tensor* cur, ggml_tensor* norm);

};

class llama_context {
public:
    std::mt19937 rng;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;
    bool has_evaluated_once = false;

    int64_t t_sample_us = 0;
    int64_t t_eval_us   = 0;
    int64_t t_p_eval_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_eval   = 0; // number of eval calls
    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)

    llama_model model;
    llama_vocab vocab;

    size_t mem_per_token = 0;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;
    bool logits_all = false;

    // input embedding (1-dimensional array: [n_embd])
    std::vector<float> embedding;

    // memory buffers used to evaluate the model
    // TODO: move in llama_state
    std::vector<uint8_t> buf_compute;
    std::vector<uint8_t> buf_scratch[LLAMA_MAX_SCRATCH_BUFFERS];

    int    buf_last = 0;
    size_t buf_max_size[LLAMA_MAX_SCRATCH_BUFFERS] = { 0 };

    static void sample_top_k(std::vector<std::pair<float, llama_vocab::id>> & logits_id, int top_k) {
        // find the top k tokens
        std::partial_sort(
                logits_id.begin(),
                logits_id.begin() + top_k, logits_id.end(),
                [](const std::pair<float, llama_vocab::id> & a, const std::pair<float, llama_vocab::id> & b) {
                    return a.first > b.first;
                });

        logits_id.resize(top_k);
    }


    llama_vocab::id llama_sample_top_p_top_k(
            const std::vector<llama_vocab::id> & last_n_tokens,
            int top_k,
            float top_p,
            float temp,
            float repeat_penalty) {

        const int n_logits = model.hparams.n_vocab;
        const auto * plogits = logits.data() + logits.size() - n_logits;

        std::vector<std::pair<float, llama_vocab::id>> logits_id;
        logits_id.reserve(n_logits);

        {
            const float scale = 1.0f/temp;
            for (int i = 0; i < n_logits; ++i) {
                // repetition penalty from ctrl paper (https://arxiv.org/abs/1909.05858)
                // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
                if (std::find(last_n_tokens.begin(), last_n_tokens.end(), i) != last_n_tokens.end()) {
                    // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if (plogits[i] < 0.0f) {
                        logits_id.push_back(std::make_pair(plogits[i]*scale*repeat_penalty, i));
                    } else {
                        logits_id.push_back(std::make_pair(plogits[i]*scale/repeat_penalty, i));
                    }
                } else {
                    logits_id.push_back(std::make_pair(plogits[i]*scale, i));
                }
            }
        }

        sample_top_k(logits_id, top_k);

        float maxl = -std::numeric_limits<float>::infinity();
        for (const auto & kv : logits_id) {
            maxl = Max(maxl, kv.first);
        }

        // compute probs for the top k tokens
        std::vector<float> probs;
        probs.reserve(logits_id.size());

        double sum = 0.0;
        for (const auto & kv : logits_id) {
            const float p = expf(kv.first - maxl);
            probs.push_back(p);
            sum += p;
        }

        // normalize the probs
        for (auto & p : probs) {
            p /= sum;
        }

        if (top_p < 1.0) {
            double cumsum = 0.0;
            for (int i = 0; i < (int) probs.size(); i++) {
                cumsum += probs[i];
                if (cumsum >= top_p) {
                    probs.resize(i + 1);
                    logits_id.resize(i + 1);
                    break;
                }
            }

            cumsum = 1.0/cumsum;
            for (int i = 0; i < (int) probs.size(); i++) {
                probs[i] *= cumsum;
            }
        }

        //printf("\n");
        //for (int i = 0; i < (int) 10; i++) {
        //    printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
        //}
        //printf("\n\n");
        //exit(0);

        std::discrete_distribution<> dist(probs.begin(), probs.end());
        int idx = dist(rng);

        return logits_id[idx].second;
    }

    void use_buf(struct ggml_context * ctx, int i) {
#if defined(LLAMA_USE_SCRATCH)
        size_t last_size = 0;

        if (i == -1) {
            last_size = ggml_set_scratch(ctx, { 0, 0, nullptr, });
        } else {
            auto & buf = buf_scratch[i];
            last_size = ggml_set_scratch(ctx, { 0, buf.size(), buf.data(), });
        }

        if (buf_last >= 0) {
            buf_max_size[buf_last] = Max(buf_max_size[buf_last], last_size);
        }

        buf_last = i;
#else
        (void) i;
        (void) ctx;
#endif
    }

    size_t get_buf_max_mem(int i) const {
#if defined(LLAMA_USE_SCRATCH)
        return buf_max_size[i];
#else
        (void) i;
        return 0;
#endif
    }

    int llama_n_vocab() {
        return vocab.id_to_token.size();
    }

    int llama_n_ctx() {
        return model.hparams.n_ctx;
    }

    int llama_n_embd() {
        return model.hparams.n_embd;
    }

    float * llama_get_logits() {
        return logits.data();
    }

    float * llama_get_embeddings() {
        return embedding.data();
    }

    const char * llama_token_to_str(llama_token token) {
        if (token >= llama_n_vocab()) {
            return nullptr;
        }

        return vocab.id_to_token[token].tok.c_str();
    }

    llama_token llama_token_bos() {
        return 1;
    }

    llama_token llama_token_eos() {
        return 2;
    }

    void llama_print_timings() {
        const int64_t t_end_us = ggml_time_us();

        const int32_t n_sample = Max(1, n_sample);
        const int32_t n_eval   = Max(1, n_eval);
        const int32_t n_p_eval = Max(1, n_p_eval);

        fprintf(stderr, "\n");
        fprintf(stderr, "%s:        load time = %8.2f ms\n", __func__, t_load_us / 1000.0);
        fprintf(stderr, "%s:      sample time = %8.2f ms / %5d runs   (%8.2f ms per run)\n",   __func__, 1e-3 * t_sample_us, n_sample, 1e-3 * t_sample_us / n_sample);
        fprintf(stderr, "%s: prompt eval time = %8.2f ms / %5d tokens (%8.2f ms per token)\n", __func__, 1e-3 * t_p_eval_us, n_p_eval, 1e-3 * t_p_eval_us / n_p_eval);
        fprintf(stderr, "%s:        eval time = %8.2f ms / %5d runs   (%8.2f ms per run)\n",   __func__, 1e-3 * t_eval_us,   n_eval,   1e-3 * t_eval_us   / n_eval);
        fprintf(stderr, "%s:       total time = %8.2f ms\n", __func__, (t_end_us - t_start_us)/1000.0);
    }
};

static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

struct llama_sp_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

struct llama_sp_bigram {
    struct comparator {
        bool operator()(llama_sp_bigram & l, llama_sp_bigram & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llama_sp_bigram>;
    using queue = std::priority_queue<llama_sp_bigram, queue_storage, comparator>;
    llama_sp_symbol::index left;
    llama_sp_symbol::index right;
    float score;
    size_t size;
};

// original implementation:
// https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4
struct llama_tokenizer {
    llama_tokenizer(const llama_vocab & vocab): vocab_(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llama_sp_symbol sym;
            size_t char_len = Min(text.size() - offs, utf8_len(text[offs]));
            sym.text = text.c_str() + offs;
            sym.n = char_len;
            offs += char_len;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols_.emplace_back(std::move(sym));
        }

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols_.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue_.empty()) {
            auto bigram = work_queue_.top();
            work_queue_.pop();

            auto & left_sym = symbols_[bigram.left];
            auto & right_sym = symbols_[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //printf("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols_[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols_[i].next) {
            auto & symbol = symbols_[i];
            auto token = vocab_.token_to_id.find(std::string(symbol.text, symbol.n));

            if (token == vocab_.token_to_id.end()) {
                // output any symbols that did not form tokens as bytes.
                for (int j = 0; j < (int) symbol.n; ++j) {
                    llama_vocab::id token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
                    output.push_back(token_id);
                }
            } else {
                output.push_back((*token).second);
            }
        }
    }

private:
    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols_[left].text, symbols_[left].n + symbols_[right].n);
        auto token = vocab_.token_to_id.find(text);

        if (token == vocab_.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab_.id_to_token.size()) {
            return;
        }

        const auto &tok_score = vocab_.id_to_token[(*token).second];

        llama_sp_bigram bigram;
        bigram.left = left;
        bigram.right = right;
        bigram.score = tok_score.score;
        bigram.size = text.size();
        work_queue_.push(bigram);
    }

    const llama_vocab & vocab_;
    std::vector<llama_sp_symbol> symbols_;
    llama_sp_bigram::queue work_queue_;
};

#endif //LLAMA_CPP_LLAMA_H
