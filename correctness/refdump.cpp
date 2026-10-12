// llama-refdump: the reference harness. Runs a model one token at a time (the batch-1 decode path
// a megakernel replaces) over fixed prompts and dumps what a candidate is checked against:
//   tokens.i32       prompt tokens, then greedy continuation (or the base run's tokens, teacher forced)
//   logits.f32       [n_out][n_vocab] logits for every position from --logits-from on
//   t.<name>.bin     selected intermediate tensors per position (via the scheduler's eval callback)
//   kv.bin           llama_state_seq_get_data() after the last position (KV cache contents)
//   meta.json        shapes, counts and the run's settings
// Watching tensors splits the graph at each watched node (fusions across it and CUDA graphs are lost), so a
// run with --tensors is its own configuration: compare it only with another run watching the same tensors.
// --inject perturbs one watched tensor in place (sensitivity calibration for the comparison thresholds):
//   name@pos:rel[:seed]  pos is a position or * for every position; rel is relative Gaussian noise
//   (x += rel * rms(x) * N(0,1)), or a scale error with an x suffix (0.01x: x *= 1.01).
// --replay basedir [--replay-tensors l_out]: after recording each matching watched tensor, overwrite it with the
// reference run's value (basedir/<prompt>/t.<name>.bin, a run watching the same tensors), so every layer starts
// from the reference's input and the dump measures each layer's own error rather than the accumulated drift.
//
// usage: llama-refdump -m model.gguf -o outdir [--gen N] [--tokens-from basedir] [--logits-from P]
//                      [--tensors prefix,prefix,...] [--inject name@pos:rel[:seed]] [--ctx N] [--kv]
//                      [--replay basedir [--replay-tensors l_out,...]] [--no-fa] [--prefill] prompt.txt...
// Megakernel mode (llama.cpp with megakernel.patch and GGML_CUDA_MEGAKERNEL=1): a decode step is one fused node, so the
// eval callback sees none of the watched tensors; the kernel's debug hooks (ggml-megakernel.h; the functions are found
// with dlsym) record, replace and inject instead, and the same files are written. Differences: an injected Q / K / V
// fault takes its rms per head, kqv_out takes scale faults only, and only the megakernel's watch points can be watched.
// Each prompt file is tokenized with special tokens parsed (chat templates are applied beforehand) and
// written to outdir/<file stem>/. --prefill (stock batched prefill of the prompt, then decode) logs logits from the
// last prompt token on and cannot be combined with watched tensors.
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <dlfcn.h>
#if __has_include("ggml-megakernel.h")
#include "ggml-megakernel.h"
#define REFDUMP_MEGAKERNEL 1
#endif

struct watch {
    std::vector<std::string> prefixes;
    std::string dir;
    int pos = 0;                                    // position being decoded
    std::map<std::string, FILE *> files;
    std::map<std::string, std::string> shapes;      // name -> "type:ne0xne1x..." (first seen)
    std::map<std::string, int> counts;
    std::map<std::string, std::pair<int, int>> seen;  // name -> (last position, occurrences there)
    // fault injection
    std::string inj_name; int inj_pos = -1; double inj_rel = 0; unsigned inj_seed = 1; bool inj_scale = false;
    int injected = 0;                               // positions perturbed
    // replay: after recording a matching tensor, overwrite it with the reference run's value at this position
    std::string replay_dir;
    std::vector<std::string> replay_prefixes;
    std::map<std::string, FILE *> replay_files;
    int replayed = 0;
    std::vector<char> buf;
    std::vector<float> f32;
};

static bool wanted(const watch * w, const char * name) {
    if (strchr(name, ' ')) return false;  // "<name> (view)", "(copy)", ...: ggml's derived tensors duplicate the data
    for (const auto & p : w->prefixes) {
        if (p == "*" || strncmp(name, p.c_str(), p.size()) == 0) return true;
    }
    return false;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    auto * w = (watch *) ud;
    if (ask) return wanted(w, t->name);
    if (!wanted(w, t->name)) return true;
    // some graphs reuse a name within one step (llama: "Qcur" before and after RoPE); later occurrences at the
    // same position get their own key, "<name>#<k>", so files, replay and comparisons stay aligned
    std::string key = t->name;
    {
        auto it = w->seen.find(key);
        const int occ = it != w->seen.end() && it->second.first == w->pos ? it->second.second + 1 : 0;
        w->seen[key] = {w->pos, occ};
        if (occ > 0) key += "#" + std::to_string(occ);
    }
    if (!ggml_is_contiguous(t)) {
        if (!w->shapes.count(key)) fprintf(stderr, "refdump: skipping non-contiguous %s\n", key.c_str());
        w->shapes[key] = "noncontiguous";
        return true;
    }
    const size_t nb = ggml_nbytes(t);
    const int64_t n = ggml_nelements(t);
    w->buf.resize(nb);
    ggml_backend_tensor_get(t, w->buf.data(), 0, nb);
    const char * kind;
    if (t->type == GGML_TYPE_F32) {
        kind = "f32";
        w->f32.assign((float *) w->buf.data(), (float *) w->buf.data() + n);
    } else if (t->type == GGML_TYPE_F16) {
        kind = "f32";
        w->f32.resize(n);
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) w->buf.data(), w->f32.data(), n);
    } else if (t->type == GGML_TYPE_BF16) {
        kind = "f32";
        w->f32.resize(n);
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) w->buf.data(), w->f32.data(), n);
    } else if (t->type == GGML_TYPE_I32) {
        kind = "i32";
    } else {
        if (!w->shapes.count(key)) fprintf(stderr, "refdump: skipping %s of type %s\n", key.c_str(), ggml_type_name(t->type));
        w->shapes[key] = "unsupported";
        return true;
    }
    if (!w->inj_name.empty() && w->inj_name == key && (w->inj_pos == -2 || w->pos == w->inj_pos) &&
        t->type == GGML_TYPE_F32) {
        // noise: x += rel * rms(x) * N(0,1); scale: x *= 1 + rel. Written back so every later node sees it.
        if (w->inj_scale) {
            for (auto & v : w->f32) v = (float) (v * (1.0 + w->inj_rel));
        } else {
            double ss = 0;
            for (float v : w->f32) ss += (double) v * v;
            const double rms = std::sqrt(ss / std::max<int64_t>(n, 1));
            std::mt19937 rng(w->inj_seed + 7919u * (unsigned) w->pos);
            std::normal_distribution<double> nd(0.0, 1.0);
            for (auto & v : w->f32) v = (float) (v + w->inj_rel * rms * nd(rng));
        }
        ggml_backend_tensor_set(t, w->f32.data(), 0, nb);
        w->injected++;
    }
    if (!w->files.count(key)) {
        std::ostringstream s;
        s << kind;
        for (int d = 0; d < GGML_MAX_DIMS; d++) s << (d ? "x" : ":") << t->ne[d];
        w->shapes[key] = s.str();
        w->files[key] = fopen((w->dir + "/t." + key + ".bin").c_str(), "wb");
    }
    FILE * f = w->files[key];
    if (strcmp(kind, "i32") == 0) fwrite(w->buf.data(), 1, nb, f);
    else fwrite(w->f32.data(), sizeof(float), n, f);
    w->counts[key]++;
    bool replay = false;
    for (const auto & p : w->replay_prefixes) replay |= strncmp(t->name, p.c_str(), p.size()) == 0;
    if (replay && t->type == GGML_TYPE_F32 && w->pos >= 0) {
        if (!w->replay_files.count(key)) {
            w->replay_files[key] = fopen((w->replay_dir + "/t." + key + ".bin").c_str(), "rb");
            if (!w->replay_files[key]) { fprintf(stderr, "refdump: no reference %s to replay\n", key.c_str()); exit(1); }
        }
        FILE * rf = w->replay_files[key];
        if (fseek(rf, (long) w->pos * (long) nb, SEEK_SET) != 0 || fread(w->f32.data(), sizeof(float), n, rf) != (size_t) n) {
            fprintf(stderr, "refdump: reference %s has no position %d\n", key.c_str(), w->pos); exit(1);
        }
        ggml_backend_tensor_set(t, w->f32.data(), 0, nb);
        w->replayed++;
    }
    return true;
}

static std::string read_file(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(1); }
    std::stringstream s; s << f.rdbuf(); return s.str();
}

static std::vector<int32_t> read_i32(const std::string & p) {
    std::string s = read_file(p);
    std::vector<int32_t> v(s.size() / 4);
    memcpy(v.data(), s.data(), v.size() * 4);
    return v;
}

static std::string stem(const std::string & p) {
    size_t a = p.find_last_of('/'); a = a == std::string::npos ? 0 : a + 1;
    size_t b = p.find_last_of('.'); if (b == std::string::npos || b < a) b = p.size();
    return p.substr(a, b - a);
}

static std::string json_escape(const std::string & s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if ((unsigned char) c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
    return o;
}

#ifdef REFDUMP_MEGAKERNEL
// megakernel mode: the megakernel's watch points, filtered by --tensors; per step the hooks are armed before llama_decode
// and the records written after it, in cb_eval's order (inject, record, replace)
struct mk_watch {
    std::string key; size_t off; int64_t ne0, ne1; int w, layer; bool i32;
    FILE * rf = nullptr;   // replay reference
};
struct mk_mode {
    decltype(&ggml_megakernel_debug_watch) watch_fn = nullptr;
    decltype(&ggml_megakernel_debug_set) set_fn = nullptr;
    std::vector<mk_watch> ws;
    std::vector<float> rec, rep, noise;
    ggml_megakernel_debug d {};
};

static bool mk_setup(mk_mode & m, const watch & w) {
    const char * e = getenv("GGML_CUDA_MEGAKERNEL");
    if (!e || atoi(e) == 0) return false;
    m.watch_fn = (decltype(m.watch_fn)) dlsym(RTLD_DEFAULT, "ggml_megakernel_debug_watch");
    m.set_fn = (decltype(m.set_fn)) dlsym(RTLD_DEFAULT, "ggml_megakernel_debug_set");
    if (!m.watch_fn || !m.set_fn) { fprintf(stderr, "refdump: GGML_CUDA_MEGAKERNEL set but no debug hooks in this build\n"); exit(1); }
    size_t total = 0;
    for (int wp = 0; wp <= GGML_MEGAKERNEL_WATCH_N; wp++) {
        for (int L = 0; L < (wp < GGML_MEGAKERNEL_WATCH_N ? GGML_MEGAKERNEL_LAYERS : 1); L++) {
            const char * name; mk_watch x;
            total = m.watch_fn(wp, L, &name, &x.off, &x.ne0, &x.ne1);
            x.key = wp < GGML_MEGAKERNEL_WATCH_N ? std::string(name) + "-" + std::to_string(L) : std::string(name);
            x.w = wp; x.layer = L; x.i32 = x.key.rfind("ffn_moe_topk", 0) == 0;
            if (wanted(&w, x.key.c_str())) m.ws.push_back(x);
        }
    }
    m.rec.assign(total, 0.f); m.rep.assign(total, 0.f); m.noise.assign(8192, 0.f);
    m.d.rec = m.rec.data(); m.d.rep = m.rep.data(); m.d.noise = m.noise.data();
    fprintf(stderr, "refdump: megakernel mode, %zu watch points\n", m.ws.size());
    return true;
}

// before the step at w.pos: replacements (from the reference) and the planted fault
static void mk_arm(mk_mode & m, watch & w) {
    for (auto & r : m.d.rep_mask) r = 0;
    m.d.inj_w = -1;
    for (auto & x : m.ws) {
        const int64_t n = x.ne0 * x.ne1;
        if (!w.inj_name.empty() && w.inj_name == x.key && (w.inj_pos == -2 || w.pos == w.inj_pos) && !x.i32) {
            if (x.w == GGML_MEGAKERNEL_WATCH_N) { fprintf(stderr, "refdump: cannot inject into %s\n", x.key.c_str()); exit(1); }
            m.d.inj_w = x.w; m.d.inj_layer = x.layer;
            m.d.inj_scale = w.inj_scale ? (float) w.inj_rel : 0.f;
            m.d.inj_rel = w.inj_scale ? 0.f : (float) w.inj_rel;
            std::mt19937 rng(w.inj_seed + 7919u * (unsigned) w.pos);   // as cb_eval draws it
            std::normal_distribution<double> nd(0.0, 1.0);
            for (int64_t i = 0; i < n; i++) m.noise[i] = (float) nd(rng);
            w.injected++;
        }
        bool replay = false;
        for (const auto & p : w.replay_prefixes) replay |= strncmp(x.key.c_str(), p.c_str(), p.size()) == 0;
        if (!replay || x.i32 || x.w == GGML_MEGAKERNEL_WATCH_N) continue;   // as cb_eval: f32 only (result_norm: nothing reads it)
        if (!x.rf) {
            x.rf = fopen((w.replay_dir + "/t." + x.key + ".bin").c_str(), "rb");
            if (!x.rf) { fprintf(stderr, "refdump: no reference %s to replay\n", x.key.c_str()); exit(1); }
        }
        if (fseek(x.rf, (long) w.pos * (long) (n * 4), SEEK_SET) != 0 || fread(m.rep.data() + x.off, 4, n, x.rf) != (size_t) n) {
            fprintf(stderr, "refdump: reference %s has no position %d\n", x.key.c_str(), w.pos); exit(1);
        }
        m.d.rep_mask[x.layer] |= 1u << x.w;
        w.replayed++;
    }
    m.rec[m.rec.size() - 1] = NAN;   // result_norm's last value: the kernel writes it, so a step that bypassed it is caught
    m.set_fn(&m.d);
}

// after the step: append the records
static void mk_collect(mk_mode & m, watch & w) {
    if (std::isnan(m.rec[m.rec.size() - 1])) { fprintf(stderr, "refdump: position %d did not run the megakernel\n", w.pos); exit(1); }
    for (auto & x : m.ws) {
        if (!w.files.count(x.key)) {
            w.shapes[x.key] = std::string(x.i32 ? "i32" : "f32") + ":" + std::to_string(x.ne0) + "x" + std::to_string(x.ne1) + "x1x1";
            w.files[x.key] = fopen((w.dir + "/t." + x.key + ".bin").c_str(), "wb");
        }
        fwrite(m.rec.data() + x.off, 4, x.ne0 * x.ne1, w.files[x.key]);
        w.counts[x.key]++;
    }
}

static void mk_reset(mk_mode & m) {
    for (auto & x : m.ws) if (x.rf) { fclose(x.rf); x.rf = nullptr; }
}
#endif

int main(int argc, char ** argv) {
    std::string model_path, out_dir, tokens_from, tensors, inject, replay_from, replay_tensors = "l_out";
    int n_gen = 0, logits_from = 0, n_ctx = 4096;
    bool kv = false, fa = true, prefill = false;
    std::vector<std::string> prompts;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if (a == "-m") model_path = next();
        else if (a == "-o") out_dir = next();
        else if (a == "--gen") n_gen = atoi(next().c_str());
        else if (a == "--tokens-from") tokens_from = next();
        else if (a == "--logits-from") logits_from = atoi(next().c_str());
        else if (a == "--tensors") tensors = next();
        else if (a == "--inject") inject = next();
        else if (a == "--replay") replay_from = next();
        else if (a == "--replay-tensors") replay_tensors = next();
        else if (a == "--ctx") n_ctx = atoi(next().c_str());
        else if (a == "--kv") kv = true;
        else if (a == "--no-fa") fa = false;
        else if (a == "--prefill") prefill = true;
        else if (a.size() && a[0] == '-') { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
        else prompts.push_back(a);
    }
    if (model_path.empty() || out_dir.empty() || prompts.empty()) {
        fprintf(stderr, "usage: llama-refdump -m model.gguf -o outdir [options] prompt.txt...\n");
        return 1;
    }

    watch w;
    {
        std::stringstream ss(tensors); std::string p;
        while (std::getline(ss, p, ',')) if (!p.empty()) w.prefixes.push_back(p);
    }
    if (!inject.empty()) {  // name@pos:rel[x][:seed]
        size_t at = inject.find('@'), c1 = inject.find(':', at);
        if (at == std::string::npos || c1 == std::string::npos) { fprintf(stderr, "bad --inject\n"); return 1; }
        w.inj_name = inject.substr(0, at);
        const std::string ps = inject.substr(at + 1, c1 - at - 1);
        w.inj_pos = ps == "*" ? -2 : atoi(ps.c_str());
        size_t c2 = inject.find(':', c1 + 1);
        const std::string rs = inject.substr(c1 + 1, c2 == std::string::npos ? std::string::npos : c2 - c1 - 1);
        w.inj_scale = !rs.empty() && rs.back() == 'x';
        w.inj_rel = atof(rs.c_str());
        if (c2 != std::string::npos) w.inj_seed = (unsigned) atoi(inject.substr(c2 + 1).c_str());
        bool listed = false;
        for (auto & p : w.prefixes) listed |= w.inj_name.rfind(p, 0) == 0 || p == "*";
        if (!listed) w.prefixes.push_back(w.inj_name);
    }

    if (prefill && (!tensors.empty() || !replay_from.empty() || !inject.empty())) {
        // watched rows are indexed by position; a prefill chunk would record many positions as one row
        fprintf(stderr, "--prefill cannot be combined with --tensors, --replay or --inject\n");
        return 1;
    }
    if (!replay_from.empty()) {
        std::stringstream ss(replay_tensors); std::string p;
        while (std::getline(ss, p, ',')) if (!p.empty()) { w.replay_prefixes.push_back(p); w.prefixes.push_back(p); }
        if (tokens_from.empty()) tokens_from = replay_from;  // replay needs the reference's positions
    }

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.no_perf = true;
#ifdef REFDUMP_MEGAKERNEL
    mk_mode mk;
    const bool mk_on = !w.prefixes.empty() && mk_setup(mk, w);
#else
    const bool mk_on = false;
#endif
    if (!w.prefixes.empty() && !mk_on) { cp.cb_eval = cb_eval; cp.cb_eval_user_data = &w; }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    mkdir(out_dir.c_str(), 0755);
    for (const auto & pf : prompts) {
        const std::string name = stem(pf);
        const std::string dir = out_dir + "/" + name;
        mkdir(dir.c_str(), 0755);
        const std::string text = read_file(pf);
        std::vector<llama_token> toks(text.size() + 16);
        int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(),
                                llama_vocab_get_add_bos(vocab), true);
        if (nt < 0) { fprintf(stderr, "tokenize failed for %s\n", pf.c_str()); return 1; }
        toks.resize(nt);
        const int n_prompt = nt;
        std::vector<int32_t> forced;
        if (!tokens_from.empty()) {
            forced = read_i32(tokens_from + "/" + name + "/tokens.i32");
            if ((int) forced.size() < n_prompt || memcmp(forced.data(), toks.data(), n_prompt * 4) != 0) {
                fprintf(stderr, "%s: prompt tokens differ from %s\n", name.c_str(), tokens_from.c_str());
                return 1;
            }
        }
        const int n_max = forced.empty() ? n_prompt + n_gen : (int) forced.size();
        if (n_max > n_ctx) { fprintf(stderr, "%s: %d tokens > ctx %d\n", name.c_str(), n_max, n_ctx); return 1; }

        llama_memory_clear(llama_get_memory(ctx), true);
        w.dir = dir; w.files.clear(); w.shapes.clear(); w.counts.clear(); w.seen.clear(); w.injected = 0; w.replayed = 0;
        for (auto & kvp : w.replay_files) if (kvp.second) fclose(kvp.second);
        w.replay_files.clear();
#ifdef REFDUMP_MEGAKERNEL
        if (mk_on) mk_reset(mk);
#endif
        if (!replay_from.empty()) w.replay_dir = replay_from + "/" + name;
        FILE * flog = fopen((dir + "/logits.f32").c_str(), "wb");
        // every token in seq is decoded (so a generated run and a teacher-forced rerun of its tokens produce the
        // same number of positions); generation appends the argmax until EOG or n_prompt + n_gen tokens
        std::vector<int32_t> seq = forced.empty() ? std::vector<int32_t>(toks.begin(), toks.end()) : forced;
        int n_out = 0, first_argmax_diff = -1, start = 0;
        bool stopped_eog = false, done = !forced.empty() || (int) seq.size() >= n_max;
        if (prefill && n_prompt > 1) {  // stock prefill for all but the last prompt token, then decode
            w.pos = -1;
            llama_batch b = llama_batch_get_one(seq.data(), n_prompt - 1);
            if (llama_decode(ctx, b)) { fprintf(stderr, "prefill failed\n"); return 1; }
            start = n_prompt - 1;
        }
        for (int p = start; p < (int) seq.size(); p++) {
            w.pos = p;
            llama_token t = seq[p];
            llama_batch b = llama_batch_get_one(&t, 1);
#ifdef REFDUMP_MEGAKERNEL
            if (mk_on) mk_arm(mk, w);
#endif
            if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed at %d\n", p); return 1; }
#ifdef REFDUMP_MEGAKERNEL
            if (mk_on) { llama_synchronize(ctx); mk_collect(mk, w); }
#endif
            const float * lg = llama_get_logits_ith(ctx, -1);
            int am = 0;
            for (int v = 1; v < n_vocab; v++) if (lg[v] > lg[am]) am = v;
            if (p >= logits_from) { fwrite(lg, sizeof(float), n_vocab, flog); n_out++; }
            if (p + 1 < (int) seq.size()) {
                if (!forced.empty() && p + 1 >= n_prompt && seq[p + 1] != am && first_argmax_diff < 0)
                    first_argmax_diff = p + 1;  // teacher forcing: where greedy decoding would have diverged
            } else if (!done) {
                seq.push_back(am);
                stopped_eog = llama_vocab_is_eog(vocab, am);
                done = stopped_eog || (int) seq.size() >= n_max;
            }
        }
        fclose(flog);
        for (auto & kvp : w.files) fclose(kvp.second);
        w.files.clear();
        {
            FILE * f = fopen((dir + "/tokens.i32").c_str(), "wb");
            fwrite(seq.data(), 4, seq.size(), f); fclose(f);
        }
        size_t kv_bytes = 0;
        if (kv) {
            kv_bytes = llama_state_seq_get_size(ctx, 0);
            std::vector<uint8_t> st(kv_bytes);
            kv_bytes = llama_state_seq_get_data(ctx, st.data(), st.size(), 0);
            FILE * f = fopen((dir + "/kv.bin").c_str(), "wb");
            fwrite(st.data(), 1, kv_bytes, f); fclose(f);
        }
        std::string piece_text;
        for (size_t i = n_prompt; i < seq.size() && i < (size_t) n_prompt + 64; i++) {
            char b[256]; int n = llama_token_to_piece(vocab, seq[i], b, sizeof b, 0, true);
            if (n > 0) piece_text.append(b, n);
        }
        FILE * f = fopen((dir + "/meta.json").c_str(), "w");
        fprintf(f, "{\"prompt\": \"%s\", \"model\": \"%s\", \"n_prompt\": %d, \"n_tokens\": %zu, \"n_vocab\": %d, "
                   "\"logits_from\": %d, \"n_logits\": %d, \"teacher_forced\": %s, \"first_argmax_diff\": %d, "
                   "\"stopped_eog\": %s, \"prefill\": %s, \"flash_attn\": %s, \"kv_bytes\": %zu, "
                   "\"inject\": \"%s\", \"injected\": %d, \"replay\": \"%s\", \"replayed\": %d, "
                   "\"continuation\": \"%s\", \"tensors\": {",
                json_escape(pf).c_str(), json_escape(model_path).c_str(), n_prompt, seq.size(), n_vocab,
                std::max(logits_from, start), n_out, forced.empty() ? "false" : "true", first_argmax_diff,
                stopped_eog ? "true" : "false",
                prefill ? "true" : "false", fa ? "true" : "false", kv_bytes, json_escape(inject).c_str(),
                w.injected, json_escape(replay_from).c_str(), w.replayed, json_escape(piece_text).c_str());
        bool first = true;
        for (auto & s : w.shapes) {
            fprintf(f, "%s\"%s\": {\"shape\": \"%s\", \"count\": %d}", first ? "" : ", ", s.first.c_str(), s.second.c_str(),
                    w.counts.count(s.first) ? w.counts[s.first] : 0);
            first = false;
        }
        fprintf(f, "}}\n");
        fclose(f);
        fprintf(stderr, "refdump: %s: %d prompt + %zu more tokens, %d logit rows%s\n", name.c_str(), n_prompt,
                seq.size() - n_prompt, n_out, w.injected ? ", injected" : "");
    }
#ifdef REFDUMP_MEGAKERNEL
    if (mk_on) mk.set_fn(nullptr);
#endif
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
