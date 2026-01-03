#include "whisper.h"
#include "ggml-backend.h"

#include "httplib.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 3004;
    std::string model_path = "models/ggml-small.en.bin";
    std::string public_dir = "frontend";
    std::string language = "en";

    int threads = std::min(4, static_cast<int>(std::thread::hardware_concurrency()));
    int step_ms = 1000;
    int window_ms = 8000;
    int commit_lag_ms = 2000;
    int max_prompt_tokens = 224;
    int max_repeat = 2;

    bool use_gpu = true;
    bool flash_attn = true;
    bool no_context = true;
    bool no_fallback = false;
};

struct Session {
    std::mutex mutex;
    std::condition_variable cv;
    int in_flight = 0;
    bool closing = false;

    std::vector<float> audio;
    std::vector<whisper_token> prompt_tokens;
    std::string committed_text;
    std::string pending_text;
    int64_t samples_since_last = 0;
    int64_t total_samples = 0;
    int64_t last_committed_ms = 0;

    whisper_state * state = nullptr;
};

struct SegmentResult {
    std::string text;
    int64_t t0_ms = 0;
    int64_t t1_ms = 0;
    std::vector<whisper_token> tokens;
};

static int64_t ms_to_samples(int ms) {
    return static_cast<int64_t>(ms) * WHISPER_SAMPLE_RATE / 1000;
}

static int64_t samples_to_ms(int64_t samples) {
    return samples * 1000 / WHISPER_SAMPLE_RATE;
}

static std::string normalize_sentence(const std::string &text) {
    std::string out;
    out.reserve(text.size());
    bool last_space = false;
    for (char ch : text) {
        unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isalnum(uch)) {
            out.push_back(static_cast<char>(std::tolower(uch)));
            last_space = false;
        } else if (std::isspace(uch)) {
            if (!last_space) {
                out.push_back(' ');
                last_space = true;
            }
        }
    }
    while (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

static std::string dedupe_repeated_sentences(const std::string &text, int max_repeat) {
    if (max_repeat <= 0) {
        return text;
    }

    std::ostringstream out;
    std::string sentence;
    std::string last_norm;
    int repeat = 0;

    auto flush = [&]() {
        if (sentence.empty()) {
            return;
        }
        std::string norm = normalize_sentence(sentence);
        if (norm.empty()) {
            out << sentence;
        } else {
            if (norm == last_norm) {
                repeat++;
            } else {
                last_norm = norm;
                repeat = 1;
            }
            if (repeat <= max_repeat) {
                out << sentence;
            }
        }
        sentence.clear();
    };

    for (char ch : text) {
        sentence.push_back(ch);
        if (ch == '.' || ch == '!' || ch == '?' || ch == '\n') {
            flush();
        }
    }
    flush();
    return out.str();
}

static std::string json_escape(const std::string &input) {
    std::ostringstream out;
    for (unsigned char c : input) {
        switch (c) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(c) << std::dec;
            } else {
                out << c;
            }
        }
    }
    return out.str();
}

static void respond_json(httplib::Response &res, int status, const std::string &body) {
    res.status = status;
    res.set_content(body, "application/json");
}

static std::string random_session_id() {
    static thread_local std::mt19937 rng{std::random_device{}()};
    static const char *kHex = "0123456789abcdef";
    std::string id(32, '0');
    for (char &ch : id) {
        ch = kHex[rng() % 16];
    }
    return id;
}

static size_t longest_overlap(const std::string &prev, const std::string &current, size_t limit) {
    const size_t max_len = std::min({prev.size(), current.size(), limit});
    for (size_t len = max_len; len > 0; --len) {
        if (prev.compare(prev.size() - len, len, current, 0, len) == 0) {
            return len;
        }
    }
    return 0;
}

static void append_pcm16le_to_f32(const std::string &body, std::vector<float> &out) {
    const size_t sample_count = body.size() / sizeof(int16_t);
    out.reserve(out.size() + sample_count);
    const int16_t *data = reinterpret_cast<const int16_t *>(body.data());
    for (size_t i = 0; i < sample_count; ++i) {
        out.push_back(std::max(-1.0f, std::min(1.0f, data[i] / 32768.0f)));
    }
}

static bool run_inference(
    whisper_context *ctx,
    whisper_state *state,
    const ServerConfig &cfg,
    const std::vector<float> &window,
    const std::vector<whisper_token> &prompt_tokens,
    std::vector<SegmentResult> &segments
) {
    if (window.empty()) {
        return false;
    }

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress = false;
    wparams.print_special = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;
    wparams.translate = false;
    wparams.no_timestamps = false;
    wparams.single_segment = false;
    wparams.no_context = cfg.no_context;
    wparams.language = cfg.language.c_str();
    wparams.n_threads = cfg.threads;
    wparams.max_tokens = 0;
    if (cfg.no_fallback) {
        wparams.temperature_inc = 0.0f;
    }

    if (!cfg.no_context && !prompt_tokens.empty()) {
        wparams.prompt_tokens = prompt_tokens.data();
        wparams.prompt_n_tokens = static_cast<int>(prompt_tokens.size());
    } else {
        wparams.prompt_tokens = nullptr;
        wparams.prompt_n_tokens = 0;
    }

    if (whisper_full_with_state(ctx, state, wparams, window.data(), static_cast<int>(window.size())) != 0) {
        return false;
    }

    const int n_segments = whisper_full_n_segments_from_state(state);
    segments.clear();
    segments.reserve(n_segments);

    for (int i = 0; i < n_segments; ++i) {
        const char *segment_text = whisper_full_get_segment_text_from_state(state, i);
        SegmentResult segment;
        segment.text = segment_text ? segment_text : "";
        segment.t0_ms = whisper_full_get_segment_t0_from_state(state, i) * 10;
        segment.t1_ms = whisper_full_get_segment_t1_from_state(state, i) * 10;
        const int token_count = whisper_full_n_tokens_from_state(state, i);
        segment.tokens.reserve(token_count);
        for (int j = 0; j < token_count; ++j) {
            segment.tokens.push_back(whisper_full_get_token_id_from_state(state, i, j));
        }
        segments.push_back(std::move(segment));
    }

    return true;
}

static void print_usage(const char *argv0, const ServerConfig &cfg) {
    std::fprintf(stderr, "\nusage: %s [options]\n\n", argv0);
    std::fprintf(stderr, "options:\n");
    std::fprintf(stderr, "  --host HOST                 [default: %s]\n", cfg.host.c_str());
    std::fprintf(stderr, "  --port PORT                 [default: %d]\n", cfg.port);
    std::fprintf(stderr, "  --model PATH                [default: %s]\n", cfg.model_path.c_str());
    std::fprintf(stderr, "  --public DIR                [default: %s]\n", cfg.public_dir.c_str());
    std::fprintf(stderr, "  --language LANG             [default: %s]\n", cfg.language.c_str());
    std::fprintf(stderr, "  --threads N                 [default: %d]\n", cfg.threads);
    std::fprintf(stderr, "  --step-ms N                 [default: %d]\n", cfg.step_ms);
    std::fprintf(stderr, "  --window-ms N               [default: %d]\n", cfg.window_ms);
    std::fprintf(stderr, "  --commit-lag-ms N           [default: %d]\n", cfg.commit_lag_ms);
    std::fprintf(stderr, "  --max-prompt-tokens N        [default: %d]\n", cfg.max_prompt_tokens);
    std::fprintf(stderr, "  --max-repeat N              [default: %d]\n", cfg.max_repeat);
    std::fprintf(stderr, "  --no-gpu                    disable GPU\n");
    std::fprintf(stderr, "  --no-flash-attn             disable flash attention\n");
    std::fprintf(stderr, "  --keep-context              reuse context between chunks\n");
    std::fprintf(stderr, "  --no-context                do not reuse context between chunks\n");
    std::fprintf(stderr, "  --no-fallback               disable temperature fallback\n");
    std::fprintf(stderr, "  -h, --help                  show this help\n\n");
}

static bool parse_args(int argc, char **argv, ServerConfig &cfg, bool &show_help) {
    show_help = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const std::string &flag) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag.c_str());
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0], cfg);
            show_help = true;
            return false;
        } else if (arg == "--host") {
            const char *val = require_value(arg);
            if (!val) return false;
            cfg.host = val;
        } else if (arg == "--port") {
            const char *val = require_value(arg);
            if (!val) return false;
            cfg.port = std::stoi(val);
        } else if (arg == "--model") {
            const char *val = require_value(arg);
            if (!val) return false;
            cfg.model_path = val;
        } else if (arg == "--public") {
            const char *val = require_value(arg);
            if (!val) return false;
            cfg.public_dir = val;
        } else if (arg == "--language") {
            const char *val = require_value(arg);
            if (!val) return false;
            cfg.language = val;
        } else if (arg == "--threads") {
            const char *val = require_value(arg);
            if (!val) return false;
            cfg.threads = std::stoi(val);
        } else if (arg == "--step-ms") {
            const char *val = require_value(arg);
            if (!val) return false;
            cfg.step_ms = std::stoi(val);
        } else if (arg == "--window-ms") {
            const char *val = require_value(arg);
            if (!val) return false;
            cfg.window_ms = std::stoi(val);
        } else if (arg == "--commit-lag-ms") {
            const char *val = require_value(arg);
            if (!val) return false;
            cfg.commit_lag_ms = std::stoi(val);
        } else if (arg == "--max-prompt-tokens") {
            const char *val = require_value(arg);
            if (!val) return false;
            cfg.max_prompt_tokens = std::stoi(val);
        } else if (arg == "--max-repeat") {
            const char *val = require_value(arg);
            if (!val) return false;
            cfg.max_repeat = std::stoi(val);
        } else if (arg == "--keep-context") {
            cfg.no_context = false;
        } else if (arg == "--no-gpu") {
            cfg.use_gpu = false;
        } else if (arg == "--no-flash-attn") {
            cfg.flash_attn = false;
        } else if (arg == "--no-context") {
            cfg.no_context = true;
        } else if (arg == "--no-fallback") {
            cfg.no_fallback = true;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            print_usage(argv[0], cfg);
            return false;
        }
    }

    if (cfg.step_ms <= 0) {
        cfg.step_ms = 1000;
    }
    if (cfg.window_ms < cfg.step_ms) {
        cfg.window_ms = cfg.step_ms;
    }
    if (cfg.commit_lag_ms < 0) {
        cfg.commit_lag_ms = 0;
    }
    if (cfg.commit_lag_ms > cfg.window_ms) {
        cfg.commit_lag_ms = cfg.window_ms;
    }
    if (cfg.threads <= 0) {
        cfg.threads = 1;
    }
    if (cfg.max_prompt_tokens < 0) {
        cfg.max_prompt_tokens = 0;
    }
    if (cfg.max_repeat < 0) {
        cfg.max_repeat = 0;
    }

    return true;
}

int main(int argc, char **argv) {
    ggml_backend_load_all();

    ServerConfig cfg;
    bool show_help = false;
    if (!parse_args(argc, argv, cfg, show_help)) {
        return show_help ? 0 : 1;
    }

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = cfg.use_gpu;
    cparams.flash_attn = cfg.flash_attn;

    whisper_context *ctx = whisper_init_from_file_with_params(cfg.model_path.c_str(), cparams);
    if (!ctx) {
        std::fprintf(stderr, "Failed to load model: %s\n", cfg.model_path.c_str());
        return 2;
    }

    const int64_t step_samples = ms_to_samples(cfg.step_ms);
    const int64_t window_samples = ms_to_samples(cfg.window_ms);

    std::unordered_map<std::string, std::shared_ptr<Session>> sessions;
    std::mutex sessions_mutex;
    std::mutex whisper_mutex;

    httplib::Server server;

    server.Get("/health", [](const httplib::Request &, httplib::Response &res) {
        respond_json(res, 200, "{\"ok\":true}");
    });

    server.Post("/session", [&](const httplib::Request &, httplib::Response &res) {
        auto session = std::make_shared<Session>();
        session->state = whisper_init_state(ctx);
        if (!session->state) {
            respond_json(res, 500, "{\"ok\":false,\"error\":\"failed to initialize session\"}");
            return;
        }

        std::string session_id = random_session_id();
        {
            std::lock_guard<std::mutex> lock(sessions_mutex);
            sessions.emplace(session_id, session);
        }

        std::string payload = "{\"ok\":true,\"session_id\":\"" + json_escape(session_id) + "\"}";
        respond_json(res, 200, payload);
    });

    server.Post("/chunk", [&](const httplib::Request &req, httplib::Response &res) {
        const auto session_id = req.get_header_value("X-Session-Id");
        if (session_id.empty()) {
            respond_json(res, 400, "{\"ok\":false,\"error\":\"missing X-Session-Id header\"}");
            return;
        }

        std::shared_ptr<Session> session;
        {
            std::lock_guard<std::mutex> lock(sessions_mutex);
            auto it = sessions.find(session_id);
            if (it == sessions.end()) {
                respond_json(res, 404, "{\"ok\":false,\"error\":\"session not found\"}");
                return;
            }
            session = it->second;
        }

        if (req.body.size() % sizeof(int16_t) != 0) {
            respond_json(res, 400, "{\"ok\":false,\"error\":\"invalid pcm payload\"}");
            return;
        }

        std::vector<float> window;
        std::vector<whisper_token> prompt_tokens;
        int64_t total_samples_snapshot = 0;
        int64_t window_start_samples = 0;
        bool should_infer = false;

        {
            std::unique_lock<std::mutex> lock(session->mutex);
            if (session->closing) {
                respond_json(res, 409, "{\"ok\":false,\"error\":\"session closing\"}");
                return;
            }

            session->in_flight++;
            const int64_t new_samples = static_cast<int64_t>(req.body.size() / sizeof(int16_t));
            append_pcm16le_to_f32(req.body, session->audio);
            session->samples_since_last += new_samples;
            session->total_samples += new_samples;

            const size_t max_keep = static_cast<size_t>(window_samples + step_samples);
            if (session->audio.size() > max_keep) {
                const size_t excess = session->audio.size() - max_keep;
                session->audio.erase(session->audio.begin(), session->audio.begin() + excess);
            }

            if (session->samples_since_last >= step_samples) {
                session->samples_since_last -= step_samples;
                should_infer = true;

                const size_t total = session->audio.size();
                const size_t win = static_cast<size_t>(std::min<int64_t>(window_samples, total));
                window.assign(session->audio.end() - win, session->audio.end());
                if (!cfg.no_context) {
                    prompt_tokens = session->prompt_tokens;
                }
                total_samples_snapshot = session->total_samples;
                window_start_samples = std::max<int64_t>(0, total_samples_snapshot - static_cast<int64_t>(win));
            }
        }

        std::vector<SegmentResult> segments;
        bool updated = false;
        int64_t stable_cutoff_ms = 0;
        int64_t window_start_ms = 0;

        if (should_infer) {
            std::lock_guard<std::mutex> infer_lock(whisper_mutex);
            updated = run_inference(ctx, session->state, cfg, window, prompt_tokens, segments);
            const int64_t total_ms = samples_to_ms(total_samples_snapshot);
            window_start_ms = samples_to_ms(window_start_samples);
            stable_cutoff_ms = std::max<int64_t>(0, total_ms - cfg.commit_lag_ms);
        }

        std::string transcript;
        {
            std::unique_lock<std::mutex> lock(session->mutex);
            if (updated) {
                std::string pending;
                std::vector<whisper_token> next_prompt_tokens = session->prompt_tokens;
                int64_t last_committed_ms = session->last_committed_ms;

                for (const auto &segment : segments) {
                    if (segment.text.empty()) {
                        continue;
                    }
                    const int64_t seg_end_ms = window_start_ms + segment.t1_ms;
                    if (seg_end_ms <= stable_cutoff_ms && seg_end_ms > last_committed_ms) {
                        const size_t overlap = longest_overlap(session->committed_text, segment.text, 200);
                        session->committed_text.append(segment.text.substr(overlap));
                        last_committed_ms = seg_end_ms;
                        if (!cfg.no_context) {
                            next_prompt_tokens.insert(next_prompt_tokens.end(),
                                                      segment.tokens.begin(), segment.tokens.end());
                        }
                    } else if (seg_end_ms > last_committed_ms) {
                        pending += segment.text;
                    }
                }

                session->last_committed_ms = last_committed_ms;
                session->pending_text = pending;
                if (!cfg.no_context) {
                    if (cfg.max_prompt_tokens > 0 &&
                        next_prompt_tokens.size() > static_cast<size_t>(cfg.max_prompt_tokens)) {
                        next_prompt_tokens.erase(next_prompt_tokens.begin(),
                                                 next_prompt_tokens.end() - cfg.max_prompt_tokens);
                    }
                    session->prompt_tokens = std::move(next_prompt_tokens);
                } else {
                    session->prompt_tokens.clear();
                }
            }
            transcript = session->committed_text + session->pending_text;
            session->in_flight--;
            session->cv.notify_all();
        }

        const std::string display_text = dedupe_repeated_sentences(transcript, cfg.max_repeat);
        std::ostringstream payload;
        payload << "{\"ok\":true,\"updated\":" << (updated ? "true" : "false")
                << ",\"text\":\"" << json_escape(display_text) << "\"}";
        respond_json(res, 200, payload.str());
    });

    server.Post("/stop", [&](const httplib::Request &req, httplib::Response &res) {
        const auto session_id = req.get_header_value("X-Session-Id");
        if (session_id.empty()) {
            respond_json(res, 400, "{\"ok\":false,\"error\":\"missing X-Session-Id header\"}");
            return;
        }

        std::shared_ptr<Session> session;
        {
            std::lock_guard<std::mutex> lock(sessions_mutex);
            auto it = sessions.find(session_id);
            if (it == sessions.end()) {
                respond_json(res, 404, "{\"ok\":false,\"error\":\"session not found\"}");
                return;
            }
            session = it->second;
            sessions.erase(it);
        }

        std::string transcript;
        whisper_state *state = nullptr;
        {
            std::unique_lock<std::mutex> lock(session->mutex);
            session->closing = true;
            session->cv.wait(lock, [&]() { return session->in_flight == 0; });
            transcript = session->committed_text + session->pending_text;
            state = session->state;
            session->state = nullptr;
        }

        if (state) {
            whisper_free_state(state);
        }

        const std::string display_text = dedupe_repeated_sentences(transcript, cfg.max_repeat);
        std::string payload = "{\"ok\":true,\"session_id\":\"" + json_escape(session_id) +
                              "\",\"text\":\"" + json_escape(display_text) + "\"}";
        respond_json(res, 200, payload);
    });

    server.set_base_dir(cfg.public_dir.c_str());

    std::fprintf(stderr, "\nrealtime server listening at http://%s:%d\n", cfg.host.c_str(), cfg.port);

    server.listen(cfg.host.c_str(), cfg.port);

    whisper_free(ctx);
    return 0;
}
