#include "audio_capture.hpp"
#include "cli.hpp"
#include "interactive.hpp"
#include "kms_capture.hpp"
#include "pipeline.hpp"
#include "privileges.hpp"
#include "user_config.hpp"

#include <atomic>
#include <csignal>
#include <format>
#include <iostream>
#include <string_view>
#include <unistd.h>

namespace {
vkcap::PipelineSignals g_signals{};

void handle_stop_signal(int sig) {
    g_signals.stop.store(true, std::memory_order_release);
    // Restore default SIGINT so a second Ctrl+C force-exits if finalization is slow
    if (sig == SIGINT) {
        std::signal(SIGINT, SIG_DFL);
    }
}

void handle_pause_signal(int) {
    bool cur = g_signals.paused.load(std::memory_order_relaxed);
    g_signals.paused.store(!cur, std::memory_order_release);
}

void handle_save_replay_signal(int) {
    g_signals.save_replay.store(true, std::memory_order_release);
}
}  // namespace

int main(int argc, char **argv) {
    // Immediately drop all unneeded capabilities and clear Effective set;
    // CAP_SYS_ADMIN is kept only in Permitted and raised briefly around drmModeGetFB2/Prime.
    vkcap::init_least_privilege();

    // If --instant-replay / --replay-buffer is requested, seed from saved user defaults
    // (~/.config/vkcap/defaults.conf) so instant replay respects personal display, profile,
    // audio, volume, and output directory settings (explicit CLI flags still override).
    std::optional<vkcap::Config> saved_defaults;
    bool wants_replay = false;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--instant-replay" || arg.starts_with("--instant-replay=") ||
            arg == "--replay-buffer" || arg.starts_with("--replay-buffer=")) {
            wants_replay = true;
            break;
        }
    }
    if (wants_replay) {
        auto dres = vkcap::load_user_defaults(vkcap::user_defaults_path());
        if (dres && dres->has_value()) {
            saved_defaults = std::move(**dres);
        }
    }

    auto cfg_res = vkcap::parse_args(argc, argv, saved_defaults ? &*saved_defaults : nullptr);
    if (!cfg_res) {
        std::cerr << "Error: " << cfg_res.error() << '\n';
        return 2;
    }
    if (cfg_res->show_help) {
        vkcap::print_help(argv[0]);
        return 0;
    }
    if (cfg_res->list_profiles) {
        vkcap::print_profiles();
        return 0;
    }
    if (cfg_res->list_mics) {
        auto mics = vkcap::AudioCapture::enumerate_microphones();
        if (!mics) {
            std::cerr << "Error: " << mics.error() << '\n';
            return 1;
        }
        if (mics->empty()) {
            std::cout << "No PipeWire microphone sources found.\n";
            return 0;
        }
        for (size_t i = 0; i < mics->size(); ++i) {
            const auto &m = (*mics)[i];
            std::cout << std::format("[{}] id={:<4} {}  ({})\n", i, m.id, m.description, m.name);
        }
        return 0;
    }

    if (cfg_res->list_displays) {
        auto disps = vkcap::KmsCapture::enumerate_displays();
        if (!disps) {
            std::cerr << "Error: " << disps.error() << '\n';
            return 1;
        }
        if (disps->empty()) {
            std::cout << "No connected active DRM/KMS displays found.\n";
            return 0;
        }
        for (size_t i = 0; i < disps->size(); ++i) {
            const auto &d = (*disps)[i];
            std::cout << std::format(
                "[{}] {:<10} {}x{}@{:.1f}Hz  {}  crtc={}  plane={}  cursor={}\n",
                i, d.connector_name, d.width, d.height, d.refresh_hz,
                d.card_path, d.crtc_id, d.primary_plane_id, d.cursor_plane_id);
        }
        return 0;
    }

    // Launch interactive setup prompt when invoked with no arguments in a terminal,
    // or whenever explicitly requested via -i / --interactive.
    vkcap::Config cfg = *cfg_res;
    if (cfg.interactive || (argc == 1 && ::isatty(STDIN_FILENO) != 0)) {
        auto wiz = vkcap::run_interactive_setup(cfg, argv[0], std::cin, std::cout);
        if (!wiz) {
            std::cerr << "Error: " << wiz.error() << '\n';
            return 1;
        }
        if (!*wiz) {
            std::cout << "Cancelled.\n";
            return 0;
        }
    }

    struct sigaction sa{};
    sa.sa_handler = handle_stop_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGHUP, &sa, nullptr);

    struct sigaction sa_usr1{};
    sa_usr1.sa_handler = handle_pause_signal;
    sigemptyset(&sa_usr1.sa_mask);
    sa_usr1.sa_flags = SA_RESTART;
    sigaction(SIGUSR1, &sa_usr1, nullptr);

    struct sigaction sa_usr2{};
    sa_usr2.sa_handler = handle_save_replay_signal;
    sigemptyset(&sa_usr2.sa_mask);
    sa_usr2.sa_flags = SA_RESTART;
    sigaction(SIGUSR2, &sa_usr2, nullptr);

    auto res = vkcap::run_kms_pipeline(cfg, g_signals);
    if (!res) {
        std::cerr << "vkcap failed: " << res.error() << '\n';
        return 1;
    }
    return 0;
}
