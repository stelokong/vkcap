#include "audio_capture.hpp"
#include "cli.hpp"
#include "interactive.hpp"
#include "kms_capture.hpp"
#include "pipeline.hpp"
#include "privileges.hpp"

#include <atomic>
#include <csignal>
#include <format>
#include <iostream>
#include <unistd.h>

namespace {
std::atomic<bool> g_stop{false};

void handle_signal(int) {
    g_stop.store(true, std::memory_order_release);
}
}  // namespace

int main(int argc, char **argv) {
    // Immediately drop all unneeded capabilities and clear Effective set;
    // CAP_SYS_ADMIN is kept only in Permitted and raised briefly around drmModeGetFB2/Prime.
    vkcap::init_least_privilege();

    auto cfg_res = vkcap::parse_args(argc, argv);
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
    sa.sa_handler = handle_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGHUP, &sa, nullptr);

    auto res = vkcap::run_kms_pipeline(cfg, g_stop);
    if (!res) {
        std::cerr << "vkcap failed: " << res.error() << '\n';
        return 1;
    }
    return 0;
}
