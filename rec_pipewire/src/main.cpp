#include "cli.hpp"
#include "pipeline.hpp"

#include <atomic>
#include <csignal>
#include <iostream>

namespace {
std::atomic<bool> g_stop{false};

void handle_signal(int) {
    g_stop.store(true, std::memory_order_release);
}
}  // namespace

int main(int argc, char **argv) {
    auto cfg_res = vkcap::parse_args(argc, argv);
    if (!cfg_res) {
        std::cerr << "Error: " << cfg_res.error() << '\n';
        return 2;
    }
    if (cfg_res->show_help) {
        vkcap::print_help(argv[0]);
        return 0;
    }

    struct sigaction sa{};
    sa.sa_handler = handle_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    auto res = vkcap::run_capture_pipeline(*cfg_res, g_stop);
    if (!res) {
        std::cerr << "vkcap failed: " << res.error() << '\n';
        return 1;
    }
    return 0;
}
