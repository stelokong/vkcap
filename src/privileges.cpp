#include "privileges.hpp"

#include <atomic>
#include <cstdint>
#include <linux/capability.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace vkcap {

namespace {

std::atomic<bool> g_has_cap_sys_admin{false};

constexpr uint32_t kCapSysAdminBit = 1u << (CAP_SYS_ADMIN & 31);
constexpr size_t kCapSysAdminIdx = CAP_SYS_ADMIN >> 5;

bool set_effective_cap_sys_admin(bool enable) noexcept {
    __user_cap_header_struct hdr{};
    hdr.version = _LINUX_CAPABILITY_VERSION_3;
    hdr.pid = 0;

    __user_cap_data_struct data[2]{};
    data[kCapSysAdminIdx].permitted = kCapSysAdminBit;
    data[kCapSysAdminIdx].effective = enable ? kCapSysAdminBit : 0u;
    data[kCapSysAdminIdx].inheritable = 0u;

    return ::syscall(SYS_capset, &hdr, data) == 0;
}

}  // namespace

bool init_least_privilege() noexcept {
    // Prevent privilege escalation across any future execve
    ::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);

    __user_cap_header_struct hdr{};
    hdr.version = _LINUX_CAPABILITY_VERSION_3;
    hdr.pid = 0;
    __user_cap_data_struct data[2]{};
    if (::syscall(SYS_capget, &hdr, data) != 0) {
        g_has_cap_sys_admin.store(false, std::memory_order_relaxed);
        return false;
    }

    const bool permitted = (data[kCapSysAdminIdx].permitted & kCapSysAdminBit) != 0;
    g_has_cap_sys_admin.store(permitted, std::memory_order_relaxed);

    // Drop every capability from Effective, Inheritable, and Permitted except
    // CAP_SYS_ADMIN in Permitted (Effective = 0 while idle / running Vulkan / FFmpeg / file I/O).
    __user_cap_data_struct minimal[2]{};
    if (permitted) {
        minimal[kCapSysAdminIdx].permitted = kCapSysAdminBit;
    }
    ::syscall(SYS_capset, &hdr, minimal);
    return permitted;
}

bool has_cap_sys_admin_permitted() noexcept {
    return g_has_cap_sys_admin.load(std::memory_order_relaxed);
}

ScopedCapSysAdmin::ScopedCapSysAdmin() noexcept {
    if (g_has_cap_sys_admin.load(std::memory_order_relaxed)) {
        active_ = set_effective_cap_sys_admin(true);
    }
}

ScopedCapSysAdmin::~ScopedCapSysAdmin() {
    if (active_) {
        set_effective_cap_sys_admin(false);
    }
}

}  // namespace vkcap
