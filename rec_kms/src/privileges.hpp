#pragma once

namespace vkcap {

/// Configures strict least-privilege process capabilities at startup:
///  1. Sets PR_SET_NO_NEW_PRIVS so no exec/child can gain privileges.
///  2. Strips ALL capabilities from Effective, Inheritable, and Permitted
///     EXCEPT CAP_SYS_ADMIN in Permitted only (Effective becomes 0).
/// Returns true if CAP_SYS_ADMIN is present in the Permitted set.
bool init_least_privilege() noexcept;

/// Returns true if CAP_SYS_ADMIN is available in the Permitted set.
bool has_cap_sys_admin_permitted() noexcept;

/// RAII guard that temporarily raises CAP_SYS_ADMIN into the Effective set
/// solely for drmModeGetFB2() / drmPrimeHandleToFD() and drops it immediately
/// upon scope exit. No-op if CAP_SYS_ADMIN is not permitted.
class ScopedCapSysAdmin {
public:
    ScopedCapSysAdmin() noexcept;
    ~ScopedCapSysAdmin();

    ScopedCapSysAdmin(const ScopedCapSysAdmin &) = delete;
    ScopedCapSysAdmin &operator=(const ScopedCapSysAdmin &) = delete;

    bool active() const noexcept { return active_; }

private:
    bool active_ = false;
};

}  // namespace vkcap
