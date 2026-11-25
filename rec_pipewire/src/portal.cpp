#include "portal.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <gio/gio.h>
#include <gio/gunixfdlist.h>
#include <iostream>
#include <unistd.h>

namespace vkcap {

namespace {

constexpr const char *kPortalBus = "org.freedesktop.portal.Desktop";
constexpr const char *kPortalObj = "/org/freedesktop/portal/desktop";
constexpr const char *kScreenCastIface = "org.freedesktop.portal.ScreenCast";
constexpr const char *kRequestIface = "org.freedesktop.portal.Request";
constexpr const char *kSessionIface = "org.freedesktop.portal.Session";

struct RequestWaiter {
    GMainLoop *loop = nullptr;
    uint32_t response = 2;
    GVariant *results = nullptr;
    bool done = false;
};

void on_request_response(GDBusConnection *, const gchar *, const gchar *, const gchar *,
                         const gchar *, GVariant *parameters, gpointer user_data) {
    auto *w = static_cast<RequestWaiter *>(user_data);
    g_variant_get(parameters, "(u@a{sv})", &w->response, &w->results);
    w->done = true;
    if (w->loop) {
        g_main_loop_quit(w->loop);
    }
}

std::string sanitize_sender(const char *unique_name) {
    std::string s(unique_name ? unique_name : "");
    if (!s.empty() && s.front() == ':') {
        s.erase(s.begin());
    }
    std::replace(s.begin(), s.end(), '.', '_');
    return s;
}

std::string make_request_path(GDBusConnection *conn, const std::string &token) {
    const char *uname = g_dbus_connection_get_unique_name(conn);
    return std::format("/org/freedesktop/portal/desktop/request/{}/{}", sanitize_sender(uname), token);
}

std::expected<GVariant *, std::string> call_portal_request(
    GDBusConnection *conn, const char *method, GVariant *params, const std::string &handle_token) {
    RequestWaiter waiter;
    waiter.loop = g_main_loop_new(nullptr, FALSE);

    std::string expected_path = make_request_path(conn, handle_token);
    guint sub_id = g_dbus_connection_signal_subscribe(
        conn, kPortalBus, kRequestIface, "Response", expected_path.c_str(),
        nullptr, G_DBUS_SIGNAL_FLAGS_NO_MATCH_RULE, on_request_response, &waiter, nullptr);

    GError *err = nullptr;
    GVariant *reply = g_dbus_connection_call_sync(
        conn, kPortalBus, kPortalObj, kScreenCastIface, method, params,
        G_VARIANT_TYPE("(o)"), G_DBUS_CALL_FLAGS_NONE, 15000, nullptr, &err);

    if (!reply) {
        g_dbus_connection_signal_unsubscribe(conn, sub_id);
        g_main_loop_unref(waiter.loop);
        std::string msg = err ? err->message : "unknown D-Bus error";
        if (err) g_error_free(err);
        return std::unexpected(std::format("Portal {} call failed: {}", method, msg));
    }

    const gchar *actual_path = nullptr;
    g_variant_get(reply, "(&o)", &actual_path);
    if (actual_path && expected_path != actual_path && !waiter.done) {
        g_dbus_connection_signal_unsubscribe(conn, sub_id);
        sub_id = g_dbus_connection_signal_subscribe(
            conn, kPortalBus, kRequestIface, "Response", actual_path,
            nullptr, G_DBUS_SIGNAL_FLAGS_NO_MATCH_RULE, on_request_response, &waiter, nullptr);
    }
    g_variant_unref(reply);

    if (!waiter.done) {
        g_main_loop_run(waiter.loop);
    }
    g_dbus_connection_signal_unsubscribe(conn, sub_id);
    g_main_loop_unref(waiter.loop);

    if (waiter.response != 0) {
        if (waiter.results) g_variant_unref(waiter.results);
        if (waiter.response == 1) {
            return std::unexpected(std::format("Portal {} cancelled by user", method));
        }
        return std::unexpected(std::format("Portal {} failed with response code {}", method, waiter.response));
    }
    return waiter.results;
}

uint32_t read_portal_uint_prop(GDBusConnection *conn, const char *prop_name, uint32_t fallback) {
    GError *err = nullptr;
    GVariant *res = g_dbus_connection_call_sync(
        conn, kPortalBus, kPortalObj, "org.freedesktop.DBus.Properties", "Get",
        g_variant_new("(ss)", kScreenCastIface, prop_name),
        G_VARIANT_TYPE("(v)"), G_DBUS_CALL_FLAGS_NONE, 5000, nullptr, &err);
    if (!res) {
        if (err) g_error_free(err);
        return fallback;
    }
    GVariant *inner = nullptr;
    g_variant_get(res, "(v)", &inner);
    uint32_t val = fallback;
    if (inner && g_variant_is_of_type(inner, G_VARIANT_TYPE_UINT32)) {
        val = g_variant_get_uint32(inner);
    }
    if (inner) g_variant_unref(inner);
    g_variant_unref(res);
    return val;
}

std::filesystem::path restore_token_path() {
    const char *state_home = std::getenv("XDG_STATE_HOME");
    std::filesystem::path base;
    if (state_home && *state_home) {
        base = state_home;
    } else {
        const char *home = std::getenv("HOME");
        base = home ? std::filesystem::path(home) / ".local/state" : std::filesystem::path("/tmp");
    }
    return base / "vkcap" / "restore_token";
}

}  // namespace

struct PortalSession::Impl {
    GDBusConnection *conn = nullptr;
    std::string session_handle;
};

PortalSession::~PortalSession() {
    close();
}

PortalSession::PortalSession(PortalSession &&other) noexcept
    : impl_(other.impl_), stream_(std::move(other.stream_)) {
    other.impl_ = nullptr;
    other.stream_.pipewire_fd = -1;
}

PortalSession &PortalSession::operator=(PortalSession &&other) noexcept {
    if (this != &other) {
        close();
        impl_ = other.impl_;
        stream_ = std::move(other.stream_);
        other.impl_ = nullptr;
        other.stream_.pipewire_fd = -1;
    }
    return *this;
}

void PortalSession::close() {
    if (stream_.pipewire_fd >= 0) {
        ::close(stream_.pipewire_fd);
        stream_.pipewire_fd = -1;
    }
    if (impl_) {
        if (impl_->conn && !impl_->session_handle.empty()) {
            GError *err = nullptr;
            GVariant *res = g_dbus_connection_call_sync(
                impl_->conn, kPortalBus, impl_->session_handle.c_str(), kSessionIface,
                "Close", nullptr, nullptr, G_DBUS_CALL_FLAGS_NONE, 2000, nullptr, &err);
            if (res) g_variant_unref(res);
            if (err) g_error_free(err);
        }
        if (impl_->conn) {
            g_object_unref(impl_->conn);
        }
        delete impl_;
        impl_ = nullptr;
    }
}

std::optional<std::string> load_saved_restore_token() {
    std::ifstream in(restore_token_path());
    if (!in) return std::nullopt;
    std::string tok;
    std::getline(in, tok);
    if (tok.empty()) return std::nullopt;
    return tok;
}

bool save_restore_token(const std::string &token) {
    if (token.empty()) return false;
    auto path = restore_token_path();
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) return false;
    std::ofstream out(path, std::ios::trunc);
    if (!out) return false;
    out << token << '\n';
    return out.good();
}

std::expected<PortalSession, std::string> PortalSession::acquire(const Config &cfg) {
    GError *err = nullptr;
    GDBusConnection *conn = g_bus_get_sync(G_BUS_TYPE_SESSION, nullptr, &err);
    if (!conn) {
        std::string msg = err ? err->message : "failed to connect to session D-Bus";
        if (err) g_error_free(err);
        return std::unexpected(std::format("D-Bus session error (is Wayland/DBus running?): {}", msg));
    }

    PortalSession session;
    session.impl_ = new Impl{conn, {}};

    const uint64_t pid = static_cast<uint64_t>(getpid());

    // 1. CreateSession
    std::string create_tok = std::format("vkcap_req_create_{}", pid);
    std::string sess_tok = std::format("vkcap_sess_{}", pid);
    GVariantBuilder create_opts;
    g_variant_builder_init(&create_opts, G_VARIANT_TYPE_VARDICT);
    g_variant_builder_add(&create_opts, "{sv}", "handle_token", g_variant_new_string(create_tok.c_str()));
    g_variant_builder_add(&create_opts, "{sv}", "session_handle_token", g_variant_new_string(sess_tok.c_str()));

    auto create_res = call_portal_request(
        conn, "CreateSession", g_variant_new("(a{sv})", &create_opts), create_tok);
    if (!create_res) {
        return std::unexpected(create_res.error());
    }

    const gchar *sh_str = nullptr;
    if (!g_variant_lookup(*create_res, "session_handle", "&s", &sh_str) || !sh_str) {
        g_variant_unref(*create_res);
        return std::unexpected("Portal CreateSession did not return session_handle");
    }
    session.impl_->session_handle = sh_str;
    g_variant_unref(*create_res);

    // 2. SelectSources
    uint32_t avail_sources = read_portal_uint_prop(conn, "AvailableSourceTypes", 1u);
    uint32_t avail_cursors = read_portal_uint_prop(conn, "AvailableCursorModes", 1u | 2u);
    uint32_t portal_version = read_portal_uint_prop(conn, "version", 1u);

    uint32_t want_sources = 1u;
    switch (cfg.source_type) {
        case SourceType::Monitor: want_sources = 1u; break;
        case SourceType::Window:  want_sources = 2u; break;
        case SourceType::All:     want_sources = 1u | 2u; break;
    }
    if ((want_sources & avail_sources) != 0) {
        want_sources &= avail_sources;
    }

    uint32_t want_cursor = 2u;  // Embedded
    if (cfg.cursor == CursorMode::Hidden) want_cursor = 1u;
    else if (cfg.cursor == CursorMode::Metadata) want_cursor = 4u;
    if ((want_cursor & avail_cursors) == 0) {
        if (avail_cursors & 2u) want_cursor = 2u;
        else if (avail_cursors & 1u) want_cursor = 1u;
        else want_cursor = avail_cursors;
    }

    std::optional<std::string> restore_tok = cfg.restore_token;
    if (!restore_tok && cfg.use_saved_restore) {
        restore_tok = load_saved_restore_token();
    }

    std::string sel_tok = std::format("vkcap_req_select_{}", pid);
    GVariantBuilder sel_opts;
    g_variant_builder_init(&sel_opts, G_VARIANT_TYPE_VARDICT);
    g_variant_builder_add(&sel_opts, "{sv}", "handle_token", g_variant_new_string(sel_tok.c_str()));
    g_variant_builder_add(&sel_opts, "{sv}", "types", g_variant_new_uint32(want_sources));
    g_variant_builder_add(&sel_opts, "{sv}", "multiple", g_variant_new_boolean(FALSE));
    g_variant_builder_add(&sel_opts, "{sv}", "cursor_mode", g_variant_new_uint32(want_cursor));
    if (portal_version >= 4) {
        uint32_t persist_mode = (cfg.save_restore || restore_tok.has_value()) ? 2u : 0u;
        g_variant_builder_add(&sel_opts, "{sv}", "persist_mode", g_variant_new_uint32(persist_mode));
        if (restore_tok && !restore_tok->empty()) {
            g_variant_builder_add(&sel_opts, "{sv}", "restore_token", g_variant_new_string(restore_tok->c_str()));
        }
    }

    auto sel_res = call_portal_request(
        conn, "SelectSources",
        g_variant_new("(oa{sv})", session.impl_->session_handle.c_str(), &sel_opts),
        sel_tok);
    if (!sel_res) {
        return std::unexpected(sel_res.error());
    }
    g_variant_unref(*sel_res);

    // 3. Start (triggers portal display picker dialog unless restored)
    std::string start_tok = std::format("vkcap_req_start_{}", pid);
    GVariantBuilder start_opts;
    g_variant_builder_init(&start_opts, G_VARIANT_TYPE_VARDICT);
    g_variant_builder_add(&start_opts, "{sv}", "handle_token", g_variant_new_string(start_tok.c_str()));

    auto start_res = call_portal_request(
        conn, "Start",
        g_variant_new("(osa{sv})", session.impl_->session_handle.c_str(), "", &start_opts),
        start_tok);
    if (!start_res) {
        return std::unexpected(start_res.error());
    }

    GVariant *streams = g_variant_lookup_value(*start_res, "streams", G_VARIANT_TYPE("a(ua{sv})"));
    if (!streams || g_variant_n_children(streams) == 0) {
        if (streams) g_variant_unref(streams);
        g_variant_unref(*start_res);
        return std::unexpected("Portal Start returned no screen cast streams");
    }

    GVariant *first_stream = g_variant_get_child_value(streams, 0);
    GVariant *props = nullptr;
    g_variant_get(first_stream, "(u@a{sv})", &session.stream_.node_id, &props);

    if (props) {
        guint64 serial = 0;
        if (g_variant_lookup(props, "pipewire-serial", "t", &serial)) {
            session.stream_.pipewire_serial = static_cast<uint64_t>(serial);
        }
        GVariant *size_v = g_variant_lookup_value(props, "size", G_VARIANT_TYPE("(ii)"));
        if (size_v) {
            g_variant_get(size_v, "(ii)", &session.stream_.width, &session.stream_.height);
            g_variant_unref(size_v);
        }
        g_variant_unref(props);
    }
    g_variant_unref(first_stream);
    g_variant_unref(streams);

    const gchar *new_restore_tok = nullptr;
    if (g_variant_lookup(*start_res, "restore_token", "&s", &new_restore_tok) && new_restore_tok) {
        session.stream_.restore_token = std::string(new_restore_tok);
        if (cfg.save_restore) {
            save_restore_token(*session.stream_.restore_token);
        }
    }
    g_variant_unref(*start_res);

    // 4. OpenPipeWireRemote
    GVariantBuilder open_opts;
    g_variant_builder_init(&open_opts, G_VARIANT_TYPE_VARDICT);
    GUnixFDList *fd_list = nullptr;
    GVariant *open_reply = g_dbus_connection_call_with_unix_fd_list_sync(
        conn, kPortalBus, kPortalObj, kScreenCastIface, "OpenPipeWireRemote",
        g_variant_new("(oa{sv})", session.impl_->session_handle.c_str(), &open_opts),
        G_VARIANT_TYPE("(h)"), G_DBUS_CALL_FLAGS_NONE, 5000, nullptr, &fd_list, nullptr, &err);

    if (!open_reply) {
        std::string msg = err ? err->message : "unknown error";
        if (err) g_error_free(err);
        return std::unexpected(std::format("Portal OpenPipeWireRemote failed: {}", msg));
    }

    gint32 fd_idx = -1;
    g_variant_get(open_reply, "(h)", &fd_idx);
    g_variant_unref(open_reply);

    if (!fd_list || fd_idx < 0) {
        if (fd_list) g_object_unref(fd_list);
        return std::unexpected("Portal OpenPipeWireRemote did not return a valid fd");
    }

    int pw_fd = g_unix_fd_list_get(fd_list, fd_idx, &err);
    g_object_unref(fd_list);
    if (pw_fd < 0) {
        std::string msg = err ? err->message : "g_unix_fd_list_get failed";
        if (err) g_error_free(err);
        return std::unexpected(std::format("Failed to extract PipeWire fd: {}", msg));
    }
    session.stream_.pipewire_fd = pw_fd;

    if (cfg.verbose) {
        std::cerr << std::format(
            "[portal] ScreenCast started: node_id={} serial={} size={}x{} portal_v={}\n",
            session.stream_.node_id,
            session.stream_.pipewire_serial ? std::to_string(*session.stream_.pipewire_serial) : "n/a",
            session.stream_.width, session.stream_.height, portal_version);
    }

    return session;
}

}  // namespace vkcap
