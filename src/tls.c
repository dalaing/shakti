/* tls — OpenSSL transport for long-lived sockets (wss, rest.listen_tls). */
#include "tls.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <openssl/err.h>
#include <openssl/sha.h>
#include <openssl/ssl.h>

static int g_tls_inited;
static SSL_CTX *g_client_ctx;

static void tls_harden_ctx(SSL_CTX *ctx) {
    if (!ctx) return;
#ifdef TLS1_2_VERSION
    SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
#endif
}

void shakti_tls_init(void) {
    if (g_tls_inited) return;
    g_tls_inited = 1;
    OPENSSL_init_ssl(0, NULL);
    g_client_ctx = SSL_CTX_new(TLS_client_method());
    if (g_client_ctx) {
        tls_harden_ctx(g_client_ctx);
        SSL_CTX_set_default_verify_paths(g_client_ctx);
        SSL_CTX_set_verify(g_client_ctx, SSL_VERIFY_PEER, NULL);
    }
}

int tls_sha1(const void *data, size_t n, unsigned char out[20]) {
    if (!out) return -1;
    if (!SHA1((const unsigned char *)data, n, out)) return -1;
    return 0;
}

static void tls_clear_err(void) {
    while (ERR_get_error()) {
    }
}

static long tls_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000L + ts.tv_nsec / 1000000L;
}

static int tls_is_ip_literal(const char *s) {
    unsigned char buf[16];
    return inet_pton(AF_INET, s, buf) == 1 || inet_pton(AF_INET6, s, buf) == 1;
}

void *shakti_tls_connect(int fd, const char *hostname, int insecure) {
    shakti_tls_init();
    if (fd < 0 || !g_client_ctx) return NULL;
    SSL *ssl = SSL_new(g_client_ctx);
    if (!ssl) return NULL;
    if (SSL_set_fd(ssl, fd) != 1) {
        SSL_free(ssl);
        return NULL;
    }
    int ip = hostname && hostname[0] && tls_is_ip_literal(hostname);
    if (hostname && hostname[0] && !ip)
        SSL_set_tlsext_host_name(ssl, hostname);
    if (insecure) {
        SSL_set_verify(ssl, SSL_VERIFY_NONE, NULL);
    } else {
        SSL_set_verify(ssl, SSL_VERIFY_PEER, NULL);
        int ok;
        if (!hostname || !hostname[0])
            ok = 0;
        else if (ip)
            ok = X509_VERIFY_PARAM_set1_ip_asc(SSL_get0_param(ssl), hostname) == 1;
        else
            ok = SSL_set1_host(ssl, hostname) == 1;
        if (!ok) {
            SSL_free(ssl);
            return NULL;
        }
    }
    tls_clear_err();
    if (SSL_connect(ssl) != 1) {
        SSL_free(ssl);
        return NULL;
    }
    return ssl;
}

void *shakti_tls_server_ctx(const char *cert_path, const char *key_path) {
    shakti_tls_init();
    if (!cert_path || !key_path || !cert_path[0] || !key_path[0]) return NULL;
    SSL_CTX *ctx = SSL_CTX_new(TLS_server_method());
    if (!ctx) return NULL;
    tls_harden_ctx(ctx);
    if (SSL_CTX_use_certificate_file(ctx, cert_path, SSL_FILETYPE_PEM) != 1) {
        SSL_CTX_free(ctx);
        return NULL;
    }
    if (SSL_CTX_use_PrivateKey_file(ctx, key_path, SSL_FILETYPE_PEM) != 1) {
        SSL_CTX_free(ctx);
        return NULL;
    }
    if (SSL_CTX_check_private_key(ctx) != 1) {
        SSL_CTX_free(ctx);
        return NULL;
    }
    return ctx;
}

void shakti_tls_server_ctx_free(void *ctx) {
    if (ctx) SSL_CTX_free((SSL_CTX *)ctx);
}

void *shakti_tls_accept_timeout(int fd, void *server_ctx, int timeout_ms) {
    if (fd < 0 || !server_ctx || timeout_ms <= 0) return NULL;
    int fl = fcntl(fd, F_GETFL, 0);
    if (fl < 0 || fcntl(fd, F_SETFL, fl | O_NONBLOCK) < 0) return NULL;
    SSL *ssl = SSL_new((SSL_CTX *)server_ctx);
    if (!ssl || SSL_set_fd(ssl, fd) != 1) {
        if (ssl) SSL_free(ssl);
        fcntl(fd, F_SETFL, fl);
        return NULL;
    }
    long deadline = tls_now_ms() + timeout_ms;
    int ok = 0;
    tls_clear_err();
    for (;;) {
        int r = SSL_accept(ssl);
        if (r == 1) {
            ok = 1;
            break;
        }
        int e = SSL_get_error(ssl, r);
        short ev;
        if (e == SSL_ERROR_WANT_READ) ev = POLLIN;
        else if (e == SSL_ERROR_WANT_WRITE) ev = POLLOUT;
        else if (e == SSL_ERROR_SYSCALL && errno == EINTR) continue;
        else break;
        long left = deadline - tls_now_ms();
        if (left <= 0) break;
        struct pollfd pfd = {.fd = fd, .events = ev, .revents = 0};
        int pr = poll(&pfd, 1, (int)left);
        if (pr < 0 && errno == EINTR) continue;
        if (pr <= 0) break;
    }
    fcntl(fd, F_SETFL, fl);
    if (!ok) {
        SSL_free(ssl);
        return NULL;
    }
    return ssl;
}

void *shakti_tls_accept(int fd, void *server_ctx) {
    return shakti_tls_accept_timeout(fd, server_ctx, 5000);
}

ssize_t shakti_tls_read(void *ssl, void *buf, size_t n) {
    if (!ssl || !buf) return -1;
    for (;;) {
        int r = SSL_read((SSL *)ssl, buf, (int)n);
        if (r > 0) return r;
        int e = SSL_get_error((SSL *)ssl, r);
        if (e == SSL_ERROR_WANT_READ || e == SSL_ERROR_WANT_WRITE) {
            errno = EAGAIN;
            return -1;
        }
        if (e == SSL_ERROR_ZERO_RETURN) return 0;
        if (e == SSL_ERROR_SYSCALL && errno == EINTR) continue;
        return -1;
    }
}

ssize_t shakti_tls_write(void *ssl, const void *buf, size_t n) {
    if (!ssl) return -1;
    if (!buf && n) return -1;
    size_t sent = 0;
    while (sent < n) {
        int w = SSL_write((SSL *)ssl, (const char *)buf + sent, (int)(n - sent));
        if (w > 0) {
            sent += (size_t)w;
            continue;
        }
        int e = SSL_get_error((SSL *)ssl, w);
        if (e == SSL_ERROR_WANT_READ || e == SSL_ERROR_WANT_WRITE) {
            errno = EAGAIN;
            return sent > 0 ? (ssize_t)sent : -1;
        }
        if (e == SSL_ERROR_SYSCALL && errno == EINTR) continue;
        return -1;
    }
    return (ssize_t)sent;
}

int shakti_tls_pending(void *ssl) {
    if (!ssl) return 0;
    return SSL_pending((SSL *)ssl);
}

void shakti_tls_free(void *ssl) {
    if (!ssl) return;
    SSL_shutdown((SSL *)ssl);
    SSL_free((SSL *)ssl);
}

void shakti_tls_close(void *ssl, int fd) {
    shakti_tls_free(ssl);
    if (fd >= 0) close(fd);
}
