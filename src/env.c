/* shakti/src/env.c — environments, fn_ast, interpreter globals */
#include "shakti_internal.h"

Node *fn_ast[MAX_FN];
int   fn_ast_n = 0;
static int fn_ast_refs[MAX_FN];
static int fn_ast_free_stack[MAX_FN];
static int fn_ast_free_n;
int   g_returning  = 0;
int   g_breaking   = 0;
int   g_continuing = 0;
int   g_error      = 0;
V    *g_retval     = NULL;
V    *g_error_val  = NULL;
char  g_lib_path[4096] = "";
char  g_script_dir[4096] = ".";
int fn_ast_store(Node *n) {
    if (!n) return -1;
    if (n->fn_ast_i > -1) {
        if (n->fn_ast_i < fn_ast_n) fn_ast_refs[n->fn_ast_i]++;
        return n->fn_ast_i;
    }
    int idx;
    if (fn_ast_free_n > 0)
        idx = fn_ast_free_stack[--fn_ast_free_n];
    else {
        if (fn_ast_n >= MAX_FN) return -1;
        idx = fn_ast_n++;
    }
    fn_ast[idx] = n;
    fn_ast_refs[idx] = 1;
    n->fn_ast_i = idx;
    n->fn_detached = 0;
    return idx;
}
void fn_ast_release(int idx) {
    Node *n;
    if (idx < 0 || idx >= fn_ast_n) return;
    if (fn_ast_refs[idx] <= 0) return;
    if (--fn_ast_refs[idx] > 0) return;
    n = fn_ast[idx];
    fn_ast[idx] = NULL;
    if (fn_ast_free_n < MAX_FN)
        fn_ast_free_stack[fn_ast_free_n++] = idx;
    if (!n) return;
    if (n->fn_detached) {
        n->fn_ast_i = -1;
        node_free(n);
    } else {
        n->fn_ast_i = -2;
    }
}
Env *env_new(Env *parent) {
    Env *e = x_calloc(1, sizeof(Env), "env_new");
    e->rc = 1; e->cap = 16; e->len = 0;
    e->names  = x_calloc(16, sizeof(char*), "env_new");
    e->vals   = x_calloc(16, sizeof(V*), "env_new");
    e->hashes = x_calloc(16, sizeof(uint32_t), "env_new");
    e->kinds  = x_calloc(16, sizeof(uint8_t), "env_new");
    e->parent = parent;
    if(parent) parent->rc++;
    return e;
}
#define SHAKTI_ENV_POOL_MAX 64
static Env *shakti_env_pool[SHAKTI_ENV_POOL_MAX];
static int shakti_env_pool_n;
Env *env_acquire(Env *parent) {
    Env *e;
    if(shakti_env_pool_n > 0) {
        e = shakti_env_pool[--shakti_env_pool_n];
        for(int i = 0; i < e->len; i++) {
            free(e->names[i]);
            v_free(e->vals[i]);
        }
        e->len = 0;
        e->rc = 1;
        e->parent = parent;
        if(parent) parent->rc++;
        return e;
    }
    return env_new(parent);
}
void env_release(Env *e) {
    if(!e) return;
    if(e->rc != 1 || shakti_env_pool_n >= SHAKTI_ENV_POOL_MAX) {
        env_free(e);
        return;
    }
    Env *parent = e->parent;
    e->parent = NULL;
    if(parent) env_free(parent);
    for(int i = 0; i < e->len; i++) {
        free(e->names[i]);
        v_free(e->vals[i]);
        e->names[i] = NULL;
        e->vals[i] = NULL;
    }
    e->len = 0;
    e->rc = 1;
    shakti_env_pool[shakti_env_pool_n++] = e;
}
void env_set_kind(Env *e, const char *name, V *val, unsigned kind) {
    uint32_t h = fnv1a(name);
    for(int i=0; i<e->len; i++) {
        if(e->hashes[i] == h && strcmp(e->names[i], name)==0) {
            v_free(e->vals[i]);
            e->vals[i] = v_ref(val);
            e->kinds[i] = (uint8_t)kind;
            return;
        }
    }
    if(e->len >= e->cap) {
        e->cap *= 2;
        e->names  = x_realloc(e->names,  e->cap * sizeof(char*), "env_set");
        e->vals   = x_realloc(e->vals,   e->cap * sizeof(V*), "env_set");
        e->hashes = x_realloc(e->hashes, e->cap * sizeof(uint32_t), "env_set");
        e->kinds  = x_realloc(e->kinds,  e->cap * sizeof(uint8_t), "env_set");
    }
    e->names[e->len]  = x_strdup(name, "env_set");
    e->vals[e->len]   = v_ref(val);
    e->hashes[e->len] = h;
    e->kinds[e->len]  = (uint8_t)kind;
    e->len++;
}
void env_set(Env *e, const char *name, V *val) {
    env_set_kind(e, name, val, ENV_BIND_USER);
}
int env_slot_listed(const Env *e, int i) {
    const char *n;
    if(!e || i < 0 || i >= e->len) return 0;
    if(e->kinds && e->kinds[i] != ENV_BIND_USER) return 0;
    n = e->names[i];
    if(n && n[0] == '_' && n[1] == '_') return 0;
    return 1;
}
/* Mutate an existing INT binding in place when uniquely owned (rc==1). */
int env_set_int_inplace(Env *e, const char *name, int64_t j) {
    uint32_t h = fnv1a(name);
    for(; e; e=e->parent)
        for(int i=0; i<e->len; i++)
            if(e->hashes[i] == h && strcmp(e->names[i], name)==0) {
                V *cur = e->vals[i];
                if(cur && cur->t == T_INT && cur->rc == 1) {
                    cur->j = j;
                    return 1;
                }
                V *nv = v_int(j);
                v_free(cur);
                e->vals[i] = nv;
                return 1;
            }
    return 0;
}
int env_get_int(Env *e, const char *name, int64_t *out) {
    V *v = env_get(e, name);
    if(!v || v->t != T_INT) return 0;
    *out = v->j;
    return 1;
}
void env_set_local(Env *e, const char *name, V *val) {
    env_set(e, name, val);
}
int env_update(Env *e, const char *name, V *val) {
    uint32_t h = fnv1a(name);
    for(; e; e=e->parent)
        for(int i=0; i<e->len; i++)
            if(e->hashes[i] == h && strcmp(e->names[i], name)==0) {
                if (e->vals[i] != val) {
                    v_free(e->vals[i]);
                    e->vals[i] = v_ref(val);
                }
                return 1;
            }
    return 0;
}
V *env_get_here(Env *e, const char *name) {
    uint32_t h;
    int i;
    if (!e || !name) return NULL;
    h = fnv1a(name);
    for (i = 0; i < e->len; i++)
        if (e->hashes[i] == h && strcmp(e->names[i], name) == 0)
            return e->vals[i];
    return NULL;
}
V *env_get(Env *e, const char *name) {
    uint32_t h = fnv1a(name);
    for(; e; e=e->parent)
        for(int i=0; i<e->len; i++)
            if(e->hashes[i] == h && strcmp(e->names[i], name)==0)
                return e->vals[i];
    return NULL;
}
void env_ref(Env *e) { if(e) e->rc++; }
void env_free(Env *e) {
    Pv(!e)
    Pv(--e->rc > 0)
    i(e->len,{free(e->names[i]); v_free(e->vals[i]);})
    free(e->names); free(e->vals); free(e->hashes); free(e->kinds);
    if(e->parent) env_free(e->parent);
    free(e);
}
int env_save(Env *e, const char *path) {
    FILE *fp = fopen(path, "wb");
    P(!fp,0)
    fwrite("KAIO_MCP", 1, 8, fp);
    fwrite(&e->len, 4, 1, fp);
    i(e->len,{
        int nlen = strlen(e->names[i]);
        fwrite(&nlen, 4, 1, fp);
        fwrite(e->names[i], 1, nlen, fp);
        v_serialize(e->vals[i], fp);
    })
    fclose(fp);
    return 1;
}
int env_load(Env *e, const char *path) {
    FILE *fp = fopen(path, "rb");
    P(!fp,0)
    char hdr[8];
    if (fread(hdr, 1, 8, fp) != 8 || memcmp(hdr, "KAIO_MCP", 8)) { fclose(fp); return 0; }
    int count;
    if (fread(&count, 4, 1, fp) != 1 || count < 0 || count > SHAKTI_DESER_MAX_VARS) { fclose(fp); return 0; }
    for (int i = 0; i < count; i++) {
        int nlen;
        if (fread(&nlen, 4, 1, fp) != 1 || nlen < 0 || nlen > SHAKTI_ENV_MAX_NAME) { fclose(fp); return 0; }
        char *name = malloc((size_t)nlen + 1);
        if (!name) { fclose(fp); return 0; }
        if (nlen > 0 && fread(name, 1, (size_t)nlen, fp) != (size_t)nlen) { free(name); fclose(fp); return 0; }
        name[nlen] = 0;
        V *val = v_deserialize(fp);
        if (val->t == T_ERR) { free(name); v_free(val); fclose(fp); return 0; }
        env_set(e, name, val);
        v_free(val);
        free(name);
    }
    fclose(fp);
    return 1;
}

static int env_flag_on(const char *name) {
    const char *v = getenv(name);
    return v && v[0] && strcmp(v, "0") != 0;
}
int shakti_safe_mode(void) {
    return env_flag_on("SHAKTI_SAFE");
}
int shakti_exec_allowed(void) {
    const char *allow = getenv("SHAKTI_ALLOW_EXEC");
    if (shakti_safe_mode()) return 0;
    return !(allow && allow[0] == '0' && allow[1] == '\0');
}
int shakti_net_allowed(void) {
    return !shakti_safe_mode() || env_flag_on("SHAKTI_ALLOW_NET");
}
/* Safe mode confines file access to SHAKTI_FS_ROOT (default: the working
 * directory at first use). Paths are resolved with realpath, so symlinks that
 * point outside the root are rejected. */
int shakti_fs_check(const char *path, int write, char *err, size_t err_cap) {
#ifdef _WIN32
    (void)path; (void)write; (void)err; (void)err_cap;
    return 0;
#else
    static char root[4096];
    char full[8192];
    if (!shakti_safe_mode()) return 0;
    if (!path || !path[0]) {
        if (err && err_cap) snprintf(err, err_cap, "safe mode: empty path");
        return -1;
    }
    if (!root[0]) {
        const char *r = getenv("SHAKTI_FS_ROOT");
        char *rp = realpath(r && r[0] ? r : ".", NULL);
        if (!rp) {
            if (err && err_cap) snprintf(err, err_cap, "safe mode: bad SHAKTI_FS_ROOT");
            return -1;
        }
        snprintf(root, sizeof root, "%s", rp);
        free(rp);
    }
    char *rp = realpath(path, NULL);
    if (rp) {
        snprintf(full, sizeof full, "%s", rp);
        free(rp);
    } else {
        char dir[8192];
        const char *slash = strrchr(path, '/');
        const char *base = slash ? slash + 1 : path;
        if (slash == path) snprintf(dir, sizeof dir, "/");
        else if (slash) snprintf(dir, sizeof dir, "%.*s", (int)(slash - path), path);
        else snprintf(dir, sizeof dir, ".");
        if (!base[0] || !strcmp(base, ".") || !strcmp(base, "..")) {
            if (err && err_cap) snprintf(err, err_cap, "safe mode: bad path '%s'", path);
            return -1;
        }
        char *dp = realpath(dir, NULL);
        if (!dp) {
            if (err && err_cap) snprintf(err, err_cap, "safe mode: cannot resolve '%s'", path);
            return -1;
        }
        snprintf(full, sizeof full, "%s/%s", dp, base);
        free(dp);
    }
    size_t rl = strlen(root);
    int inside = !strcmp(root, "/") ||
                 (!strncmp(full, root, rl) && (full[rl] == 0 || full[rl] == '/'));
    if (!inside) {
        if (err && err_cap)
            snprintf(err, err_cap, "safe mode: %s outside SHAKTI_FS_ROOT: %s", write ? "write" : "read", path);
        return -1;
    }
    return 0;
#endif
}
