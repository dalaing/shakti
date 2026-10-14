/* shakti/src/shakti_internal.h — private cross-TU helpers (not public API) */
#ifndef SHAKTI_INTERNAL_H
#define SHAKTI_INTERNAL_H

#include "shakti.h"

#ifndef SHAKTI_PARSE_MAX_DEPTH
#define SHAKTI_PARSE_MAX_DEPTH 40
#endif
#ifndef SHAKTI_EXPR_MAX_DEPTH
#define SHAKTI_EXPR_MAX_DEPTH  4096
#endif
#ifndef SHAKTI_CALL_MAX_DEPTH
#define SHAKTI_CALL_MAX_DEPTH  3000
#endif
#ifndef SHAKTI_DESER_MAX_DEPTH
#define SHAKTI_DESER_MAX_DEPTH 512
#endif
#ifndef SHAKTI_DESER_MAX_LEN
#define SHAKTI_DESER_MAX_LEN   (64 * 1024 * 1024)
#endif
#ifndef SHAKTI_DESER_MAX_VARS
#define SHAKTI_DESER_MAX_VARS  65536
#endif
#ifndef SHAKTI_ENV_MAX_NAME
#define SHAKTI_ENV_MAX_NAME    4096
#endif
#ifndef SHAKTI_PRINT_MAX_DEPTH
#define SHAKTI_PRINT_MAX_DEPTH 512
#endif
/* Copying (v_copy, list()) and comparing (=, !=, in) a value that holds itself would go round
 * it until the stack ran out; they give this error value, v_loop_err(), instead. */
#define SHAKTI_LOOP_ERR "a value holds itself"

#ifdef _WIN32
#define SHAKTI_TIMEGM(tm) _mkgmtime(tm)
#else
#define SHAKTI_TIMEGM(tm) timegm(tm)
#endif

extern char g_lib_path[4096];
extern char g_script_dir[4096];
void shakti_set_file_vars(Env *e, const char *path);
int shakti_safe_mode(void);
int shakti_exec_allowed(void);
int shakti_net_allowed(void);
int shakti_fs_check(const char *path, int write, char *err, size_t err_cap);

void shakti_oom(const char *where);
void *x_malloc(size_t sz, const char *where);
void *x_calloc(size_t nmemb, size_t sz, const char *where);
void *x_realloc(void *ptr, size_t sz, const char *where);
char *x_strdup(const char *s, const char *where);
size_t x_mul(size_t a, size_t b, const char *where);
uint32_t fnv1a(const char *s);

V *v_alloc(int t);
V *try_promote_matrix(V **elems, int nch);
void print_val(V *v, FILE *fp, int repr_mode);
void v_serialize(V *v, FILE *fp);
/* The error value SHAKTI_LOOP_ERR, marked so that v_is_loop_err tells it from any other
 * error (in gives it back, where an item that doesn't compare is just not a match). */
V *v_loop_err(void);
int v_is_loop_err(const V *v);
/* A walk that recurses through lists, dicts and tables (v_copy, vec_cmp) goes on forever just
 * when it comes back to a step it is already inside: the value holds itself. LoopWalk spots
 * that without keeping the whole path, so a deep value costs little more than a wide one: the
 * step entered at each power-of-two depth is saved (1, 2, 4, ...: always steps the walk is
 * inside), and each step entered is checked against those, a few dozen at most. A walk that
 * goes round forever comes back to a step entered at depth m after r more steps; the first
 * saved depth at or past m is less than twice m, so it is caught by depth 2m + r (at depth r + 1
 * when it comes back to the start). A walk that doesn't go round is never stopped, however deep
 * it goes. A step is a value, or for a comparison two values and an operator. loop_enter says
 * whether the step is one the walk is already inside; each loop_enter is matched by a
 * loop_leave. */
typedef struct {
    int64_t depth;
    int level;
    struct { const void *a, *b; int op; } saved[64];
} LoopWalk;
static inline int loop_enter(LoopWalk *w, const void *a, const void *b, int op) {
    int64_t d = ++w->depth;
    int seen = 0;
    for (int k = w->level - 1; k >= 0 && !seen; k--)
        seen = w->saved[k].a == a && w->saved[k].b == b && w->saved[k].op == op;
    if (!(d & (d - 1))) {
        w->saved[w->level].a = a; w->saved[w->level].b = b; w->saved[w->level].op = op;
        w->level++;
    }
    return seen;
}
static inline void loop_leave(LoopWalk *w) {
    if (!(w->depth & (w->depth - 1))) w->level--;
    w->depth--;
}
V *v_deserialize(FILE *fp);

int is_truthy(V *v);
double to_float(V *v);
V *vec_logic(V *a, V *b, int is_and);
V *vec_binop(V *a, V *b, int op);
V *mat_binop(V *a, V *b, int op);
V *table_filter(V *tbl, V *mask);

void lex_fail(Lexer *l, const char *message);
int lex_peek_is_signed_literal(Lexer *l);
int lex_peek_minus_is_subtraction(Lexer *l);
void parse_fail(const char *fmt, ...);

void node_sprint(Node *n, FILE *fp);

Env *env_acquire(Env *parent);
void env_release(Env *e);
int env_set_int_inplace(Env *e, const char *name, int64_t j);
int env_get_int(Env *e, const char *name, int64_t *out);
int env_update(Env *e, const char *name, V *val);

FILE *fopen_regular(const char *path, char *err, size_t err_cap);
V *do_import(const char *name, Env *e);
V *require_sql(Env *e);

/* Call-site `*expr` parses as N_UNOP with this op. */
#define OP_SPLAT OP_MUL

void shakti_bind_params(Env *ce, V *fn, V **args, int nargs, V **kwn, V **kwv, int nkw);
V *shakti_call_fn(V *fn, V *self, V **args, int nargs, V **kwn, V **kwv, int nkw, Env *e);
int shakti_is_class(V *v);
int shakti_is_instance(V *v);
V *shakti_instance_str(V *v, Env *e);
int shakti_isinstance(V *obj, V *cls);

int shakti_prog_silent_last(Node *prog);
void run_repl(Env *e);

#endif /* SHAKTI_INTERNAL_H */
