#ifndef SHAKTI_JSON_PARSE_H
#define SHAKTI_JSON_PARSE_H

/* Deepest nesting of arrays and objects that json_loads reads and json_dumps writes. */
#ifndef SHAKTI_JSON_MAX_DEPTH
#define SHAKTI_JSON_MAX_DEPTH 512
#endif

struct V;
struct V *shakti_json_parse(const char *s, const char **end_out);

#endif
