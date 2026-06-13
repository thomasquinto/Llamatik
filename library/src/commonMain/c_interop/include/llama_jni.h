#ifdef __cplusplus
extern "C" {
#endif

bool llama_embed_init(const char *model_path);
float *llama_embed(const char *input_text);
int llama_embedding_size();
void llama_free_embedding(float *embedding);
bool llama_generate_init(const char *model_path);
char *llama_generate(const char *prompt);
void llama_generate_free();
void llama_free_cstr(char *p);

// Streaming callback types (iOS)
typedef void (*llm_on_delta)(const char *utf8, void *user);
typedef void (*llm_on_done)(void *user);
typedef void (*llm_on_error)(const char *utf8, void *user);

// Chat template streaming with message array (iOS)
void llama_generate_messages_stream(const char **roles,
        const char **contents,
        int n_messages,
        llm_on_delta on_delta,
        llm_on_done on_done,
        llm_on_error on_error,
        void *user);

bool llama_vision_available(void);
bool llama_vision_init(const char *model_path, const char *projection_model_path);
void llama_vision_generate_messages_stream(const char **roles,
        const char **contents,
        int n_messages,
        const char **image_paths,
        int n_images,
        llm_on_delta on_delta,
        llm_on_done on_done,
        llm_on_error on_error,
        void *user);

#ifdef __cplusplus
}
#endif
