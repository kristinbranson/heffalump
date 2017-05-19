#ifndef H_NGC_MINGLMEX_APP
#define H_NGC_MINGLMEX_APP

#ifdef  __cplusplus
extern "C" {
#endif

/// The api functions are used by the caller (e.g. main.c) which is responsible
/// for implementing the main loop and making sure the right inputs/events get
/// mapped to the right behaviors.

const char* app_version();

void app_init(
    void(*log)(int is_error,const char *file,int line,const char* function,const char *fmt,...));
int  app_is_running();
void app_wait_till_close();
void app_teardown();

double app_uptime_s();




// layered window

struct Layer {
    void(*draw)();
    void(*resize)(int w,int h);
    struct Layer *next;
    int added;
};

void window_add_layer(struct Layer* layer);

#ifdef  __cplusplus
}
#endif
#endif /* ifndef H_NGC_MINGLMEX_APP */

