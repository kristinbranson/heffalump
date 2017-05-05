#pragma 
#ifndef H_NGC_HOG
#define H_NGC_HOG

#ifdef __cplusplus
extern "C" {
#endif


enum hog_scalar_type {
    hog_u8,
    hog_u16,
    hog_u32,
    hog_u64,
    hog_i8,
    hog_i16,
    hog_i32,
    hog_i64,
    hog_f32,
    hog_f64,
};

struct hog_parameters {
    int patch,cell,nbins;
};

struct hog_context {
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...);
};

void hog_init(struct hog_context *context);

void hog(const struct hog_parameters params, )


#ifdef __cplusplus
}
#endif

#endif