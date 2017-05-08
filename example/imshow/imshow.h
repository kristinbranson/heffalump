#pragma once
#ifndef H_NGC_IMSHOW
#define H_NGC_IMSHOW

#ifdef __cplusplus
#extern "C" {
#endif

    enum imshow_scalar_type {
        imshow_u8,
        imshow_u16,
        imshow_u32,
        imshow_u64,
        imshow_i8,
        imshow_i16,
        imshow_i32,
        imshow_i64,
        imshow_f32,
        imshow_f64
    };

    void imshow(enum imshow_scalar_type type,int w, int h,const void *data);
    void imshow_contrast(enum imshow_scalar_type type,float min,float max);
    void imshow_viewport(int w,int h);

#ifdef __cplusplus
}
#endif

#endif


/* NOTES
 * 
 * All of these operate on the currently active window.
 * See app.h for creating/manipulating windows.
 *
 * TODO
 *
 * Consider case when changes are being pushed from another
 * thread:  We only want the gl calls to happen in a blessed
 * thread.  Need to aggregate state changes and commit them on
 * the appropriate update/draw pass by the blessed thread. This
 * may require buffering a frame; need some definition of data ptr
 * ownership/lifetime.
 */