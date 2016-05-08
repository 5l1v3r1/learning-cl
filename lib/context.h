#ifndef __CONTEXT_H__
#define __CONTEXT_H__

#include <OpenCL/opencl.h>

typedef struct {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;

  size_t kernelCount;
  cl_kernel * kernels;

  size_t bufferCount;
  size_t * bufferSizes;
  cl_mem * buffers;
} context_t;

typedef struct {
  const char * program;
  size_t kernelCount;
  const char ** kernelNames;

  size_t bufferCount;
  size_t * bufferSizes;
} context_params_t;

context_t * context_create(context_params_t * params);
int context_set_params(context_t * ctx, int kernelIdx, size_t count,
                       void ** params, size_t * sizes);
void * context_map(context_t * ctx, int bufIdx, cl_bool write);
void context_unmap(context_t * ctx, int bufIdx, void * ptr);
int context_run_nd(context_t * ctx, int kernelIdx, size_t dim, size_t * offsets, size_t * sizes);
void context_free(context_t * context);

#endif
