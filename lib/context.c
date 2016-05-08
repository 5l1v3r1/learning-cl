#include "context.h"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#ifndef PRINT_PROGRAM_LOG
#define PRINT_PROGRAM_LOG 1
#endif

static context_t * allocate_context(context_params_t * params);

context_t * context_create(context_params_t * params) {
  cl_uint resultCount;
  cl_int statusCode;

  cl_platform_id platform;
  if (clGetPlatformIDs(1, &platform, &resultCount) || resultCount != 1) {
    return NULL;
  }

  cl_device_id devices[10];
  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 10, devices, &resultCount)) {
    return NULL;
  } else if (resultCount == 0) {
    return NULL;
  }
  cl_device_id device = devices[resultCount - 1];

  context_t * ctx = allocate_context(params);

  ctx->platform = platform;
  ctx->device = device;

  ctx->context = clCreateContext(0, 1, &device, NULL, NULL, &statusCode);
  if (statusCode) {
    context_free(ctx);
    return NULL;
  }

  ctx->queue = clCreateCommandQueue(ctx->context, device, 0, &statusCode);
  if (statusCode) {
    context_free(ctx);
    return NULL;
  }

  size_t programLen = strlen(params->program);
  ctx->program = clCreateProgramWithSource(ctx->context, 1, &params->program,
    &programLen, &statusCode);
  if (statusCode) {
    context_free(ctx);
    return NULL;
  }

  if (clBuildProgram(ctx->program, 1, &device, NULL, NULL, NULL)) {
    if (PRINT_PROGRAM_LOG) {
      size_t logSize;
      clGetProgramBuildInfo(ctx->program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

      char * logInfo = (char *)malloc(logSize);
      clGetProgramBuildInfo(ctx->program, device, CL_PROGRAM_BUILD_LOG, logSize, logInfo, NULL);

      printf("%s\n", logInfo);
      free(logInfo);
    }

    context_free(ctx);
    return NULL;
  }

  for (size_t i = 0; i < params->kernelCount; ++i) {
    ctx->kernels[i] = clCreateKernel(ctx->program, params->kernelNames[i], &statusCode);
    if (statusCode) {
      context_free(ctx);
      return NULL;
    }
    ++(ctx->kernelCount);
  }

  for (size_t i = 0; i < params->bufferCount; ++i) {
    size_t size = params->bufferSizes[i];
    ctx->bufferSizes[i] = size;
    ctx->buffers[i] = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, size, NULL, &statusCode);
    if (statusCode) {
      context_free(ctx);
      return NULL;
    }
    ++(ctx->bufferCount);
  }

  return ctx;
}

int context_set_params(context_t * ctx, int kernelIdx, size_t count,
                       void ** params, size_t * sizes) {
  cl_kernel kernel = ctx->kernels[kernelIdx];
  for (size_t i = 0; i < count; ++i) {
    if (clSetKernelArg(kernel, i, sizes[i], params[i])) {
      return -1;
    }
  }
  return 0;
}

void * context_map(context_t * ctx, int bufIdx, cl_bool write) {
  cl_map_flags flags = CL_MAP_READ;
  if (write) {
    flags |= CL_MAP_WRITE;
  }
  return clEnqueueMapBuffer(ctx->queue, ctx->buffers[bufIdx], CL_TRUE, flags, 0,
    ctx->bufferSizes[bufIdx], 0, NULL, NULL, NULL);
}

void context_unmap(context_t * ctx, int bufIdx, void * ptr) {
  cl_event event;
  if (clEnqueueUnmapMemObject(ctx->queue, ctx->buffers[bufIdx], ptr, 0, NULL, &event)) {
    return;
  }
  clWaitForEvents(1, &event);
  clReleaseEvent(event);
}

int context_run_nd(context_t * ctx, int kernelIdx, size_t dim, size_t * offsets, size_t * sizes) {
  cl_event event;
  if (clEnqueueNDRangeKernel(ctx->queue, ctx->kernels[kernelIdx], dim, offsets, sizes,
      NULL, 0, NULL, &event)) {
    return -1;
  }

  cl_int res = clWaitForEvents(1, &event);
  clReleaseEvent(event);
  if (res) {
    return -1;
  } else {
    return 0;
  }
}

void context_free(context_t * ctx) {
  if (ctx->queue) {
    clFlush(ctx->queue);
    clFinish(ctx->queue);
  }
  for (size_t i = 0; i < ctx->kernelCount; ++i) {
    clReleaseKernel(ctx->kernels[i]);
  }
  if (ctx->program) {
    clReleaseProgram(ctx->program);
  }
  for (size_t i = 0; i < ctx->bufferCount; ++i) {
    clReleaseMemObject(ctx->buffers[i]);
  }
  if (ctx->queue) {
    clReleaseCommandQueue(ctx->queue);
  }
  if (ctx->context) {
    clReleaseContext(ctx->context);
  }
  free(ctx->kernels);
  free(ctx->buffers);
  free(ctx->bufferSizes);
  free(ctx);
}

static context_t * allocate_context(context_params_t * params) {
  context_t * res = (context_t *)malloc(sizeof(context_t));
  if (!res) {
    return NULL;
  }
  bzero(res, sizeof(context_t));

  res->kernels = (cl_kernel *)malloc(sizeof(cl_kernel) * params->kernelCount);
  if (!res->kernels) {
    free(res);
    return NULL;
  }
  bzero(res->kernels, sizeof(cl_kernel) * params->kernelCount);

  res->buffers = (cl_mem *)malloc(sizeof(cl_mem) * params->bufferCount);
  if (!res->buffers) {
    free(res->kernels);
    free(res);
    return NULL;
  }
  bzero(res->buffers, sizeof(cl_mem) * params->bufferCount);

  res->bufferSizes = (size_t *)malloc(sizeof(size_t) * params->bufferCount);
  if (!res->bufferSizes) {
    free(res->buffers);
    free(res->kernels);
    free(res);
    return NULL;
  }
  bzero(res->bufferSizes, sizeof(size_t) * params->bufferCount);

  return res;
}
