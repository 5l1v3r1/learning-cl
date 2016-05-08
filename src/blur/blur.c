#include "blur.h"
#include "context.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static cl_float * make_weights(int radius, cl_float sigma);
static context_t * create_blur_context(bmp_t * input, int radius, cl_float * weights);
static int run_blur_context(context_t * ctx, bmp_t * input, int radius);

int blur_image(bmp_t * image, int radius, cl_float sigma) {
  cl_float * weights = make_weights(radius, sigma);
  if (!weights) {
    return -1;
  }

  context_t * ctx = create_blur_context(image, radius, weights);
  free(weights);

  if (!ctx) {
    return -1;
  }

  if (run_blur_context(ctx, image, radius)) {
    context_free(ctx);
    return -1;
  }

  void * output = context_map(ctx, 1, CL_FALSE);
  if (!output) {
    context_free(ctx);
    return -1;
  }
  memcpy(image->pixels, output, ctx->bufferSizes[0]);
  context_unmap(ctx, 1, output);

  context_free(ctx);
  return 0;
}

static cl_float * make_weights(int radius, cl_float sigma) {
  size_t weightsSide = radius*2 + 1;
  size_t weightCount = weightsSide * weightsSide;
  cl_float * weights = (cl_float *)malloc(weightCount * sizeof(cl_float));
  if (!weights) {
    return NULL;
  }
  cl_float weightSum = 0;
  int weightIdx = 0;
  for (int y = -radius; y <= radius; ++y) {
    for (int x = -radius; x <= radius; ++x) {
      cl_float dist = (cl_float)(x*x + y*y);
      weights[weightIdx] = (cl_float)expf(-(float)dist / (2 * sigma * sigma));
      weightSum += weights[weightIdx];
      ++weightIdx;
    }
  }
  cl_float weightNorm = 1 / weightSum;
  while (weightIdx--) {
    weights[weightIdx] *= weightNorm;
  }
  return weights;
}

static const char * blurKernel = "\
__kernel void blur(__global uchar4 * input, __global uchar4 * output, \
                   __global float * weights, int radius, int width) { \
  /*int globalX = get_global_id(0); \
  int globalY = get_global_id(1); \
  int inputRow = globalX + (globalY-radius)*width; \
  int weightIdx = 0; \
  float4 outputFloat = 0; \
  for (int y = globalY-radius; y <= globalY+radius; ++y) { \
    for (int x = -radius; x <= radius; ++x) { \
      float4 fIn = convert_float4(input[inputRow + x]); \
      float weight = weights[weightIdx++]; \
      outputFloat += fIn * weight; \
    } \
    inputRow += width; \
  } \
  output[globalX + globalY*width] = convert_uchar4_sat(outputFloat);*/ \
} \
";

static const char * blurKernelName = "blur";

static context_t * create_blur_context(bmp_t * input, cl_int radius, cl_float * weights) {
  size_t bitmapSize = input->width * input->height * sizeof(cl_uchar4);
  size_t bufferSizes[3] = {bitmapSize, bitmapSize,
    (radius*2 + 1) * (radius*2 + 1) * sizeof(cl_float)};
  context_params_t params;
  params.program = blurKernel;
  params.kernelCount = 1;
  params.kernelNames = &blurKernelName;
  params.bufferCount = 3;
  params.bufferSizes = bufferSizes;

  context_t * ctx = context_create(&params);
  if (!ctx) {
    return NULL;
  }

  void * inputBuf = context_map(ctx, 0, CL_TRUE);
  if (!inputBuf) {
    context_free(ctx);
    return NULL;
  }
  memcpy(inputBuf, input->pixels, bitmapSize);
  context_unmap(ctx, 0, inputBuf);

  void * weightBuf = context_map(ctx, 2, CL_TRUE);
  if (!weightBuf) {
    context_free(ctx);
    return NULL;
  }
  memcpy(weightBuf, weights, bufferSizes[2]);
  context_unmap(ctx, 2, weightBuf);

  cl_int width = input->width;
  void * args[5] = {&ctx->buffers[0], &ctx->buffers[1], &ctx->buffers[2],
    &radius, &width};
  size_t sizes[5] = {sizeof(cl_mem), sizeof(cl_mem), sizeof(cl_mem),
    sizeof(cl_int), sizeof(cl_int)};
  if (context_set_params(ctx, 0, 5, args, sizes)) {
    context_free(ctx);
    return NULL;
  }

  return ctx;
}

static int run_blur_context(context_t * ctx, bmp_t * input, int radius) {
  size_t workSizes[2] = {input->width - radius*2, input->height - radius*2};
  size_t workOffsets[2] = {radius, radius};
  return context_run_nd(ctx, 0, 2, workOffsets, workSizes);
}
