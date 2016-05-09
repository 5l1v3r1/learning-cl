#include "power_iter.h"
#include <math.h>
#include <string.h>

#define ROW_MATRIX_BUFF 0
#define COL_MATRIX_BUFF 1
#define ROW_OUTPUT_BUFF 2
#define COL_OUTPUT_BUFF 3

#define ROW_MULT_KERNEL 0
#define COL_MULT_KERNEL 1

static cl_float random_float();
static int write_output_vector(power_iter_t * iter);
static int read_output_vector(power_iter_t * iter);
static void normalize_output(power_iter_t * iter);

static const char * multProgram = "\
__kernel void apply(__global float3 * mat, int cols, \
                    __global float3 * input, __global float3 * output) { \
  int row = get_global_id(0); \
  __global float3 * matRow = &mat[cols * row]; \
  float3 result = 0; \
  for (int i = 0; i < cols; ++i) { \
    result += matRow[i] * input[i]; \
  } \
  output[row] = result; \
} \
";

power_iter_t * power_iter_new(matrix_t * rowMat) {
  const char * kernelNames[2] = {"apply", "apply"};
  size_t matrixSize = rowMat->cols * rowMat->rows * sizeof(cl_float3);
  size_t outputSize1 = rowMat->rows * sizeof(cl_float3);
  size_t outputSize2 = rowMat->cols * sizeof(cl_float3);
  size_t bufferSizes[4] = {matrixSize, matrixSize, outputSize1, outputSize2};

  context_params_t params;
  params.program = multProgram;
  params.kernelCount = 2;
  params.kernelNames = kernelNames;
  params.bufferCount = 4;
  params.bufferSizes = bufferSizes;
  context_t * ctx = context_create(&params);
  if (!ctx) {
    return NULL;
  }

  void * mappedBuff = context_map(ctx, ROW_MATRIX_BUFF, CL_TRUE);
  if (!mappedBuff) {
    context_free(ctx);
    return NULL;
  }
  memcpy(mappedBuff, rowMat->entries, matrixSize);
  context_unmap(ctx, ROW_MATRIX_BUFF, mappedBuff);

  matrix_t * colMat = matrix_transpose(rowMat);
  if (!colMat) {
    context_free(ctx);
    return NULL;
  }

  mappedBuff = context_map(ctx, COL_MATRIX_BUFF, CL_TRUE);
  if (!mappedBuff) {
    matrix_free(colMat);
    context_free(ctx);
    return NULL;
  }
  memcpy(mappedBuff, colMat->entries, matrixSize);
  matrix_free(colMat);
  context_unmap(ctx, COL_MATRIX_BUFF, mappedBuff);

  cl_int cols = rowMat->cols;
  cl_int rows = rowMat->rows;
  void * args[4] = {&ctx->buffers[ROW_MATRIX_BUFF], &cols, &ctx->buffers[COL_OUTPUT_BUFF],
    &ctx->buffers[ROW_OUTPUT_BUFF]};
  size_t argSizes[4] = {sizeof(cl_mem), sizeof(cl_int), sizeof(cl_mem), sizeof(cl_mem)};
  if (context_set_params(ctx, ROW_MULT_KERNEL, 4, args, argSizes)) {
    context_free(ctx);
    return NULL;
  }
  args[0] = &ctx->buffers[COL_MATRIX_BUFF];
  args[1] = &rows;
  args[2] = &ctx->buffers[ROW_OUTPUT_BUFF];
  args[3] = &ctx->buffers[COL_OUTPUT_BUFF];
  if (context_set_params(ctx, COL_MULT_KERNEL, 4, args, argSizes)) {
    context_free(ctx);
    return NULL;
  }

  power_iter_t * res = (power_iter_t *)malloc(sizeof(power_iter_t));
  res->context = ctx;
  res->vectorSize = rowMat->cols;
  res->intermediateSize = rowMat->rows;
  res->vector = (cl_float3 *)malloc(sizeof(cl_float3) * res->vectorSize);
  if (!res->vector) {
    free(res);
    context_free(ctx);
    return NULL;
  }
  for (size_t i = 0; i < res->vectorSize; ++i) {
    cl_float3 r;
    r.s[0] = random_float();
    r.s[1] = random_float();
    r.s[2] = random_float();
    res->vector[i] = r;
  }

  return res;
}

int power_iter_run(power_iter_t * iter, int iterations) {
  if (write_output_vector(iter)) {
    return -1;
  }

  for (int i = 0; i < iterations; ++i) {
    size_t outputSize = iter->intermediateSize;
    if (context_run_nd(iter->context, ROW_MULT_KERNEL, 1, NULL, &outputSize)) {
      return -1;
    }
    outputSize = iter->vectorSize;
    if (context_run_nd(iter->context, COL_MULT_KERNEL, 1, NULL, &outputSize)) {
      return -1;
    }
  }

  if (read_output_vector(iter)) {
    return -1;
  }

  normalize_output(iter);

  return 0;
}

void power_iter_free(power_iter_t * iter) {
  context_free(iter->context);
  free(iter->vector);
  free(iter);
}

static cl_float random_float() {
  return ((cl_float)(rand() % 1025) / 1024.0)*2 - 1;
}

static int write_output_vector(power_iter_t * iter) {
  cl_float3 * mappedInput = (cl_float3 *)context_map(iter->context, COL_OUTPUT_BUFF, CL_TRUE);
  if (!mappedInput) {
    return -1;
  }
  memcpy(mappedInput, iter->vector, iter->context->bufferSizes[COL_OUTPUT_BUFF]);
  context_unmap(iter->context, COL_OUTPUT_BUFF, mappedInput);
  return 0;
}

static int read_output_vector(power_iter_t * iter) {
  cl_float3 * mappedInput = (cl_float3 *)context_map(iter->context, COL_OUTPUT_BUFF, CL_FALSE);
  if (!mappedInput) {
    return -1;
  }
  for (size_t i = 0; i < iter->vectorSize; ++i) {
    iter->vector[i] = mappedInput[i];
  }
  context_unmap(iter->context, COL_OUTPUT_BUFF, mappedInput);
  return 0;
}

static void normalize_output(power_iter_t * iter) {
  for (size_t i = 0; i < 3; ++i) {
    cl_float mag = 0;
    for (size_t j = 0; j < iter->vectorSize; ++j) {
      cl_float val = iter->vector[j].s[i];
      mag += val * val;
    }
    cl_float recip = 1.0f / sqrtf(mag);
    for (size_t j = 0; j < iter->vectorSize; ++j) {
      iter->vector[j].s[i] *= recip;
    }
  }
}
