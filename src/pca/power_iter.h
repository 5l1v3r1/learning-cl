#ifndef __POWER_ITER_H__
#define __POWER_ITER_H__

#include "matrix.h"
#include "context.h"

typedef struct {
  context_t * context;
  cl_float3 * vector;
  size_t vectorSize;
  size_t intermediateSize;
} power_iter_t;

// power_iter_new creates a new power iterator
// which applies rowMat'*rowMat to a vector.
power_iter_t * power_iter_new(matrix_t * rowMat);
int power_iter_run(power_iter_t * iter, int iterations);
void power_iter_free(power_iter_t * iter);

#endif
