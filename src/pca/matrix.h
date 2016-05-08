#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <OpenCL/opencl.h>
#include "bmp.h"

typedef struct {
  cl_float3 * entries;
  int rows;
  int cols;
} matrix_t;

matrix_t * matrix_for_image_rows(bmp_t ** images, size_t count);
matrix_t * matrix_transpose(matrix_t * mat);
void matrix_free(matrix_t * mat);

#endif
