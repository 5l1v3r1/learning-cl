#include "matrix.h"
#include <assert.h>

matrix_t * matrix_for_image_rows(bmp_t ** images, size_t count) {
  matrix_t * mat = (matrix_t *)malloc(sizeof(matrix_t));
  if (!mat) {
    return NULL;
  }

  size_t pixelCount = images[0]->width * images[0]->height;
  mat->entries = (cl_float3 *)malloc(sizeof(cl_float3) * count * pixelCount);
  if (!mat->entries) {
    free(mat);
    return NULL;
  }

  mat->rows = count;
  mat->cols = pixelCount;

  size_t entryIdx = 0;
  for (size_t row = 0; row < count; ++row) {
    bmp_t * image = images[row];
    assert(image->width * image->height == pixelCount);
    for (size_t col = 0; col < pixelCount; ++col) {
      cl_uchar4 pixel = image->pixels[col];
      cl_float3 entry;
      entry.s[0] = (cl_float)pixel.s[0];
      entry.s[1] = (cl_float)pixel.s[1];
      entry.s[2] = (cl_float)pixel.s[2];
      mat->entries[entryIdx++] = entry;
    }
  }

  return mat;
}

matrix_t * matrix_transpose(matrix_t * mat) {
  matrix_t * trans = (matrix_t *)malloc(sizeof(matrix_t));
  if (!trans) {
    return NULL;
  }
  trans->entries = (cl_float3 *)malloc(sizeof(cl_float3) * mat->rows * mat->cols);
  if (!trans->entries) {
    free(trans);
    return NULL;
  }
  trans->rows = mat->cols;
  trans->cols = mat->rows;
  size_t destIdx = 0;
  for (size_t row = 0; row < mat->cols; ++row) {
    for (size_t col = 0; col < mat->rows; ++col) {
      trans->entries[destIdx++] = mat->entries[col*mat->cols + row];
    }
  }
  return trans;
}

void matrix_free(matrix_t * mat) {
  free(mat->entries);
  free(mat);
}
