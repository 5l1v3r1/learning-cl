#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include "bmp.h"
#include "matrix.h"
#include "power_iter.h"

#define MIN(x,y) (x < y ? x : y)
#define MAX(x,y) (-(MIN(-x,-y)))

bmp_t ** read_bitmaps(const char * dir, size_t * countOut);
void free_bitmaps(bmp_t ** bmps, size_t count);
bmp_t * vec_to_image(cl_float3 * vec, size_t width, size_t height);
void vec_to_image_chan(size_t chan, cl_float3 * vec, cl_uchar4 * out, size_t count);

int main(int argc, const char ** argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <face-db> <output.bmp>\n", argv[0]);
    return 1;
  }

  size_t bmpCount;
  bmp_t ** bitmaps = read_bitmaps(argv[1], &bmpCount);
  if (!bitmaps) {
    fprintf(stderr, "Failed to read bitmaps.\n");
    return 1;
  }

  if (bmpCount == 0) {
    fprintf(stderr, "No images.\n");
    return 1;
  }

  size_t width = bitmaps[0]->width;
  size_t height = bitmaps[0]->height;

  matrix_t * rowMatrix = matrix_for_image_rows(bitmaps, bmpCount);
  free_bitmaps(bitmaps, bmpCount);
  if (!rowMatrix) {
    fprintf(stderr, "Failed to allocate row matrix.\n");
    return 1;
  }

  power_iter_t * iter = power_iter_new(rowMatrix);
  matrix_free(rowMatrix);

  if (!iter) {
    fprintf(stderr, "Could not initialize power iterator.\n");
    return 1;
  }

  printf("Running power iteration...\n");

  for (int i = 0; i < 100; ++i) {
    power_iter_run(iter, 1);
  }

  printf("Generating output file...\n");

  bmp_t * outImage = vec_to_image(iter->vector, width, height);
  power_iter_free(iter);

  int res = outImage ? bmp_write(outImage, argv[2]) : -1;
  if (outImage) {
    bmp_free(outImage);
  }

  if (res) {
    fprintf(stderr, "Failed to write output image.\n");
    return -1;
  }

  return 0;
}

bmp_t ** read_bitmaps(const char * dir, size_t * countOut) {
  DIR * dh = opendir(dir);
  if (!dh) {
    return NULL;
  }

  (*countOut) = 0;
  bmp_t ** results = (bmp_t **)malloc(1);

  while (1) {
    struct dirent * ent = readdir(dh);
    if (!ent) {
      closedir(dh);
      return results;
    }
    char * path = (char *)malloc(strlen(ent->d_name) + strlen(dir) + 2);
    sprintf(path, "%s/%s", dir, ent->d_name);
    bmp_t * img = bmp_read(path);
    free(path);
    if (img == NULL) {
      continue;
    }
    ++(*countOut);
    results = realloc(results, sizeof(bmp_t *) * (*countOut));
    results[(*countOut)-1] = img;
  }
}

void free_bitmaps(bmp_t ** bmps, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    bmp_free(bmps[i]);
  }
  free(bmps);
}

bmp_t * vec_to_image(cl_float3 * vec, size_t width, size_t height) {
  bmp_t * output = (bmp_t *)malloc(sizeof(bmp_t));
  if (!output) {
    return NULL;
  }

  output->pixels = malloc(sizeof(cl_uchar4) * width * height);
  if (!output->pixels) {
    free(output);
    return NULL;
  }

  output->width = width;
  output->height = height;

  for (size_t i = 0; i < 3; ++i) {
    vec_to_image_chan(i, vec, output->pixels, width*height);
  }

  return output;
}

void vec_to_image_chan(size_t chan, cl_float3 * vec, cl_uchar4 * out, size_t count) {
  cl_float minValue = vec[0].s[chan];
  cl_float maxValue = minValue;

  for (size_t i = 1; i < count; ++i) {
    minValue = MIN(minValue, vec[i].s[chan]);
    maxValue = MAX(maxValue, vec[i].s[chan]);
  }

  for (size_t i = 0; i < count; ++i) {
    cl_float3 v = vec[i];
    cl_float num = v.s[chan] + minValue;
    num /= (maxValue - minValue) / 255.0f;

    out[i].s[chan] = (cl_uchar)num;
  }
}
