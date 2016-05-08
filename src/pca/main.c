#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include "bmp.h"
#include "matrix.h"
#include "power_iter.h"

bmp_t ** read_bitmaps(const char * dir, size_t * countOut);
void free_bitmaps(bmp_t ** bmps, size_t count);

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

  // TODO: run the power iterator here and wait
  // until the answer converges.
  power_iter_run(iter, 1);

  power_iter_free(iter);
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
