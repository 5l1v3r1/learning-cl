#include <OpenCL/opencl.h>
#include <stdio.h>
#include "bmp.h"

int main(int argc, const char ** argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <input.bmp> <output.bmp>\n", argv[0]);
    return 1;
  }

  bmp_t * inputImage = bmp_read(argv[1]);
  if (inputImage == NULL) {
    fprintf(stderr, "Could not read input image: %s\n", argv[1]);
    return 1;
  }

  // TODO: blur the image and write it to a file here.

  free(inputImage);
  return 0;
}
