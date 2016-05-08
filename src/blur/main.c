#include <OpenCL/opencl.h>
#include <stdio.h>
#include "bmp.h"
#include "blur.h"

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

  if (blur_image(inputImage, 10, 3)) {
    fprintf(stderr, "Blur operation failed.\n");
    return 1;
  }

  if (bmp_write(inputImage, argv[2])) {
    fprintf(stderr, "Could not create output image: %s\n", argv[2]);
    bmp_free(inputImage);
    return 1;
  }

  bmp_free(inputImage);
  return 0;
}
