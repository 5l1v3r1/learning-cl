#include <OpenCL/opencl.h>

typedef struct {
  int width;
  int height;
  cl_uchar4 * pixels;
} bmp_t;

bmp_t * bmp_read(const char * path);
int bmp_write(bmp_t * b, const char * path);
void bmp_free(bmp_t * b);
