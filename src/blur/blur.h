#ifndef __BLUR_H__
#define __BLUR_H__

#include "bmp.h"
#include <OpenCL/opencl.h>

int blur_image(bmp_t * image, int radius, cl_float sigma);

#endif
