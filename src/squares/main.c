#include <OpenCL/opencl.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#define NUMBER_LIST_SIZE (1<<28)

const char * squareKernel = "\
__kernel void square(__global float * values) { \
  int idx = get_global_id(0); \
  values[idx] = values[idx] * values[idx]; \
}";

cl_float * makeNumberList(int count);
long long microtime();

int main() {
  cl_uint resultCount;
  cl_int statusCode;

  cl_platform_id platform;
  if (clGetPlatformIDs(1, &platform, &resultCount) || resultCount != 1) {
    fprintf(stderr, "Unable to get valid platform ID.\n");
    return 1;
  }

  cl_device_id device;
  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &resultCount)) {
    fprintf(stderr, "Failed to list devices.\n");
    return 1;
  } else if (resultCount == 0) {
    fprintf(stderr, "No GPUs.\n");
    return 1;
  }

  cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &statusCode);
  if (statusCode) {
    fprintf(stderr, "Unable to create context.\n");
    return 1;
  }

  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &statusCode);
  if (statusCode) {
    clReleaseContext(context);
    fprintf(stderr, "Unable to create command queue.\n");
    return 1;
  }

  cl_float * list = makeNumberList(NUMBER_LIST_SIZE);
  cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
    NUMBER_LIST_SIZE*sizeof(cl_float), list, &statusCode);
  if (statusCode) {
    free(list);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    fprintf(stderr, "Failed to create buffer.\n");
    return 1;
  }

  size_t squareLen = strlen(squareKernel);
  cl_program program = clCreateProgramWithSource(context, 1, &squareKernel,
    &squareLen, &statusCode);
  if (statusCode) {
    clReleaseMemObject(buffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(list);
    fprintf(stderr, "Failed to create program.\n");
    return 1;
  }

  if (clBuildProgram(program, 1, &device, NULL, NULL, NULL)) {
    clReleaseProgram(program);
    clReleaseMemObject(buffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(list);
    fprintf(stderr, "Failed to build program.\n");
    return 1;
  }

  cl_kernel kernel = clCreateKernel(program, "square", &statusCode);
  if (statusCode) {
    clReleaseProgram(program);
    clReleaseMemObject(buffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(list);
    fprintf(stderr, "Failed to create kernel.\n");
    return 1;
  }

  if (clSetKernelArg(kernel, 0, sizeof(buffer), (void *)&buffer)) {
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(buffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(list);
    fprintf(stderr, "Failed to set kernel argument.\n");
    return 1;
  }

  long long startTime = microtime();

  cl_event event;
  size_t workSize = NUMBER_LIST_SIZE;
  if (clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &workSize, NULL, 0, NULL, &event)) {
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(buffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(list);
    fprintf(stderr, "Failed to enqueue task.\n");
    return 1;
  }

  if (clWaitForEvents(1, &event)) {
    clFinish(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(buffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(list);
    fprintf(stderr, "Failed to wait for event.\n");
    return 1;
  }

  long long duration = microtime() - startTime;

  printf("GPU completed %d multiplies in %lld microseconds.\n", NUMBER_LIST_SIZE, duration);

  int i = 0;
  int remaining = NUMBER_LIST_SIZE;
  while (remaining > 0) {
    cl_float orig = 3.1415 + (cl_float)((--remaining) % 10);
    cl_float square = orig * orig;
    if (square != list[i++]) {
      fprintf(stderr, "Invalid result %d: got %f, not %f\n", i, list[i-1], square);
    }
    list[i-1] = orig;
  }

  startTime = microtime();
  for (i = 0; i < NUMBER_LIST_SIZE; i++) {
    list[i] = list[i] * list[i];
  }
  duration = microtime() - startTime;
  printf("CPU completed %d multiplies in %lld microseconds.\n", NUMBER_LIST_SIZE, duration);

  clFlush(queue);
  clFinish(queue);
  clReleaseProgram(program);
  clReleaseMemObject(buffer);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  free(list);

  return 0;
}

cl_float * makeNumberList(int count) {
  cl_float * res = (cl_float *)malloc(count * sizeof(cl_float));

  cl_float * head = res;
  while (count--) {
    (*head) = 3.1415 + (cl_float)(count % 10);
    head++;
  }

  return res;
}

long long microtime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return ((long long)t.tv_sec)*1000000 + (long long)t.tv_usec;
}
