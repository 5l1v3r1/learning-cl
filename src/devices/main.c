#include <OpenCL/opencl.h>
#include <stdio.h>

#define MAX_PLATFORMS 16
#define MAX_DEVICES 16
#define MAX_STRING 256

#define ATTR_COUNT 11

typedef struct {
  cl_ulong globalCache;
  cl_uint globalCacheLine;

  cl_ulong globalMemSize;
  cl_ulong localMemSize;

  cl_uint clockFrequency;
  cl_uint computeUnits;
  size_t workGroupSize;

  char name[MAX_STRING];
  char driverVersion[MAX_STRING];
  char deviceVersion[MAX_STRING];
  char deviceVendor[MAX_STRING];
} device_attributes;

int get_attributes(cl_device_id dev, device_attributes * out);

int main() {
  cl_uint platformCount;
  cl_platform_id platforms[MAX_PLATFORMS];
  if (clGetPlatformIDs(MAX_PLATFORMS, platforms, &platformCount)) {
    fprintf(stderr, "Failed to list platforms.\n");
    return 1;
  }

  printf("Got %d platforms.\n", (int)platformCount);

  for (cl_uint i = 0; i < platformCount; ++i) {
    cl_uint deviceCount;
    cl_device_id devices[MAX_DEVICES];
    if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, MAX_DEVICES, devices, &deviceCount)) {
      fprintf(stderr, "Failed to list devices.\n");
      return 1;
    }
    printf("\nPlatform %d:\n", (int)i);
    for (cl_uint j = 0; j < deviceCount; ++j) {
      device_attributes attrs;
      if (get_attributes(devices[j], &attrs)) {
        fprintf(stderr, "Failed to get attributes.\n");
        return 1;
      }

      printf("\n- Device %d: %s\n", (int)j, attrs.name);
      printf("  Driver version: %s\n", attrs.driverVersion);
      printf("  Device version: %s\n", attrs.deviceVersion);
      printf("  Device vendor: %s\n", attrs.deviceVendor);
      printf("  Global memory size: 0x%llx\n", (long long)attrs.globalMemSize);
      printf("  Local memory size: 0x%llx\n", (long long)attrs.localMemSize);
      printf("  Global cache size: 0x%llx\n", (long long)attrs.globalCache);
      printf("  Global cache line size: 0x%llx\n", (long long)attrs.globalCacheLine);
      printf("  Max clock frequency: %lld\n", (long long)attrs.clockFrequency);
      printf("  Max compute units: %lld\n", (long long)attrs.computeUnits);
      printf("  Max work group size: %lld\n", (long long)attrs.workGroupSize);
    }
  }
}

int get_attributes(cl_device_id dev, device_attributes * out) {
  void * attrPointers[ATTR_COUNT] = {
    out->name, out->driverVersion, out->deviceVersion, out->deviceVendor,
    &out->globalCache, &out->globalCacheLine, &out->globalMemSize,
    &out->clockFrequency, &out->computeUnits, &out->workGroupSize,
    &out->localMemSize
  };

  cl_device_info attrs[ATTR_COUNT] = {
    CL_DEVICE_NAME, CL_DRIVER_VERSION, CL_DEVICE_VERSION, CL_DEVICE_VENDOR,
    CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
    CL_DEVICE_GLOBAL_MEM_SIZE, CL_DEVICE_MAX_CLOCK_FREQUENCY, CL_DEVICE_MAX_COMPUTE_UNITS,
    CL_DEVICE_MAX_WORK_GROUP_SIZE, CL_DEVICE_LOCAL_MEM_SIZE
  };

  size_t attrSizes[ATTR_COUNT] = {
    MAX_STRING-1, MAX_STRING-1, MAX_STRING-1, MAX_STRING-1, sizeof(cl_ulong),
    sizeof(cl_uint), sizeof(cl_ulong), sizeof(cl_uint), sizeof(cl_uint), sizeof(size_t),
    sizeof(cl_ulong)
  };

  for (size_t i = 0; i < ATTR_COUNT; ++i) {
    if (clGetDeviceInfo(dev, attrs[i], attrSizes[i], attrPointers[i], NULL)) {
      return 1;
    }
  }

  return 0;
}
