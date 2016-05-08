#include "bmp.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
  char identifier[2];
  uint32_t fileSize;
  uint32_t reserved;
  uint32_t dataOffset;

  uint32_t headerSize;
  uint32_t width;
  uint32_t height;
  uint16_t planeCount;
  uint16_t bitsPerPixel;
} __attribute__((packed)) bmp_header_t;

bmp_t * bmp_read(const char * path) {
  assert(sizeof(cl_uchar4) == 4);

  FILE * fp = fopen(path, "r");
  if (fp == NULL) {
    return NULL;
  }

  bmp_header_t head;
  if (fread(&head, sizeof(head), 1, fp) != 1) {
    fclose(fp);
    return NULL;
  }

  if (head.bitsPerPixel != 24) {
    fclose(fp);
    return NULL;
  }

  if (fseek(fp, (long)head.dataOffset, SEEK_SET)) {
    fclose(fp);
    return NULL;
  }

  size_t pixelCount = (size_t)(head.width * head.height);
  size_t dataSize = pixelCount * 3;
  uint8_t * data = (uint8_t *)malloc(dataSize);

  if (fread(data, 1, dataSize, fp) != dataSize) {
    fclose(fp);
    free(data);
    return NULL;
  }

  uint8_t * paddedBuff = (uint8_t *)malloc(pixelCount * 4);
  size_t i;
  for (i = 0; i < pixelCount; i++) {
    size_t sourceIdx = i * 3;
    size_t destIdx = i * 4;
    paddedBuff[destIdx++] = data[sourceIdx++];
    paddedBuff[destIdx++] = data[sourceIdx++];
    paddedBuff[destIdx++] = data[sourceIdx++];
    paddedBuff[destIdx] = 0;
  }

  free(data);

  bmp_t * res = (bmp_t *)malloc(sizeof(bmp_t));
  res->width = (int)head.width;
  res->height = (int)head.height;
  res->pixels = (cl_uchar4 *)paddedBuff;

  return res;
}

int bmp_write(bmp_t * b, const char * path) {
  // TODO: this.
  return -1;
}

void bmp_free(bmp_t * b) {
  free(b->pixels);
  free(b);
}
