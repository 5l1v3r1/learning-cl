#include "bmp.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <strings.h>

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
  uint32_t compressionMethod;
  uint32_t imageSize;
  uint32_t horizontalResolution;
  uint32_t verticalResolution;
  uint32_t paletteSize;
  uint32_t importantColors;
} __attribute__((packed)) bmp_header_t;

static uint8_t * read_padded_rgb_data(FILE * fp, bmp_header_t head);

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

  if (head.bitsPerPixel != 24 && head.bitsPerPixel != 32) {
    fclose(fp);
    return NULL;
  }

  if (fseek(fp, (long)head.dataOffset, SEEK_SET)) {
    fclose(fp);
    return NULL;
  }

  uint8_t * pixelData;
  if (head.bitsPerPixel == 24) {
    pixelData = read_padded_rgb_data(fp, head);
  } else {
    size_t dataSize = 4 * head.width * head.height;
    pixelData = (uint8_t *)malloc(dataSize);
    if (pixelData) {
      if (fread(pixelData, 1, dataSize, fp) != dataSize) {
        free(pixelData);
        pixelData = NULL;
      }
    }
  }

  if (!pixelData) {
    fclose(fp);
    return NULL;
  }

  bmp_t * res = (bmp_t *)malloc(sizeof(bmp_t));

  if (!res) {
    free(pixelData);
    return NULL;
  }

  res->width = (int)head.width;
  res->height = (int)head.height;
  res->pixels = (cl_uchar4 *)pixelData;

  return res;
}

int bmp_write(bmp_t * b, const char * path) {
  FILE * fp = fopen(path, "w");
  if (fp == NULL) {
    return -1;
  }

  bmp_header_t header;
  bzero(&header, sizeof(header));

  memcpy(header.identifier, "BM", 2);
  header.fileSize = sizeof(header) + (b->width*b->height*3);
  header.dataOffset = sizeof(header);
  header.headerSize = 40;
  header.width = b->width;
  header.height = b->height;
  header.planeCount = 1;
  header.bitsPerPixel = 24;
  header.imageSize = b->width * b->height * 3;

  if (fwrite(&header, sizeof(header), 1, fp) != 1) {
    fclose(fp);
    return -1;
  }

  size_t pixelCount = b->width * b->height;
  uint8_t * packedData = (uint8_t *)malloc(pixelCount * 3);
  if (!packedData) {
    fclose(fp);
    return -1;
  }

  size_t i;
  uint8_t * sourceData = (uint8_t *)b->pixels;
  for (i = 0; i < pixelCount; ++i) {
    size_t destIdx = i * 3;
    size_t sourceIdx = i * 4;
    packedData[destIdx++] = sourceData[sourceIdx++];
    packedData[destIdx++] = sourceData[sourceIdx++];
    packedData[destIdx++] = sourceData[sourceIdx++];
  }

  size_t writeRes = fwrite(packedData, 1, pixelCount*3, fp);
  free(packedData);
  fclose(fp);

  if (writeRes == pixelCount*3) {
    return 0;
  } else {
    return -1;
  }
}

void bmp_free(bmp_t * b) {
  free(b->pixels);
  free(b);
}

static uint8_t * read_padded_rgb_data(FILE * fp, bmp_header_t head) {
  size_t lineSize = head.width * 3;
  size_t linePadding = 0;
  if (lineSize % 4) {
    linePadding = 4 - (lineSize % 4);
    lineSize += linePadding;
  }
  size_t dataSize = lineSize * head.height;
  uint8_t * data = (uint8_t *)malloc(dataSize);

  if (!data) {
    return NULL;
  }

  if (fread(data, 1, dataSize, fp) != dataSize) {
    free(data);
    return NULL;
  }
  fclose(fp);

  size_t pixelCount = head.width * head.height;
  uint8_t * paddedBuff = (uint8_t *)malloc(pixelCount * 4);

  if (!paddedBuff) {
    free(data);
    return NULL;
  }

  size_t destIdx = 0;
  size_t sourceIdx = 0;
  for (size_t y = 0; y < head.height; ++y) {
    for (size_t x = 0; x < head.width; ++x) {
      paddedBuff[destIdx++] = data[sourceIdx++];
      paddedBuff[destIdx++] = data[sourceIdx++];
      paddedBuff[destIdx++] = data[sourceIdx++];
      paddedBuff[destIdx++] = 0;
    }
    sourceIdx += linePadding;
  }

  free(data);
  return paddedBuff;
}
