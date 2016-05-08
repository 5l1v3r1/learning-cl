SOURCE_DIRECTORIES = $(sort $(dir $(wildcard src/*/*)))
BUILD_FILES = $(addprefix build/, $(notdir $(SOURCE_DIRECTORIES:%/=%)))

all: $(BUILD_FILES)

build/%: src/% build/
	$(CC) -std=c99 -Ilib -Wall -O3 lib/*.c $</*.c -o $@ -framework OpenCL

build:
	mkdir build/

clean:
	rm -r build/
