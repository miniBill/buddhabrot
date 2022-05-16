.PHONY: all

CC=gcc
CXX=g++
NVCC=nvcc
CFLAGS=-O2 -Wall -pedantic -Werror -Wextra -lm -lpthread -march=native
NVCCFLAGS=-O2
INCLUDES=src/color.h src/ppm.h src/global.h src/vec3.h

all: out/mandelbrot.png out/buddha.png out/anti-buddha.png out/buddha-cuda.png

bin/%: src/%.c ${INCLUDES}
	mkdir -p bin
	${CC} ${CFLAGS} -o $@ $<

bin/%: src/%.cpp ${INCLUDES}
	mkdir -p bin
	${CXX} ${CFLAGS} -o $@ $<

bin/%: src/%.cu ${INCLUDES}
	mkdir -p bin
	${NVCC} ${NVCCFLAGS} -o $@ $<

out/%.ppm: bin/%
	mkdir -p out
	$(eval TMPFILE := $(shell mktemp))
	time $^ ${TMPFILE}
	mv ${TMPFILE} $@

out/%.png: out/%.ppm
	mkdir -p out
	$(eval TMPFILE := $(shell mktemp))
	convert $^ ${TMPFILE}
	mv ${TMPFILE} $@

.PHONY: clean
clean:
	rm -f bin out/*.ppm
