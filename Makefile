.PHONY: all

TMPFILE := $(shell mktemp)

CC=gcc
CFLAGS=-O2 -Wall -pedantic -Werror -Wextra -lm -march=native

all: mandelbrot.png buddha.png anti-buddha.png

%.ppm: %
	time ./$^ > ${TMPFILE}
	mv ${TMPFILE} $@

mandelbrot: mandelbrot.c color.h ppm.h global.h vec3.h
	${CC} ${CFLAGS} -o $@ $<

buddha: buddha.c color.h ppm.h global.h vec3.h
	${CC} ${CFLAGS} -o $@ $<

anti-buddha: anti-buddha.c color.h ppm.h global.h vec3.h
	${CC} ${CFLAGS} -o $@ $<

%.png: %.ppm
	convert $^ $@
