.PHONY: all

TMPFILE := $(shell mktemp)

CFLAGS=-O2 -Wall -pedantic -Werror -Wextra -lm

all: mandelbrot.ppm

%.ppm: %
	./$^ > ${TMPFILE}
	mv ${TMPFILE} $@

mandelbrot: mandelbrot.c color.h ppm.h
	${CC} ${CFLAGS} -o $@ $<
