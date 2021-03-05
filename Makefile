.PHONY: all

CC=gcc
CFLAGS=-O2 -Wall -pedantic -Werror -Wextra -lm -lpthread -march=native

all: mandelbrot.png buddha.png anti-buddha.png

mandelbrot: mandelbrot.c color.h ppm.h global.h vec3.h
	${CC} ${CFLAGS} -o $@ $<

buddha: buddha.c color.h ppm.h global.h vec3.h
	${CC} ${CFLAGS} -o $@ $<

anti-buddha: anti-buddha.c color.h ppm.h global.h vec3.h
	${CC} ${CFLAGS} -o $@ $<

%.ppm: %
	$(eval TMPFILE := $(shell mktemp))
	time ./$^ ${TMPFILE}
	mv ${TMPFILE} $@

%.png: %.ppm
	$(eval TMPFILE := $(shell mktemp))
	convert $^ ${TMPFILE}
	mv ${TMPFILE} $@
