.PHONY: all

CC=gcc
CFLAGS=-O2 -Wall -pedantic -Werror -Wextra -lm -lpthread -march=native
INCLUDES=src/color.h src/ppm.h src/global.h src/vec3.h

all: out/mandelbrot.png out/buddha.png out/anti-buddha.png

bin/%: src/%.c ${INCLUDES}
	mkdir -p bin
	${CC} ${CFLAGS} -o $@ $<

out/%.ppm: bin/%
	mkdir -p out
	$(eval TMPFILE := $(shell mktemp))
	time ./bin/$^ ${TMPFILE}
	mv ${TMPFILE} $@

out/%.png: out/%.ppm
	mkdir -p out
	$(eval TMPFILE := $(shell mktemp))
	convert $^ ${TMPFILE}
	mv ${TMPFILE} $@

.PHONY: clean
clean:
	rm -f bin out/*.ppm
