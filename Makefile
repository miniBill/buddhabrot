.PHONY: all

TMPFILE := $(shell mktemp)

CFLAGS=-O2 -Wall -pedantic -Werror -Wextra -lm

all: output.ppm

output.ppm: main
	./main > ${TMPFILE}
	mv ${TMPFILE} $@
