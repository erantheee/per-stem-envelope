CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -Werror -Iinclude -Itests
LDLIBS ?= -lm

all: tests/parity

tests/parity: src/stemenvelope_s3.c tests/parity.c tests/golden.h include/stemenvelope_s3.h
	$(CC) $(CFLAGS) src/stemenvelope_s3.c tests/parity.c -o $@ $(LDLIBS)

check: tests/parity
	./tests/parity

clean:
	rm -f tests/parity

.PHONY: all check clean
