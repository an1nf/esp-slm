CC      ?= cc
CFLAGS  ?= -O3 -std=c99 -Wall -Wextra -Wshadow -Wvla -pedantic -ffp-contract=off
PY      ?= python3
OUT     = build
OMP     ?= 1
ifeq ($(OMP),1)
OMPFLAGS = -fopenmp
endif

TERN_SRC = $(wildcard src/tern/*.c)

.PHONY: all test clean

all: $(OUT)/tern

$(OUT)/tern: $(TERN_SRC) $(wildcard src/tern/*.h) | $(OUT)
	$(CC) $(CFLAGS) -pthread -o $@ $(TERN_SRC) -lm -pthread

test: all
	./$(OUT)/tern selftest
	$(PY) tests/tiny.py --out $(OUT)
	./$(OUT)/tern test $(OUT)/tiny_t2.tbin $(OUT)/tiny.ref
	./$(OUT)/tern test $(OUT)/tiny_t3.tbin $(OUT)/tiny.ref
	./$(OUT)/tern test $(OUT)/tiny_t2.tbin $(OUT)/tiny_kv8.ref --kv int8
	./$(OUT)/tern test $(OUT)/tiny_t3_h8.tbin $(OUT)/tiny_h8kv8.ref --kv int8
	$(PY) tests/check_t2_t3.py $(OUT)/tern $(OUT)/tiny_t2.tbin $(OUT)/tiny_t3.tbin
	$(PY) tests/test_convert.py --out $(OUT)
	$(PY) tools/store_sim.py selftest
	@echo "all tests passed"

$(OUT):
	mkdir -p $(OUT)

clean:
	rm -rf $(OUT)
