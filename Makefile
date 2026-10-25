CC      ?= cc
CFLAGS  ?= -O3 -std=c99 -Wall -Wextra -Wshadow -Wvla -pedantic -ffp-contract=off
PY      ?= python3
OUT     = build
OMP     ?= 1
ifeq ($(OMP),1)
OMPFLAGS = -fopenmp
endif

TERN_SRC = $(wildcard src/tern/*.c)
Q35_SRC  = $(wildcard src/q35/*.c) $(wildcard src/cli/*.c) src/common/gguf.c
INC      = -Isrc/common -Isrc/q35 -Isrc/cli

.PHONY: all test clean

all: $(OUT)/tern $(OUT)/q35

$(OUT)/tern: $(TERN_SRC) $(wildcard src/tern/*.h) | $(OUT)
	$(CC) $(CFLAGS) -pthread -o $@ $(TERN_SRC) -lm -pthread

$(OUT)/q35: $(Q35_SRC) $(wildcard src/*/*.h) | $(OUT)
	$(CC) $(CFLAGS) $(OMPFLAGS) $(INC) -o $@ $(Q35_SRC) -lm

test: all
	./$(OUT)/tern selftest
	$(PY) tests/tiny.py --out $(OUT)
	./$(OUT)/tern test $(OUT)/tiny_t2.tbin $(OUT)/tiny.ref
	./$(OUT)/tern test $(OUT)/tiny_t3.tbin $(OUT)/tiny.ref
	./$(OUT)/tern test $(OUT)/tiny_t2.tbin $(OUT)/tiny_kv8.ref --kv int8
	./$(OUT)/tern test $(OUT)/tiny_t3_h8.tbin $(OUT)/tiny_h8kv8.ref --kv int8
	$(PY) tests/check_t2_t3.py $(OUT)/tern $(OUT)/tiny_t2.tbin $(OUT)/tiny_t3.tbin
	$(PY) tests/test_convert.py --out $(OUT)
	$(PY) tests/test_store.py --out $(OUT) --bin $(OUT)/tern
	$(PY) tools/store_sim.py selftest
	./$(OUT)/tern lanes $(OUT)/tiny_t3_h8.tbin --lanes 5 --steps 6 --kv int8 --check
	./$(OUT)/tern lanes $(OUT)/tiny_t3_h8.tbin --lanes 3 --steps 6 --kv int8 --check \
	    --place $(OUT)/tiny_t3_h8_kv8_stripe.place --chunk 1 --ring 2 \
	    --bw psram=50,flash=20,sd=10
	$(PY) tests/q35_tiny.py $(OUT)
	./$(OUT)/q35 test $(OUT)/q35_a.gguf $(OUT)/q35_a.ref
	./$(OUT)/q35 test $(OUT)/q35_b.gguf $(OUT)/q35_b.ref
	./$(OUT)/q35 test $(OUT)/q35_a.gguf $(OUT)/q35_c.ref
	./$(OUT)/q35 test $(OUT)/q35_d.gguf $(OUT)/q35_d.ref
	./$(OUT)/q35 test $(OUT)/q35_e.gguf $(OUT)/q35_e.ref
	@echo "all tests passed"

$(OUT):
	mkdir -p $(OUT)

clean:
	rm -rf $(OUT)
