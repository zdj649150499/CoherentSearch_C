# Convenience wrapper; the real build lives in src/.
all:
	$(MAKE) -C src

clean:
	$(MAKE) -C src clean

.PHONY: all clean
