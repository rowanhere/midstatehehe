NVCC ?= nvcc

# Fat binary targets:
#   sm_61  Pascal, useful for GTX 1070 Ti testing
#   sm_86  Ampere, RTX 3090
#   sm_89  Ada, RTX 4090
#   sm_120 Blackwell, RTX 5090, requires CUDA 12.8+
GENCODE ?= \
	-gencode arch=compute_61,code=sm_61 \
	-gencode arch=compute_86,code=sm_86 \
	-gencode arch=compute_89,code=sm_89 \
	-gencode arch=compute_120,code=sm_120 \
	-gencode arch=compute_120,code=compute_120

NVCCFLAGS ?= -O3 --use_fast_math -lineinfo -Xcompiler -O3
LDFLAGS ?= -cudart=static

TARGET := midstate-cuda-miner

.PHONY: all clean test

all: $(TARGET)

$(TARGET): src/midstate_cuda_miner.cu src/stratum_reader.hpp src/stratum_protocol.hpp
	$(NVCC) $(NVCCFLAGS) $(GENCODE) -std=c++17 -o $@ $< $(LDFLAGS)

test: tests/stratum_reader_test.cpp tests/stratum_protocol_test.cpp src/stratum_reader.hpp src/stratum_protocol.hpp
	$(CXX) -O2 -std=c++17 -pthread -o stratum-reader-test tests/stratum_reader_test.cpp
	./stratum-reader-test
	$(CXX) -O2 -std=c++17 -pthread -o stratum-protocol-test tests/stratum_protocol_test.cpp
	./stratum-protocol-test

clean:
	rm -f $(TARGET) stratum-reader-test stratum-protocol-test
