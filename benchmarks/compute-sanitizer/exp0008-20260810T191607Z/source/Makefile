CUDA_PATH ?= /usr/local/cuda
SANITIZER_PATH ?= $(CUDA_PATH)/compute-sanitizer
CXX ?= g++
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

BUILD_DIR ?= build
INCLUDES := -I$(CUDA_PATH)/include -I$(SANITIZER_PATH)/include -I.
CXXFLAGS := -std=c++17 -O3 -fPIC -Wall -Wextra -Werror $(INCLUDES)
LDFLAGS := -shared -L$(SANITIZER_PATH) -Wl,-rpath,$(SANITIZER_PATH)
LDLIBS := -lsanitizer-public
PATCH_FLAGS := --cubin --compile-as-tools-patch -std=c++17 -O3 $(INCLUDES)
PATCH_FLAGS += -gencode arch=compute_121,code=sm_121

.PHONY: all clean

all: $(BUILD_DIR)/libgpu_observer_sanitizer.so $(BUILD_DIR)/gpu_observer_sanitizer_patches.cubin

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/libgpu_observer_sanitizer.so: subscriber.cpp probe_types.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ subscriber.cpp $(LDLIBS)

$(BUILD_DIR)/gpu_observer_sanitizer_patches.cubin: patches.cu probe_types.h | $(BUILD_DIR)
	$(NVCC) $(PATCH_FLAGS) -o $@ -c patches.cu

clean:
	rm -f $(BUILD_DIR)/libgpu_observer_sanitizer.so $(BUILD_DIR)/gpu_observer_sanitizer_patches.cubin

