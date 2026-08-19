CUDA_PATH ?= /usr/local/cuda
CXX ?= g++
TARGET_ARCH ?= sbsa-linux
CUPTI_ROOT := $(CUDA_PATH)/targets/$(TARGET_ARCH)
BUILD_DIR ?= build
INCLUDES := -I$(CUPTI_ROOT)/include -I$(CUDA_PATH)/include
CXXFLAGS := -std=c++17 -O3 -fPIC -Wall -Wextra -Werror -fno-exceptions -fno-rtti $(INCLUDES)
LDFLAGS := -shared -L$(CUPTI_ROOT)/lib -Wl,-rpath,$(CUPTI_ROOT)/lib
LDLIBS := -lcupti -pthread

.PHONY: all clean

all: $(BUILD_DIR)/libgpu_observer_cupti_activity.so

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/libgpu_observer_cupti_activity.so: cupti_activity_agent.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f $(BUILD_DIR)/libgpu_observer_cupti_activity.so
