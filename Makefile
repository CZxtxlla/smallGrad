# Compilers
CC = gcc
NVCC = nvcc

# Flags
CFLAGS = -Wall -O3
NVCCFLAGS = -O3

# Define target executable
TARGET = train_demo

# Core Library Files (Everything EXCEPT files containing a main() function)
CORE_C_SOURCES = src/autograd.c src/ops.c src/ops_cpu.c src/autograd_cpu.c src/nn.c src/optim.c src/optim_cpu.c
CORE_CU_SOURCES = src/ops_gpu.cu src/autograd_gpu.cu src/optim_gpu.cu src/tensor.cu 

# Object files for the core library
CORE_C_OBJECTS = $(CORE_C_SOURCES:.c=.o)
CORE_CU_OBJECTS = $(CORE_CU_SOURCES:.cu=.o)

# The default rule when you type `make` builds the combined training demo
all: $(TARGET)

# Rule to link the training demo
$(TARGET): $(CORE_C_OBJECTS) $(CORE_CU_OBJECTS) src/test_train.o
	$(NVCC) $(CORE_C_OBJECTS) $(CORE_CU_OBJECTS) src/test_train.o -o $(TARGET)

# Rule to compile standard C files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to compile CUDA files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Bulletproof clean rule
.PHONY: all clean

clean:
	rm -f src/*.o
	rm -f *.o
	rm -f $(TARGET)