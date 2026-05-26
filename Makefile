# Compilers
CC = gcc
NVCC = nvcc

# Flags
CFLAGS = -Wall -O3
NVCCFLAGS = -O3

# Define target executables
TARGET_FWD = test_forward
TARGET_BWD = test_backward

# Core Library Files (Everything EXCEPT files containing a main() function)
CORE_C_SOURCES = src/autograd.c src/ops.c src/ops_cpu.c src/autograd_cpu.c
CORE_CU_SOURCES = src/ops_gpu.cu src/autograd_gpu.cu src/tensor.cu 

# Object files for the core library
CORE_C_OBJECTS = $(CORE_C_SOURCES:.c=.o)
CORE_CU_OBJECTS = $(CORE_CU_SOURCES:.cu=.o)

# The default rule when you type 'make' builds BOTH tests
all: $(TARGET_FWD) $(TARGET_BWD)

# Rule to link the Forward Test
$(TARGET_FWD): $(CORE_C_OBJECTS) $(CORE_CU_OBJECTS) src/test_forward.o
	$(NVCC) $(CORE_C_OBJECTS) $(CORE_CU_OBJECTS) src/test_forward.o -o $(TARGET_FWD)

# Rule to link the Backward Test
$(TARGET_BWD): $(CORE_C_OBJECTS) $(CORE_CU_OBJECTS) src/test_backward.o
	$(NVCC) $(CORE_C_OBJECTS) $(CORE_CU_OBJECTS) src/test_backward.o -o $(TARGET_BWD)

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
	rm -f $(TARGET_FWD) $(TARGET_BWD)