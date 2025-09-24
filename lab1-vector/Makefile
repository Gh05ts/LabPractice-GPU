
NVCC        = nvcc
NVCC_FLAGS  = -O0 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64

default: vecadd image2gray

vector_add.o: vector_add.cu kernel.cu support.h
	$(NVCC) -c -o $@ vector_add.cu $(NVCC_FLAGS)

image_to_gray.o: image_to_gray.cu kernel.cu support.h
	$(NVCC) -c -o $@ image_to_gray.cu $(NVCC_FLAGS) -use_fast_math

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

vecadd: vector_add.o support.o
	$(NVCC) vector_add.o support.o -o vecadd $(LD_FLAGS)

image2gray: image_to_gray.o support.o
	$(NVCC) image_to_gray.o support.o -o image2gray $(LD_FLAGS)

clean:
	rm -rf *.o vecadd image2gray
