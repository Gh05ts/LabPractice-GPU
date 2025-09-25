
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64

default: sgemm tgemm

sgemm_main.o: sgemm_main.cu sgemm_kernel.cu support.h
	$(NVCC) -c -o $@ sgemm_main.cu $(NVCC_FLAGS)

tgemm_main.o: tgemm_main.cu tgemm_kernel.cu support.h
	$(NVCC) -c -o $@ tgemm_main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

sgemm: sgemm_main.o support.o
	$(NVCC) sgemm_main.o support.o -o sgemm $(LD_FLAGS)

tgemm: tgemm_main.o support.o
	$(NVCC) tgemm_main.o support.o -o tgemm $(LD_FLAGS)

clean:
	rm -rf *.o sgemm tgemm