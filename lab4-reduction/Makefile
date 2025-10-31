
NVCC        = nvcc
NVCC_FLAGS  = -O3 -lineinfo -arch=sm_75 -Xptxas="-v" -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -arch=sm_75 -lineinfo -Xptxas="-v" -L/usr/local/cuda/lib64
EXE	        = reduction
OBJ	        = reduce_main.o support.o

default: $(EXE)

reduce_main.o: reduce_main.cu reduce_kernel.cu support.h
	$(NVCC) -c -o $@ reduce_main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
