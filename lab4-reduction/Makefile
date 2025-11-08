
NVCC        = nvcc
NVCC_FLAGS  = --std=c++17 -O3 -lineinfo -arch=sm_86 -Xptxas="-v" -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -arch=sm_86 -lineinfo -Xptxas="-v" -L/usr/local/cuda/lib64
EXE	        = reduction ##
OBJ	        = reduce_main.o support.o ## reducue_sum.o # red.o support.o # reduce_main.o support.o # reduce_main.o support.o # reduceee.o 

default: $(EXE)

reduce_main.o: reduce_main.cu reduce_kernel.cu support.h
	$(NVCC) -c -o $@ reduce_main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

# red.o: red.cu support.h
# 	$(NVCC) -c -o $@ red.cu $(NVCC_FLAGS)

# reducee_main.o: reduceee.cu
# 	$(NVCC) -c -o $@ reduceee.cu $(NVCC_FLAGS)

reducue_sum.o: reducue_sum.cu
	$(NVCC) -c -o $@ reducue_sum.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
