
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include # -O3 -lineinfo
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64 # -lineinfo
EXE	        = convolution
OBJ	        = conv_main.o support.o

default: $(EXE)

conv_main.o: conv_main.cu conv_kernel.cu support.h
	$(NVCC) -c -o $@ conv_main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
