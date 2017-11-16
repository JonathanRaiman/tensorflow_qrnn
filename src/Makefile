# Tensorflow includes and defines
TF_INC=$(shell python3 -c 'from __future__ import print_function; import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_CUDA=$(shell python3 -c 'from __future__ import print_function; import tensorflow as tf; print(int(tf.test.is_built_with_cuda()))')

TF_FLAGS=-D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D_GLIBCXX_USE_CXX11_ABI=0
TF_LIB=$(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# Dependencies
DEPDIR:=.d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS=-MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td

# Define our sources, compiling CUDA code if it's enabled
ifeq ($(TF_CUDA), 1)
    SOURCES=$(wildcard *.cpp *.cu)
    GOOGLE_CUDA_FLAG=-D GOOGLE_CUDA=1
    GOOGLE_CUDA_INCLUDES= -I/usr/include/cuda
else
    SOURCES=$(wildcard *.cpp)
    GOOGLE_CUDA_FLAG=
    GOOGLE_CUDA_INCLUDES=
endif

# Define objects and shared_library
OBJECTS=$(addsuffix .o, $(basename $(SOURCES)))
LIBRARY=qrnn_lib.so

# Compiler flags
INCLUDES= -I $(TF_INC) $(GOOGLE_CUDA_INCLUDES)

CPPFLAGS=-g -std=c++11 $(TF_FLAGS) $(INCLUDES) -fPIC -fopenmp -O2 -march=native -mtune=native $(GOOGLE_CUDA_FLAG)
NVCCFLAGS=-std=c++11 -D GOOGLE_CUDA=$(TF_CUDA) $(TF_FLAGS) $(INCLUDES) \
	-x cu --compiler-options "-fPIC" --gpu-architecture=sm_30 -lineinfo \
	-Xcompiler -std=c++98

LDFLAGS = -fPIC -fopenmp -L$(TF_LIB) -ltensorflow_framework

# Compiler directives
COMPILE.cpp = g++ $(DEPFLAGS) $(CPPFLAGS) -c
COMPILE.nvcc = nvcc --compiler-options " $(DEPFLAGS)" $(NVCCFLAGS) -c

all : $(LIBRARY)

%.o : %.cpp
	$(COMPILE.cpp) $<

%.o : %.cu
	$(COMPILE.nvcc) $<

clean :
	rm -f $(OBJECTS) $(LIBRARY)

$(LIBRARY) : $(OBJECTS)
	g++  -shared $(OBJECTS) -o $(LIBRARY) $(LDFLAGS)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS)))
