CC 			=	nvcc
CFLAGS		=	-O3 
PROG		=	smith-waterman/sw-test

all:$(PROG)

smith-waterman/sw-test: smith-waterman/sw.cu
	$(CC) $(CFLAGS) $^ -o $@ 

.PHONY:clean
clean:
	rm -f $(PROG)