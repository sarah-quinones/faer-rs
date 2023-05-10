gemm: gemm.o
	$(CC) -L../../target/release -lfaer_blas -o $@ gemm.o

gemm.o: gemm.c
	$(CC) -c -O2 -I../include -o $@ gemm.c

clean:
	rm -f *.o
	rm -f gemm