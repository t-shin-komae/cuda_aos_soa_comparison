CC = nvcc
CFLAGS = -O2 -g
APP = mainN2.out mainN3.out mainN4.out mainN5.out mainN6.out


run: $(APP)
	./mainN2.out
	./mainN3.out
	./mainN4.out
	./mainN5.out
	./mainN6.out

all:$(APP)

mainN2.out:main.cu functions.cu soa.cu
	sed -i "s/const int N = [0-9]*;/const int N = 2;/g" ./soa.h
	nvcc main.cu functions.cu soa.cu -o $@ -O2

mainN3.out:main.cu functions.cu soa.cu
	sed -i "s/const int N = [0-9]*;/const int N = 3;/g" ./soa.h
	nvcc main.cu functions.cu soa.cu -o $@ -O2

mainN4.out:main.cu functions.cu soa.cu
	sed -i "s/const int N = [0-9]*;/const int N = 4;/g" ./soa.h
	nvcc main.cu functions.cu soa.cu -o $@ -O2

mainN5.out:main.cu functions.cu soa.cu
	sed -i "s/const int N = [0-9]*;/const int N = 5;/g" ./soa.h
	nvcc main.cu functions.cu soa.cu -o $@ -O2

mainN6.out:main.cu functions.cu soa.cu
	sed -i "s/const int N = [0-9]*;/const int N = 6;/g" ./soa.h
	nvcc main.cu functions.cu soa.cu -o $@ -O2

clean:
	rm *.out
