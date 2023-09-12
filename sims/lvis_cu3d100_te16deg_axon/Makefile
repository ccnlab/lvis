# lvis makefile

build:
	go build -v

build_mpi:
	go build -v -tags mpi

bench:
	go test -v -bench Benchmark -run not

# example command to run benchmark
bench_cmd:
	./lvis_cu3d100_te16deg_axon -no-gui -bench -gpu -ndata=8
	
bench_cmd_mpi:
	mpirun -np 4 ./lvis_cu3d100_te16deg_axon -no-gui -bench -mpi -gpu -ndata=8

