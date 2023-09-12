// run like this:
// go test -v -bench Benchmark -run not

package main

import (
	"fmt"
	"testing"
)

// note: using nthread = 0 = default = GOMAXPROCS

// RunBench runs the lvis benchmark
func RunBench(b *testing.B, gpu bool, ndata, nthread int) {
	fmt.Printf("bench: gpu: %v  ndata: %d  nthread: %d\n", gpu, ndata, nthread)
	sim := &Sim{}

	sim.New()

	sim.Config.GUI = false
	sim.Config.Bench = true
	sim.Config.Run.GPU = gpu
	sim.Config.Run.NData = ndata
	sim.Config.Run.NThreads = nthread
	sim.Config.Run.NRuns = 1
	sim.Config.Run.NEpochs = 1
	sim.Config.Run.NTrials = 64
	sim.Config.Log.Run = false
	sim.Config.Log.Epoch = false

	sim.ConfigAll()
	sim.RunNoGUI()
}

// GPU

func BenchmarkGPUnData1(b *testing.B) {
	RunBench(b, true, 1, 0)
}

func BenchmarkGPUnData2(b *testing.B) {
	RunBench(b, true, 2, 0)
}
func BenchmarkGPUnData4(b *testing.B) {
	RunBench(b, true, 4, 0)
}
func BenchmarkGPUnData8(b *testing.B) {
	RunBench(b, true, 8, 0)
}
func BenchmarkGPUnData16(b *testing.B) {
	RunBench(b, true, 16, 0)
}

func BenchmarkCPUnData1(b *testing.B) {
	RunBench(b, false, 1, 0)
}
func BenchmarkCPUnData2(b *testing.B) {
	RunBench(b, false, 2, 0)
}
func BenchmarkCPUnData4(b *testing.B) {
	RunBench(b, false, 4, 0)
}
func BenchmarkCPUnData8(b *testing.B) {
	RunBench(b, false, 8, 0)
}
func BenchmarkCPUnData16(b *testing.B) {
	RunBench(b, false, 16, 0)
}
