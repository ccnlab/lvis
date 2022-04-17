// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
lvis explores how a hierarchy of areas in the ventral stream of visual
processing (up to inferotemporal (IT) cortex) can produce robust object
recognition that is invariant to changes in position, size, etc of retinal
input images.
*/
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/decoder"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/empi/empi"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/split"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

func main() {
	TheSim.New()
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			guirun()
		})
	}
}

func guirun() {
	TheSim.Config()
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Params and Prjns in params.go

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net    *axon.Network `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Prjns  Prjns         `desc:"special projections"`
	Params emer.Params   `view:"inline" desc:"all parameter management"`
	Tag    string        `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	Stats  estats.Stats  `desc:"contains computed statistic values"`
	Logs   elog.Logs     `desc:"Contains all the logs and information about the logs.'"`

	ConfusionEpc int             `desc:"epoch to start recording confusion matrix"`
	MinusCycles  int             `desc:"number of minus-phase cycles"`
	PlusCycles   int             `desc:"number of plus-phase cycles"`
	SubPools     bool            `desc:"if true, organize layers and connectivity with 2x2 sub-pools within each topological pool"`
	RndOutPats   bool            `desc:"if true, use random output patterns -- else localist"`
	PostCycs     int             `desc:"number of cycles to run after main alphacyc cycles, between stimuli"`
	PostDecay    float32         `desc:"decay to apply at start of PostCycs"`
	StartRun     int             `desc:"starting run number -- typically 0 but can be set in command args for parallel runs on a cluster"`
	MaxRuns      int             `desc:"maximum number of model runs to perform"`
	MaxEpcs      int             `desc:"maximum number of epochs to run per model run"`
	MaxTrls      int             `desc:"maximum number of training trials per epoch"`
	RepsInterval int             `desc:"how often to analyze the representations"`
	NZeroStop    int             `desc:"if a positive number, training will stop after this many epochs with zero Err"`
	TrainEnv     ImagesEnv       `desc:"Training environment"`
	TestEnv      ImagesEnv       `desc:"Testing environment"`
	Decoder      decoder.SoftMax `desc:"decoder for better output"`
	Time         axon.Time       `desc:"axon timing parameters and state"`
	TestInterval int             `desc:"how often to run through the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	ViewOn       bool            `desc:"whether to update the network view while running"`
	TrainUpdt    etime.Times     `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt     etime.Times     `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`

	GUI          egui.GUI `view:"-" desc:"manages all the gui elements"`
	SaveWts      bool     `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui        bool     `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams bool     `view:"-" desc:"if true, print message for all params that are set"`
	NeedsNewRun  bool     `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeeds     []int64  `view:"-" desc:"the current random seeds to use for each run"`

	UseMPI      bool      `view:"-" desc:"if true, use MPI to distribute computation across nodes"`
	SaveProcLog bool      `view:"-" desc:"if true, save logs per processor"`
	Comm        *mpi.Comm `view:"-" desc:"mpi communicator"`
	AllDWts     []float32 `view:"-" desc:"buffer of all dwt weight changes -- for mpi sharing"`
	SumDWts     []float32 `view:"-" desc:"buffer of MPI summed dwt weight changes"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	ss.Prjns.New()
	ss.Params.Params = ParamSets
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()

	ss.TestInterval = 20 // maybe causing issues?

	ss.Time.Defaults()
	ss.MinusCycles = 180
	ss.PlusCycles = 50
	ss.RepsInterval = 10
	ss.SubPools = true    // true
	ss.RndOutPats = false // change here
	if ss.RndOutPats {
		ss.Params.ExtraSets = "RndOutPats"
	}
	ss.PostCycs = 0
	ss.PostDecay = .5

	ss.RndSeeds = make([]int64, 100) // make enough for plenty of runs
	for i := 0; i < 100; i++ {
		ss.RndSeeds[i] = int64(i) + 1 // exclude 0
	}
	ss.ViewOn = true
	ss.TrainUpdt = etime.AlphaCycle
	ss.TestUpdt = etime.GammaCycle
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
}

func (ss *Sim) ConfigEnv() {
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 1
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 150 // was 1000
		ss.NZeroStop = -1
	}
	if ss.MaxTrls == 0 { // allow user override
		ss.MaxTrls = 512 / mpi.WorldSize()
	}

	plus := false // plus images are a bit worse overall -- stranger objects etc.

	var path string
	if plus {
		path = "images/CU3D_100_plus_renders"
		ss.TrainEnv.Nm = "cu3d100plus"
	} else {
		path = "images/CU3D_100_renders_lr20_u30_nb"
		ss.TrainEnv.Nm = "cu3d100old"
	}

	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Defaults()
	ss.TrainEnv.High16 = false // not useful -- may need more tuning?
	ss.TrainEnv.ColorDoG = true
	ss.TrainEnv.Images.NTestPerCat = 2
	ss.TrainEnv.Images.SplitByItm = true
	ss.TrainEnv.OutRandom = ss.RndOutPats
	ss.TrainEnv.OutSize.Set(10, 10)
	ss.TrainEnv.Images.SetPath(path, []string{".png"}, "_")
	ss.TrainEnv.OpenConfig()
	// ss.TrainEnv.Images.OpenPath(path, []string{".png"}, "_")
	// ss.TrainEnv.SaveConfig()

	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually
	ss.TrainEnv.Trial.Max = ss.MaxTrls

	ss.TestEnv.Nm = ss.TrainEnv.Nm
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Defaults()
	ss.TestEnv.High16 = ss.TrainEnv.High16
	ss.TestEnv.ColorDoG = ss.TrainEnv.ColorDoG
	ss.TestEnv.Images.NTestPerCat = 2
	ss.TestEnv.Images.SplitByItm = true
	ss.TestEnv.OutRandom = ss.RndOutPats
	ss.TestEnv.OutSize.Set(10, 10)
	ss.TestEnv.Test = true
	ss.TestEnv.Images.SetPath(path, []string{".png"}, "_")
	ss.TestEnv.OpenConfig()
	// ss.TestEnv.Images.OpenPath(path, []string{".png"}, "_")
	// ss.TestEnv.SaveConfig()
	ss.TestEnv.Trial.Max = ss.MaxTrls
	ss.TestEnv.Validate()

	/*
		// Delete to 60
			last20 := []string{"submarine", "synthesizer", "tablelamp", "tank", "telephone", "television", "toaster", "toilet", "trafficcone", "trafficlight", "trex", "trombone", "tropicaltree", "trumpet", "turntable", "umbrella", "wallclock", "warningsign", "wrench", "yacht"}
			next20 := []string{"pedestalsink", "person", "piano", "plant", "plate", "pliers", "propellor", "remote", "rolltopdesk", "sailboat", "scissors", "screwdriver", "sectionalcouch", "simpledesk", "skateboard", "skull", "slrcamera", "speaker", "spotlightlamp", "stapler"}
			last40 := append(last20, next20...)
			ss.TrainEnv.Images.DeleteCats(last40)
			ss.TestEnv.Images.DeleteCats(last40)
	*/

	/*
		objs20 := []string{"banana", "layercake", "trafficcone", "sailboat", "trex", "person", "guitar", "tablelamp", "doorknob", "handgun", "donut", "chair", "slrcamera", "elephant", "piano", "fish", "car", "heavycannon", "stapler", "motorcycle"}

		objsnxt20 := []string{"submarine", "synthesizer", "tank", "telephone", "television", "toaster", "toilet", "trafficlight", "tropicaltree", "trumpet", "turntable", "umbrella", "wallclock", "warningsign", "wrench", "yacht", "pedestalsink", "pliers", "sectionalcouch", "skull"}

		objs40 := append(objs20, objsnxt20...)

		ss.TrainEnv.Images.SelectCats(objs40)
		ss.TestEnv.Images.SelectCats(objs40)
	*/

	// remove most confusable items
	confuse := []string{"blade", "flashlight", "pckeyboard", "scissors", "screwdriver", "submarine"}
	ss.TrainEnv.Images.DeleteCats(confuse)
	ss.TestEnv.Images.DeleteCats(confuse)

	if ss.UseMPI {
		ss.TrainEnv.MPIAlloc()
		ss.TestEnv.MPIAlloc()
	}

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "Lvis")
	v1nrows := 5
	if ss.TrainEnv.V1m16.SepColor {
		v1nrows += 4
	}
	hi16 := ss.TrainEnv.High16
	cdog := ss.TrainEnv.ColorDoG

	v2mNp := 8
	v2lNp := 4
	v2Nu := 8
	v4Np := 4
	v4Nu := 10
	if ss.SubPools {
		v2mNp *= 2
		v2lNp *= 2
		v2Nu = 6
		v4Np = 8
		v4Nu = 7
	}

	v1m16 := net.AddLayer4D("V1m16", 16, 16, v1nrows, 4, emer.Input)
	v1l16 := net.AddLayer4D("V1l16", 8, 8, v1nrows, 4, emer.Input)
	v1m8 := net.AddLayer4D("V1m8", 16, 16, v1nrows, 4, emer.Input)
	v1l8 := net.AddLayer4D("V1l8", 8, 8, v1nrows, 4, emer.Input)
	v1m16.SetClass("V1m")
	v1l16.SetClass("V1l")
	v1m8.SetClass("V1m")
	v1l8.SetClass("V1l")

	v1m16.SetRepIdxs(ss.CenterPoolIdxs(v1m16, 2))
	v1l16.SetRepIdxs(ss.CenterPoolIdxs(v1l16, 2))
	v1m8.SetRepIdxs(ss.CenterPoolIdxs(v1m8, 2))
	v1l8.SetRepIdxs(ss.CenterPoolIdxs(v1l8, 2))

	// not useful so far..
	// clst := net.AddLayer2D("Claustrum", 5, 5, emer.Hidden)

	var v1cm16, v1cl16, v1cm8, v1cl8 emer.Layer
	if cdog {
		v1cm16 = net.AddLayer4D("V1Cm16", 16, 16, 2, 2, emer.Input)
		v1cl16 = net.AddLayer4D("V1Cl16", 8, 8, 2, 2, emer.Input)
		v1cm8 = net.AddLayer4D("V1Cm8", 16, 16, 2, 2, emer.Input)
		v1cl8 = net.AddLayer4D("V1Cl8", 8, 8, 2, 2, emer.Input)
		v1cm16.SetClass("V1Cm")
		v1cl16.SetClass("V1Cl")
		v1cm8.SetClass("V1Cm")
		v1cl8.SetClass("V1Cl")

		v1cm16.SetRepIdxs(ss.CenterPoolIdxs(v1cm16, 2))
		v1cl16.SetRepIdxs(ss.CenterPoolIdxs(v1cl16, 2))
		v1cm8.SetRepIdxs(ss.CenterPoolIdxs(v1cm8, 2))
		v1cl8.SetRepIdxs(ss.CenterPoolIdxs(v1cm8, 2))
	}

	v2m16 := net.AddLayer4D("V2m16", v2mNp, v2mNp, v2Nu, v2Nu, emer.Hidden)
	v2l16 := net.AddLayer4D("V2l16", v2lNp, v2lNp, v2Nu, v2Nu, emer.Hidden)
	v2m8 := net.AddLayer4D("V2m8", v2mNp, v2mNp, v2Nu, v2Nu, emer.Hidden)
	v2l8 := net.AddLayer4D("V2l8", v2lNp, v2lNp, v2Nu, v2Nu, emer.Hidden)
	v2m16.SetClass("V2m V2")
	v2l16.SetClass("V2l V2")
	v2m8.SetClass("V2m V2")
	v2l8.SetClass("V2l V2")

	v2m16.SetRepIdxs(ss.CenterPoolIdxs(v2m16, 2))
	v2l16.SetRepIdxs(ss.CenterPoolIdxs(v2l16, 2))
	v2m8.SetRepIdxs(ss.CenterPoolIdxs(v2m8, 2))
	v2l8.SetRepIdxs(ss.CenterPoolIdxs(v2l8, 2))

	var v1h16, v2h16, v3h16 emer.Layer
	if hi16 {
		v1h16 = net.AddLayer4D("V1h16", 32, 32, 5, 4, emer.Input)
		v2h16 = net.AddLayer4D("V2h16", 32, 32, v2Nu, v2Nu, emer.Hidden)
		v3h16 = net.AddLayer4D("V3h16", 16, 16, v2Nu, v2Nu, emer.Hidden)
		v1h16.SetClass("V1h")
		v2h16.SetClass("V2h V2")
		v3h16.SetClass("V3h")

		v1h16.SetRepIdxs(ss.CenterPoolIdxs(v1h16, 2))
		v2h16.SetRepIdxs(ss.CenterPoolIdxs(v2h16, 2))
		v3h16.SetRepIdxs(ss.CenterPoolIdxs(v3h16, 2))
	}

	v4f16 := net.AddLayer4D("V4f16", v4Np, v4Np, v4Nu, v4Nu, emer.Hidden)
	v4f8 := net.AddLayer4D("V4f8", v4Np, v4Np, v4Nu, v4Nu, emer.Hidden)
	v4f16.SetClass("V4")
	v4f8.SetClass("V4")

	v4f16.SetRepIdxs(ss.CenterPoolIdxs(v4f16, 2))
	v4f8.SetRepIdxs(ss.CenterPoolIdxs(v4f8, 2))

	teo16 := net.AddLayer4D("TEOf16", 2, 2, 15, 15, emer.Hidden)
	teo8 := net.AddLayer4D("TEOf8", 2, 2, 15, 15, emer.Hidden)
	teo16.SetClass("TEO")
	teo8.SetClass("TEO")

	te := net.AddLayer4D("TE", 2, 2, 15, 15, emer.Hidden)

	var out emer.Layer
	if ss.RndOutPats {
		out = net.AddLayer2D("Output", ss.TrainEnv.OutSize.Y, ss.TrainEnv.OutSize.X, emer.Target)
	} else {
		out = net.AddLayer4D("Output", ss.TrainEnv.OutSize.Y, ss.TrainEnv.OutSize.X, ss.TrainEnv.NOutPer, 1, emer.Target)
	}

	full := prjn.NewFull()
	_ = full
	rndcut := prjn.NewUnifRnd()
	rndcut.PCon = 0.1 // 0.2 == .1 459
	// rndprjn := prjn.NewUnifRnd()
	// rndprjn.PCon = 0.5 // 0.2 > .1
	pool1to1 := prjn.NewPoolOneToOne()
	_ = pool1to1

	pj := &ss.Prjns

	var p4x4s2, p2x2s1, p4x4s2send, p2x2s1send, p4x4s2recip, p2x2s1recip, v4toteo, teotov4 prjn.Pattern
	p4x4s2 = pj.Prjn4x4Skp2
	p2x2s1 = pj.Prjn2x2Skp1
	p4x4s2send = pj.Prjn4x4Skp2
	p2x2s1send = pj.Prjn2x2Skp1
	p4x4s2recip = pj.Prjn4x4Skp2Recip
	p2x2s1recip = pj.Prjn2x2Skp1Recip
	v4toteo = full
	teotov4 = full

	if ss.SubPools {
		p4x4s2 = pj.Prjn4x4Skp2Sub2
		p2x2s1 = pj.Prjn2x2Skp1Sub2
		p4x4s2send = pj.Prjn4x4Skp2Sub2Send
		p2x2s1send = pj.Prjn2x2Skp1Sub2Send
		p4x4s2recip = pj.Prjn4x4Skp2Sub2SendRecip
		p2x2s1recip = pj.Prjn2x2Skp1Sub2SendRecip
		v4toteo = pj.Prjn4x4Skp0Sub2
		teotov4 = pj.Prjn4x4Skp0Sub2Recip
	}

	net.ConnectLayers(v1m16, v2m16, p4x4s2, emer.Forward).SetClass("V1V2")
	net.ConnectLayers(v1l16, v2m16, p2x2s1, emer.Forward).SetClass("V1V2fmSm V1V2")

	net.ConnectLayers(v1l16, v2l16, p4x4s2, emer.Forward).SetClass("V1V2")

	net.ConnectLayers(v1m8, v2m8, p4x4s2, emer.Forward).SetClass("V1V2")
	net.ConnectLayers(v1l8, v2m8, p2x2s1, emer.Forward).SetClass("V1V2fmSm V1V2")

	net.ConnectLayers(v1l8, v2l8, p4x4s2, emer.Forward).SetClass("V1V2")

	if cdog {
		net.ConnectLayers(v1cm16, v2m16, p4x4s2, emer.Forward).SetClass("V1V2")
		net.ConnectLayers(v1cl16, v2m16, p2x2s1, emer.Forward).SetClass("V1V2fmSm V1V2")

		net.ConnectLayers(v1cl16, v2l16, p4x4s2, emer.Forward).SetClass("V1V2")

		net.ConnectLayers(v1cm8, v2m8, p4x4s2, emer.Forward).SetClass("V1V2")
		net.ConnectLayers(v1cl8, v2m8, p2x2s1, emer.Forward).SetClass("V1V2fmSm V1V2")

		net.ConnectLayers(v1cl8, v2l8, p4x4s2, emer.Forward).SetClass("V1V2")
	}

	v2v4, v4v2 := net.BidirConnectLayers(v2m16, v4f16, p4x4s2send)
	v2v4.SetClass("V2V4")
	v4v2.SetClass("V4V2").SetPattern(p4x4s2recip)

	v2v4, v4v2 = net.BidirConnectLayers(v2l16, v4f16, p2x2s1send)
	v2v4.SetClass("V2V4sm")
	v4v2.SetClass("V4V2").SetPattern(p2x2s1recip)

	v2v4, v4v2 = net.BidirConnectLayers(v2m8, v4f8, p4x4s2send)
	v2v4.SetClass("V2V4")
	v4v2.SetClass("V4V2").SetPattern(p4x4s2recip)

	v2v4, v4v2 = net.BidirConnectLayers(v2l8, v4f8, p2x2s1send)
	v2v4.SetClass("V2V4sm")
	v4v2.SetClass("V4V2").SetPattern(p2x2s1recip)

	if hi16 {
		net.ConnectLayers(v1h16, v2h16, p4x4s2, emer.Forward).SetClass("V1V2")
		v2v3, v3v2 := net.BidirConnectLayers(v2h16, v3h16, p4x4s2send)
		v2v3.SetClass("V2V3")
		v3v2.SetClass("V3V2").SetPattern(p4x4s2recip)
		v3v4, v4v3 := net.BidirConnectLayers(v3h16, v4f16, p4x4s2send)
		v3v4.SetClass("V3V4")
		v4v3.SetClass("V4V3").SetPattern(p4x4s2recip)
	}

	v4teo, teov4 := net.BidirConnectLayers(v4f16, teo16, v4toteo)
	v4teo.SetClass("V4TEO")
	teov4.SetClass("TEOV4").SetPattern(teotov4)
	net.ConnectLayers(v4f8, teo16, v4toteo, emer.Forward).SetClass("V4TEOoth")

	v4teo, teov4 = net.BidirConnectLayers(v4f8, teo8, v4toteo)
	v4teo.SetClass("V4TEO")
	teov4.SetClass("TEOV4").SetPattern(teotov4)
	net.ConnectLayers(v4f16, teo8, v4toteo, emer.Forward).SetClass("V4TEOoth")

	teote, teteo := net.BidirConnectLayers(teo16, te, full)
	teote.SetClass("TEOTE")
	teteo.SetClass("TETEO")
	teote, teteo = net.BidirConnectLayers(teo8, te, full)
	teote.SetClass("TEOTE")
	teteo.SetClass("TETEO")

	// full connections to output are key
	teoout, outteo := net.BidirConnectLayers(teo16, out, full)
	teoout.SetClass("TEOOut ToOut")
	outteo.SetClass("OutTEO FmOut")

	teoout, outteo = net.BidirConnectLayers(teo8, out, full)
	teoout.SetClass("TEOOut ToOut")
	outteo.SetClass("OutTEO FmOut")

	teout, _ := net.BidirConnectLayers(te, out, full)
	teout.SetClass("ToOut FmOut")

	// v59 459 -- only useful later -- TEO maybe not doing as well later?
	v4out, outv4 := net.BidirConnectLayers(v4f16, out, full)
	v4out.SetClass("V4Out ToOut")
	outv4.SetClass("OutV4 FmOut")

	v4out, outv4 = net.BidirConnectLayers(v4f8, out, full)
	v4out.SetClass("V4Out ToOut")
	outv4.SetClass("OutV4 FmOut")

	var v2inhib, v4inhib prjn.Pattern
	v2inhib = pool1to1
	v4inhib = pool1to1
	if ss.SubPools {
		v2inhib = pj.Prjn2x2Skp2 // pj.Prjn6x6Skp2Lat
		v4inhib = pj.Prjn2x2Skp2
	}

	// this extra inhibition drives decorrelation, produces significant learning benefits
	net.LateralConnectLayerPrjn(v2m16, v2inhib, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(v2l16, v2inhib, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(v2m8, v2inhib, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(v2l8, v2inhib, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(v4f16, v4inhib, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(v4f8, v4inhib, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(teo16, pool1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(teo8, pool1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(te, pool1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)

	if hi16 {
		net.LateralConnectLayerPrjn(v2h16, v2inhib, &axon.HebbPrjn{}).SetType(emer.Inhib)
		net.LateralConnectLayerPrjn(v3h16, v2inhib, &axon.HebbPrjn{}).SetType(emer.Inhib)
	}

	///////////////////////
	// 	Shortcuts:

	// clst not useful
	// net.ConnectLayers(v1l16, clst, full, emer.Forward)

	// V1 shortcuts best for syncing all layers -- like the pulvinar basically
	net.ConnectLayers(v1l16, v4f16, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v1l8, v4f8, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v1l16, teo16, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v1l16, teo16, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v1l8, teo8, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v1l8, teo8, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v1l16, te, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v1l8, te, rndcut, emer.Forward).SetClass("V1SC")

	if hi16 {
		net.ConnectLayers(v1l16, v3h16, rndcut, emer.Forward).SetClass("V1SC")
	}

	//////////////////////
	// 	Positioning

	v1m8.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v1m16.Name(), YAlign: relpos.Front, Space: 4})

	v1l16.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v1m16.Name(), XAlign: relpos.Left, Space: 4})
	v1l8.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v1m8.Name(), XAlign: relpos.Left, Space: 4})
	// clst.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v1l8.Name(), XAlign: relpos.Left, Space: 4, Scale: 2})

	if cdog {
		v1cm16.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v1m8.Name(), YAlign: relpos.Front, Space: 4})
		v1cm8.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v1cm16.Name(), YAlign: relpos.Front, Space: 4})
		v1cl16.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v1cm16.Name(), XAlign: relpos.Left, Space: 4})
		v1cl8.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v1cm8.Name(), XAlign: relpos.Left, Space: 4})
	}

	if hi16 {
		v1h16.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v1m8.Name(), YAlign: relpos.Front, Space: 4})
		v2h16.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v2m8.Name(), YAlign: relpos.Front, Space: 4})
		v3h16.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v4f16.Name(), XAlign: relpos.Left, Space: 4})
	}

	v2m16.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: v1m16.Name(), XAlign: relpos.Left, YAlign: relpos.Front})

	v2m8.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v2m16.Name(), YAlign: relpos.Front, Space: 4})

	v2l16.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v2m16.Name(), XAlign: relpos.Left, Space: 4})
	v2l8.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v2m8.Name(), XAlign: relpos.Left, Space: 4})

	v4f16.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: v2m16.Name(), XAlign: relpos.Left, YAlign: relpos.Front})
	teo16.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v4f16.Name(), YAlign: relpos.Front, Space: 4})

	v4f8.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: v2m8.Name(), XAlign: relpos.Left, YAlign: relpos.Front})
	teo8.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v4f8.Name(), YAlign: relpos.Front, Space: 4})

	te.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: teo8.Name(), XAlign: relpos.Left, Space: 15})

	out.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: te.Name(), XAlign: relpos.Left, Space: 15})

	if hi16 {
		v3h16.SetThread(1)
	}

	v4f16.SetThread(1)
	v4f8.SetThread(1)

	teo16.SetThread(1)
	teo8.SetThread(1)
	te.SetThread(1)
	out.SetThread(1)

	net.Defaults()
	ss.Params.SetObject("Network")
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	ss.InitWts(net)

	if !ss.NoGui {
		sr := net.SizeReport()
		mpi.Printf("%s", sr)
	}
	ar := net.ThreadReport() // hand tuning now..
	mpi.Printf("%s", ar)

	// adding each additional layer type improves decoding..
	layers := []emer.Layer{v4f16, v4f8, teo16, teo8, out}
	// layers := []emer.Layer{teo16, teo8, out}
	// layers := []emer.Layer{teo16, teo8}
	ss.Decoder.InitLayer(len(ss.TrainEnv.Images.Cats), layers)
	ss.Decoder.Lrate = 0.05 // 0.05 > 0.1 > 0.2 for larger number of objs!
}

func (ss *Sim) InitWts(net *axon.Network) {
	net.InitWts()
}

// CenterPoolShape returns shape for 2x2 center pools (including sub-pools).
func (ss *Sim) CenterPoolShape(ly emer.Layer, n int) []int {
	nsp := 1
	if ss.SubPools {
		nsp = 2
	}
	return []int{n * nsp, n * nsp, ly.Shape().Dim(2), ly.Shape().Dim(3)}
}

// CenterPoolIdxs returns the unit indexes for 2x2 center pools
// (including sub-pools).
func (ss *Sim) CenterPoolIdxs(ly emer.Layer, n int) []int {
	npy := ly.Shape().Dim(0)
	npx := ly.Shape().Dim(1)
	npxact := npx
	nu := ly.Shape().Dim(2) * ly.Shape().Dim(3)
	nsp := 1
	if ss.SubPools {
		npy /= 2
		npx /= 2
		nsp = 2
	}
	cpy := (npy - n) / 2
	cpx := (npx - n) / 2
	nt := n * n * nsp * nsp * nu
	idxs := make([]int, nt)

	ix := 0
	for py := 0; py < 2; py++ {
		for sy := 0; sy < nsp; sy++ {
			for px := 0; px < 2; px++ {
				for sx := 0; sx < nsp; sx++ {
					y := (py+cpy)*nsp + sy
					x := (px+cpx)*nsp + sx
					si := (y*npxact + x) * nu
					for ni := 0; ni < nu; ni++ {
						idxs[ix+ni] = si + ni
					}
					ix += nu
				}
			}
		}
	}
	return idxs
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.InitRndSeed()
	ss.TrainEnv.Run.Max = ss.MaxRuns
	ss.GUI.StopNow = false
	ss.Params.SetAll()
	// note: in general shortening the time constants based on MPI is not useful
	ss.Net.SlowInterval = 100 // 100 > 20
	ss.NewRun()
	ss.GUI.UpdateNetView()
}

// NewRndSeed gets a new random seed based on current time -- otherwise uses
// the same random seed for every run
// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed() {
	run := ss.TrainEnv.Run.Cur
	rand.Seed(ss.RndSeeds[run])
}

// NewRndSeed gets a new set of random seeds based on current time -- otherwise uses
// the same random seeds for every run
func (ss *Sim) NewRndSeed() {
	rs := time.Now().UnixNano()
	for i := 0; i < 100; i++ {
		ss.RndSeeds[i] = rs + int64(i)
	}
}

func (ss *Sim) UpdateViewTime(viewUpdt etime.Times) {
	switch viewUpdt {
	case etime.Cycle:
		ss.GUI.UpdateNetView()
	case etime.FastSpike:
		if ss.Time.Cycle%10 == 0 {
			ss.GUI.UpdateNetView()
		}
	case etime.GammaCycle:
		if ss.Time.Cycle%25 == 0 {
			ss.GUI.UpdateNetView()
		}
	case etime.AlphaCycle:
		if ss.Time.Cycle%100 == 0 {
			ss.GUI.UpdateNetView()
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// ThetaCyc runs one theta cycle (200 msec) of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope, and calls TrainStats()
func (ss *Sim) ThetaCyc(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.TrainUpdt
	mode := etime.Train.String()
	if !train {
		viewUpdt = ss.TestUpdt
		mode = etime.Test.String()
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	if train {
		ss.MPIWtFmDWt()
	}

	minusCyc := ss.MinusCycles
	plusCyc := ss.PlusCycles

	ss.Net.NewState()
	ss.Time.NewState(mode)
	for cyc := 0; cyc < minusCyc; cyc++ { // do the minus phase
		ss.Net.Cycle(&ss.Time)
		ss.StatCounters(train)
		if train {
			ss.Log(etime.Train, etime.Cycle) // used for First* stats
		}
		if ss.GUI.Active {
			ss.RasterRec(ss.Time.Cycle)
		}
		ss.Time.CycleInc()
		switch ss.Time.Cycle { // save states at beta-frequency -- not used computationally
		case 75:
			ss.Net.ActSt1(&ss.Time)
			// if erand.BoolProb(float64(ss.PAlphaPlus), -1) {
			// 	ss.Net.TargToExt()
			// 	ss.Time.PlusPhase = true
			// }
		case 100:
			ss.Net.ActSt2(&ss.Time)
			// ss.Net.ClearTargExt()
			// ss.Time.PlusPhase = false
		}

		if cyc == minusCyc-1 { // do before view update
			ss.Net.MinusPhase(&ss.Time)
		}
		if ss.ViewOn {
			ss.UpdateViewTime(viewUpdt)
		}
	}
	ss.Time.NewPhase(true)
	ss.StatCounters(train)
	if viewUpdt == etime.Phase {
		ss.GUI.UpdateNetView()
	}
	for cyc := 0; cyc < plusCyc; cyc++ { // do the plus phase
		ss.Net.Cycle(&ss.Time)
		ss.StatCounters(train)
		if train {
			ss.Log(etime.Train, etime.Cycle) // used for First* stats
		}
		if ss.GUI.Active {
			ss.RasterRec(ss.Time.Cycle)
		}
		ss.Time.CycleInc()

		if cyc == plusCyc-1 { // do before view update
			ss.Net.PlusPhase(&ss.Time)
		}
		if ss.ViewOn {
			ss.UpdateViewTime(viewUpdt)
		}
	}

	ss.TrialStats(train)
	ss.StatCounters(train)

	if train {
		ss.Net.DWt(&ss.Time)
	}

	if viewUpdt == etime.Phase || viewUpdt == etime.AlphaCycle || viewUpdt == etime.ThetaCycle {
		ss.GUI.UpdateNetView()
	}

	// include extra off cycles at end
	if ss.PostCycs > 0 {
		ss.Net.InitExt()
		ss.Net.DecayState(ss.PostDecay)
		mxcyc := ss.PostCycs
		for cyc := 0; cyc < mxcyc; cyc++ {
			ss.Net.Cycle(&ss.Time)
			ss.Time.CycleInc()
			ss.StatCounters(train)
			if ss.ViewOn {
				ss.UpdateViewTime(viewUpdt)
			}
		}
	}
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := ss.Net.LayersByClass("Input", "Target")
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}

	// ly := ss.Net.LayerByName("Claustrum").(axon.AxonLayer).AsAxon()
	// ly.ApplyExt1D32([]float32{1})
}

// TrainTrial runs one trial of training using TrainEnv
func (ss *Sim) TrainTrial() {

	if ss.NeedsNewRun {
		ss.NewRun()
	}

	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.Log(etime.Train, etime.Epoch)
		ss.EpochSched(epc)
		if ss.ViewOn && ss.TrainUpdt > etime.AlphaCycle {
			ss.GUI.UpdateNetView()
		}
		if ss.TestInterval > 0 && epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		if epc >= ss.MaxEpcs || (ss.NZeroStop > 0 && ss.Stats.Int("NZero") >= ss.NZeroStop) {
			// done with training..
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.GUI.StopNow = true
				return
			} else {
				ss.NeedsNewRun = true
				return
			}
		}
	}

	// note: type must be in place before apply inputs
	ss.Net.LayerByName("Output").SetType(emer.Target)
	ss.ApplyInputs(&ss.TrainEnv)
	ss.ThetaCyc(true) // train
	ss.Log(etime.Train, etime.Trial)
	if ss.GUI.IsRunning {
		ss.GUI.Grid("Image").SetTensor(&ss.TrainEnv.Img.Tsr)
	}
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.Log(etime.Train, etime.Run)
	if ss.SaveWts {
		ss.SaveWeights()
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	run := ss.TrainEnv.Run.Cur
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
	ss.Time.Reset()
	ss.InitWts(ss.Net)
	ss.InitStats()
	ss.StatCounters(true)
	ss.Logs.ResetLog(etime.Train, etime.Trial)
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Trial)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
	ss.NeedsNewRun = false
}

// TrainEpoch runs training trials for remainder of this epoch
func (ss *Sim) TrainEpoch() {
	ss.GUI.StopNow = false
	curEpc := ss.TrainEnv.Epoch.Cur
	for {
		ss.TrainTrial()
		if ss.GUI.StopNow || ss.TrainEnv.Epoch.Cur != curEpc {
			break
		}
	}
	ss.Stopped()
}

// TrainRun runs training trials for remainder of run
func (ss *Sim) TrainRun() {
	ss.GUI.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.TrainTrial()
		if ss.GUI.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()
}

// Train runs the full training from this point onward
func (ss *Sim) Train() {
	ss.GUI.StopNow = false
	for {
		ss.TrainTrial()
		if ss.GUI.StopNow {
			break
		}
	}
	ss.Stopped()
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.GUI.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.GUI.Stopped()
}

// SaveWeights saves the network weights with the std wts filename
func (ss *Sim) SaveWeights() {
	if mpi.WorldRank() != 0 {
		return
	}
	fnm := ss.WeightsFileName()
	mpi.Printf("Saving Weights to: %s\n", fnm)
	ss.Net.SaveWtsJSON(gi.FileName(fnm))
}

// EpochSched implements epoch-wise scheduling of things..
func (ss *Sim) EpochSched(epc int) {
	if ss.UseMPI {
		empi.RandCheck(ss.Comm) // prints error message
	}
	switch epc {
	case 25:
		// ss.SaveWeights()
	case 50:
		// ss.SaveWeights()
	case 100:
		ss.SaveWeights()
		// ss.Net.LrateSched(2)
		// mpi.Printf("increased lrate to 2.0 at epoch: %d\n", epc)
	case 150:
		// ss.SaveWeights()
		// 	ss.Params.SetObjectSet("Network", "WeakShorts")
		// 	mpi.Printf("weaker shortcut cons at epoch: %d\n", epc)
		// case 200: // these have no effect anymore -- with dopamine modulator!
		// ss.Net.LrateSched(0.5)
		// mpi.Printf("dropped lrate to 0.5 at epoch: %d\n", epc)
	case 500:
		ss.SaveWeights()
		// ss.Net.LrateSched(0.2)
		// mpi.Printf("dropped lrate to 0.2 at epoch: %d\n", epc)
		// ss.Params.SetObjectSet("Network", "ToOutTol") // increase LoTol
		// ss.Params.SetObjectSet("Network", "OutAdapt") // increase LoTol
	case 600:
		// ss.Net.LrateSched(0.1)
		// mpi.Printf("dropped lrate to 0.1 at epoch: %d\n", epc)
	case 800:
		// ss.Net.LrateSched(0.05)
		// mpi.Printf("dropped lrate to 0.05 at epoch: %d\n", epc)
	case 900:
		ss.SaveWeights()
		// ss.TrainEnv.TransSigma = 0
		// ss.TestEnv.TransSigma = 0
		// mpi.Printf("reset TransSigma to 0 at epoch: %d\n", epc)
	case 1000:
		ss.SaveWeights()
	case 1500:
		ss.SaveWeights()
	}
}

// OpenTrainedWts opens trained weights
func (ss *Sim) OpenTrainedWts() {
	// ab, err := Asset("lvis_train1.wts") // embedded in executable
	// if err != nil {
	// 	log.Println(err)
	// }
	// ss.Net.ReadWtsJSON(bytes.NewBuffer(ab))
	// ss.Net.OpenWtsJSON("lvis_train1.wts.gz")
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial(returnOnChg bool) {
	ss.TestEnv.Step()

	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdt > etime.AlphaCycle {
			ss.GUI.UpdateNetView()
		}
		ss.Log(etime.Test, etime.Epoch)
		if returnOnChg {
			return
		}
	}

	// note: type must be in place before apply inputs
	ss.Net.LayerByName("Output").SetType(emer.Compare)
	ss.ApplyInputs(&ss.TestEnv)
	ss.ThetaCyc(false) // !train
	ss.Log(etime.Test, etime.Trial)
	if ss.GUI.IsRunning {
		ss.GUI.Grid("Image").SetTensor(&ss.TestEnv.Img.Tsr)
	}
	ss.GUI.NetDataRecord()
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	for {
		ss.TestTrial(true) // return on chg, don't present
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.GUI.StopNow {
			break
		}
	}
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.GUI.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

// ConfusionTstPlot plots the current confusion probability values.
// if cat is empty then it is the diagonal accuracy across all cats
// otherwise it is the confusion row for given category.
// data goes in the TrlErr = Err column.
func (ss *Sim) ConfusionTstPlot(cat string) {
	ss.Logs.ResetLog(etime.Test, etime.Trial)
	nc := ss.Stats.Confusion.N.Len()
	ti := -1
	if cat != "" {
		ti = ss.TrainEnv.Images.CatMap[cat]
	}
	for i := 0; i < nc; i++ {
		ss.TestEnv.Trial.Cur = i
		ss.TestEnv.CurCat = ss.TrainEnv.Images.Cats[i]
		if ti >= 0 {
			ss.Stats.SetFloat("TrlErr", ss.Stats.Confusion.Prob.Value([]int{ti, i}))
		} else {
			ss.Stats.SetFloat("TrlErr", ss.Stats.Confusion.Prob.Value([]int{i, i}))
		}
		ss.Log(etime.Test, etime.Trial)
	}
	plt := ss.GUI.Plot(etime.Test, etime.Trial)
	plt.Params.XAxisCol = "Cat"
	plt.Params.Type = eplot.Bar
	plt.Params.XAxisRot = 45
	plt.GoUpdate()
}

// TestRFs runs test for receptive fields
func (ss *Sim) TestRFs() {
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	ss.Stats.ActRFs.Reset()
	for {
		ss.TestTrial(true) // return on chg, don't present
		ss.Stats.UpdateActRFs(ss.Net, "ActM", 0.01)
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.GUI.StopNow {
			break
		}
	}
	ss.Stats.ActRFsAvgNorm()
	ss.GUI.ViewActRFs(&ss.Stats.ActRFs)
}

// RunTestRFs runs test for receptive fields
func (ss *Sim) RunTestRFs() {
	ss.GUI.StopNow = false
	ss.TestRFs()
	ss.Stopped()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("TrlErr", 0.0)
	ss.Stats.SetFloat("TrlErr2", 0.0)
	ss.Stats.SetFloat("TrlUnitErr", 0.0)
	ss.Stats.SetFloat("TrlCosDiff", 0.0)
	ss.Stats.SetFloat("TrlTrgAct", 0.0)
	ss.Stats.SetString("TrlOut", "")
	ss.Stats.SetString("TrlOut", "")
	ss.Stats.SetString("Cat", "0")
	ss.Stats.SetInt("FirstZero", -1) // critical to reset to -1
	ss.Stats.SetInt("NZero", 0)

	ss.Stats.Confusion.InitFromLabels(ss.TrainEnv.Images.Cats, 12)
	ss.ConfusionEpc = 500
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them to the GUI, if the GUI is active
func (ss *Sim) StatCounters(train bool) {
	ev := ss.TrainEnv
	if !train {
		ev = ss.TestEnv
	}
	ss.Stats.SetInt("Run", ss.TrainEnv.Run.Cur)
	ss.Stats.SetInt("Epoch", ss.TrainEnv.Epoch.Cur)
	ss.Stats.SetInt("Trial", ev.Trial.Cur)
	ss.Stats.SetString("TrialName", ev.String())
	ss.Stats.SetInt("Cycle", ss.Time.Cycle)
	ss.GUI.NetViewText = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle", "TrlUnitErr", "TrlErr", "TrlCosDiff"})
}

// TrialStats computes the trial-level statistics, using the train or test env.
func (ss *Sim) TrialStats(train bool) {
	var env *ImagesEnv
	if train {
		env = &ss.TrainEnv
	} else {
		env = &ss.TestEnv
	}

	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()
	ss.Stats.SetFloat("TrlCosDiff", float64(out.CosDiff.Cos))
	ss.Stats.SetFloat("TrlUnitErr", out.PctUnitErr())

	ovt := ss.Stats.SetLayerTensor(ss.Net, "Output", "ActM")
	// rsp, trlErr, trlErr2 := ev.OutErr(ovt)

	ncats := len(env.Images.Cats)

	rsp, trlErr, trlErr2 := env.OutErr(ovt)
	ss.Stats.SetInt("TrlRespIdx", rsp)
	ss.Stats.SetFloat("TrlErr", trlErr)
	ss.Stats.SetFloat("TrlErr2", trlErr2)
	if rsp >= 0 && rsp < ncats {
		ss.Stats.SetString("TrlResp", env.Images.Cats[rsp])
	}
	ss.Stats.SetInt("TrlCatIdx", env.CurCatIdx)
	ss.Stats.SetString("TrlCat", env.CurCat)

	epc := env.Epoch.Cur
	if epc > ss.ConfusionEpc {
		ss.Stats.Confusion.Incr(ss.Stats.Int("TrlCatIdx"), ss.Stats.Int("TrlRespIdx"))
	}

	ss.Stats.SetFloat("TrlTrgAct", float64(out.Pools[0].ActP.Avg/0.01))
	decIdx := ss.Decoder.Decode("ActM")
	ss.Stats.SetInt("TrlDecRespIdx", decIdx)
	if train {
		ss.Decoder.Train(env.CurCatIdx)
	}
	decErr := float64(0)
	if decIdx != env.CurCatIdx {
		decErr = 1
	}
	ss.Stats.SetFloat("TrlDecErr", decErr)
	decErr2 := decErr
	if ss.Decoder.Sorted[1] == env.CurCatIdx {
		decErr2 = 0
	}
	ss.Stats.SetFloat("TrlDecErr2", decErr2)

	cyclog := ss.Logs.Log(etime.Train, etime.Cycle)
	var fcyc int
	fcyc, rsp, trlErr, trlErr2 = ss.FirstOut(cyclog)
	ss.Stats.SetInt("TrlOutFirstCyc", fcyc)
	ss.Stats.SetInt("TrlFirstResp", rsp)
	ss.Stats.SetFloat("TrlFirstErr", trlErr)
	ss.Stats.SetFloat("TrlFirstErr2", trlErr2)
}

// FindPeaks returns indexes of local maxima in input slice, smoothing
// the data first using GaussKernel
func (ss *Sim) FindPeaks(data []float64) []int {
	// convolve.Slice64(&ss.SmoothData, data, ss.GaussKernel)
	// dt := ss.SmoothData
	dt := data // already smooth
	sz := len(dt)
	off := 10
	peaks := []int{}
	for wd := 4; wd >= 1; wd-- {
		for i := off + wd; i < sz-wd; i++ {
			ctr := dt[i]
			fail := false
			for j := -wd; j <= wd; j++ {
				if dt[i+j] > ctr {
					fail = true
					break
				}
			}
			if !fail {
				peaks = append(peaks, i)
				i += wd
			}
		}
		if len(peaks) > 0 {
			break
		}
	}
	return peaks
}

// FindActCycle returns the point at which max activity stops going up by more than .01
// within minus phase.
// must be passed max act data cycle-by-cycle
func (ss *Sim) FindActCycle(data []float64) int {
	mx := ss.MinusCycles
	dt := data  // data is already smooth
	start := 25 // give time for prior act to decay
	thr := 0.01 // rise threshold
	hit := false
	cyc := mx
	for i := start; i < mx; i++ {
		del := dt[i] - dt[i-1]
		if !hit {
			if del > thr {
				hit = true
			}
			continue
		}
		if del < thr {
			cyc = i
			break
		}
	}
	return cyc
}

// FirstActStat returns first major activation of given layer
func (ss *Sim) FirstActStat(cyclog *etable.Table, lnm string) int {
	dc := cyclog.ColByName(lnm + "_ActMax").(*etensor.Float64)
	return ss.FindActCycle(dc.Values)
}

// FirstOut returns first output response at first peak of output activity
func (ss *Sim) FirstOut(cyclog *etable.Table) (fcyc, resp int, err float64, err2 float64) {
	fcyc = ss.FirstActStat(cyclog, "Output")
	out := cyclog.CellTensor("Output_Act", fcyc).(*etensor.Float32)
	resp, err, err2 = ss.TrainEnv.OutErr(out)
	return
}

// HogDead computes the proportion of units in given layer name with
// ActAvg over hog thr and under dead threshold.
// This has now been supplanted by the PCA stats which provide a lot more info.
func (ss *Sim) HogDead(lnm string) (hog, dead float64) {
	ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
	rix := ly.RepIdxs()
	nix := len(rix)
	if nix > 0 {
		for _, ri := range rix {
			nrn := &ly.Neurons[ri]
			if nrn.ActAvg > 0.3 {
				hog += 1
			} else if nrn.ActAvg < 0.01 {
				dead += 1
			}
		}
		hog /= float64(nix)
		dead /= float64(nix)
	} else {
		for ni := range ly.Neurons {
			nrn := &ly.Neurons[ni]
			if nrn.ActAvg > 0.3 {
				hog += 1
			} else if nrn.ActAvg < 0.01 {
				dead += 1
			}
		}
		n := len(ly.Neurons)
		hog /= float64(n)
		dead /= float64(n)
	}
	return
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.ConfigLogItems()
	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Test, etime.Cycle)
	// ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "Params")
	ss.Stats.ConfigRasters(ss.Net, ss.MinusCycles+ss.PlusCycles+ss.PostCycs, []string{"V1l16", "V2l16", "V4f16", "TEOf16", "TE", "Output"})

	ss.Stats.SetF32Tensor("Image", &ss.TestEnv.Img.Tsr) // image used for actrfs, must be there first
	ss.Stats.InitActRFs(ss.Net, []string{"V4f16:Image", "V4f16:Output", "TEOf16:Image", "TEOf16:Output", "TEOf8:Image", "TEOf8:Output"}, "ActM")

	// reshape v4 tensor for inner 2x2 set of representative units
	// v4 := ss.Net.LayerByName("V4").(axon.AxonLayer).AsAxon()
	// ss.Stats.F32Tensor("V4").SetShape([]int{2, 2, v4.Shp.Dim(2), v4.Shp.Dim(3)}, nil, nil)
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	if ss.UseMPI && time == etime.Epoch { // Must gather data for trial level if doing epoch level
		ss.Logs.MPIGatherTableRows(mode, etime.Trial, ss.Comm)
	}

	dt := ss.Logs.Table(mode, time)
	row := dt.Rows
	switch {
	case mode == etime.Test && time == etime.Epoch:
		ss.LogTestErrors()
	case mode == etime.Train && time == etime.Epoch:
		epc := ss.TrainEnv.Epoch.Cur
		if (ss.RepsInterval > 0) && ((epc-1)%ss.RepsInterval == 0) { // -1 so runs on first epc
			ss.PCAStats()
		}
		ss.LogTrainErrStats()
	case time == etime.Cycle:
		row = ss.Stats.Int("Cycle")
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
	if time == etime.Cycle {
		ss.GUI.UpdateCyclePlot(etime.Test, ss.Time.Cycle)
	} else {
		ss.GUI.UpdatePlot(mode, time)
	}

	switch {
	case mode == etime.Train && time == etime.Run:
		ss.LogRunStats()
	case mode == etime.Train && time == etime.Trial:
		epc := ss.TrainEnv.Epoch.Cur
		if ss.RepsInterval > 0 && epc%ss.RepsInterval == 0 {
			ss.Log(etime.Analyze, etime.Trial)
		}
	}
	if time == etime.Epoch { // Reset Trial log after Epoch
		ss.Logs.ResetLog(mode, etime.Trial)
	}
}

// LogTrainErrorStats summarizes train errors
func (ss *Sim) LogTrainErrStats() {
	sk := etime.Scope(etime.Train, etime.Trial)
	lt := ss.Logs.TableDetailsScope(sk)
	ix, _ := lt.NamedIdxView("TrainErrors")

	spl := split.GroupBy(ix, []string{"Err"})
	split.Desc(spl, "TrgAct")
	layers := ss.Net.LayersByClass("Hidden", "Target")
	for _, lnm := range layers {
		split.Desc(spl, lnm+"_CosDiff")
	}
	tet := spl.AggsToTable(etable.AddAggName)
	ss.Logs.MiscTables["TrainErrStats"] = tet

	pcterr := agg.Mean(ix, "Err")[0]

	if pcterr > 0 && pcterr < 1 {
		ss.Stats.SetFloat("EpcCorTrgAct", tet.CellFloat("TrgAct:Mean", 0))
		ss.Stats.SetFloat("EpcErrTrgAct", tet.CellFloat("TrgAct:Mean", 1))
	}
}

// LogTestErrors records all errors made across TestTrials, at Test Epoch scope
func (ss *Sim) LogTestErrors() {
	sk := etime.Scope(etime.Test, etime.Trial)
	lt := ss.Logs.TableDetailsScope(sk)
	ix, _ := lt.NamedIdxView("TestErrors")
	ix.Filter(func(et *etable.Table, row int) bool {
		return et.CellFloat("Err", row) > 0 // include error trials
	})
	ss.Logs.MiscTables["TestErrors"] = ix.NewTable()

	allsp := split.All(ix)
	split.Agg(allsp, "UnitErr", agg.AggMean)
	split.Agg(allsp, "CosDiff", agg.AggMean)
	// note: can add other stats to compute
	ss.Logs.MiscTables["TestErrorStats"] = allsp.AggsToTable(etable.AddAggName)
}

// LogRunStats records stats across all runs, at Train Run scope
func (ss *Sim) LogRunStats() {
	sk := etime.Scope(etime.Train, etime.Run)
	lt := ss.Logs.TableDetailsScope(sk)
	ix, _ := lt.NamedIdxView("RunStats")

	spl := split.GroupBy(ix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "PctCor")
	ss.Logs.MiscTables["RunStats"] = spl.AggsToTable(etable.AddAggName)
}

// PCAStats computes PCA statistics on recorded hidden activation patterns
// from Analyze, Trial log data
func (ss *Sim) PCAStats() {
	if ss.UseMPI {
		ss.Logs.MPIGatherTableRows(etime.Analyze, etime.Trial, ss.Comm)
	}
	reps := ss.Logs.IdxView(etime.Analyze, etime.Trial)
	ss.Stats.PCAStats(reps, "ActM", ss.Net.LayersByClass("Hidden", "Target"))
	ss.Logs.ResetLog(etime.Analyze, etime.Trial)
}

// RasterRec updates spike raster record for given cycle
func (ss *Sim) RasterRec(cyc int) {
	ss.Stats.RasterRec(ss.Net, cyc, "Spike")
}

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func (ss *Sim) RunName() string {
	rn := ""
	if ss.Tag != "" {
		rn += ss.Tag + "_"
	}
	rn += ss.Params.Name()
	if ss.StartRun > 0 {
		rn += fmt.Sprintf("_%03d", ss.StartRun)
	}
	return rn
}

// RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func (ss *Sim) RunEpochName(run, epc int) string {
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// WeightsFileName returns default current weights file name
func (ss *Sim) WeightsFileName() string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + ".wts.gz"
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	nm := ss.Net.Nm + "_" + ss.RunName() + "_" + lognm
	if mpi.WorldRank() > 0 {
		nm += fmt.Sprintf("_%d", mpi.WorldRank())
	}
	nm += ".tsv"
	return nm
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	// cam := &(nv.Scene().Camera)
	// cam.Pose.Pos.Set(0.0, 1.733, 2.3)
	// cam.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
	// cam.Pose.Quat.SetFromAxisAngle(mat32.Vec3{-1, 0, 0}, 0.4077744)
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	title := "LVis Object Recognition"
	ss.GUI.MakeWindow(ss, "lvis", title, `This simulation explores how a hierarchy of areas in the ventral stream of visual processing (up to inferotemporal (IT) cortex) can produce robust object recognition that is invariant to changes in position, size, etc of retinal input images. See <a href="https://github.com/ccnlab/lvis/blob/master/sims/lvis_cu3d100_te16deg_axon/README.md">README.md on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ConfigNetView(nv)

	ss.GUI.AddPlots(title, &ss.Logs)

	stb := ss.GUI.TabView.AddNewTab(gi.KiT_Layout, "Spike Rasters").(*gi.Layout)
	stb.Lay = gi.LayoutVert
	stb.SetStretchMax()
	for _, lnm := range ss.Stats.Rasters {
		sr := ss.Stats.F32Tensor("Raster_" + lnm)
		ss.GUI.ConfigRasterGrid(stb, lnm, sr)
	}

	tg := ss.GUI.TabView.AddNewTab(etview.KiT_TensorGrid, "Image").(*etview.TensorGrid)
	tg.SetStretchMax()
	ss.GUI.SetGrid("Image", tg)
	tg.SetTensor(&ss.TrainEnv.Img.Tsr)

	ss.GUI.AddActRFGridTabs(&ss.Stats.ActRFs)

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Train",
		Icon:    "run",
		Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				go ss.Train()
			}
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Stop",
		Icon:    "stop",
		Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.",
		Active:  egui.ActiveRunning,
		Func: func() {
			ss.Stop()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Step Trial",
		Icon:    "step-fwd",
		Tooltip: "Advances one training trial at a time.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.TrainTrial()
				ss.GUI.IsRunning = false
				ss.GUI.UpdateWindow()
			}
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Step Epoch",
		Icon:    "fast-fwd",
		Tooltip: "Advances one epoch (complete set of training patterns) at a time.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				go ss.TrainEpoch()
			}
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Step Run",
		Icon:    "fast-fwd",
		Tooltip: "Advances one full training Run at a time.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				go ss.TrainRun()
			}
		},
	})

	ss.GUI.ToolBar.AddSeparator("spcl")

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Open Trained Wts",
		Icon:    "update",
		Tooltip: "open weights trained on first phase of training (excluding 'novel' objects)",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.OpenTrainedWts()
			ss.GUI.UpdateWindow()
		},
	})

	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("test")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test Trial",
		Icon:    "fast-fwd",
		Tooltip: "Runs the next testing trial.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.TestTrial(false) // don't return on change -- wrap
				ss.GUI.IsRunning = false
				ss.GUI.UpdateWindow()
			}
		},
	})

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test All",
		Icon:    "step-fwd",
		Tooltip: "Tests a large same of testing items and records ActRFs.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				go ss.RunTestAll()
			}
		},
	})

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Conf To Test",
		Icon:    "fast-fwd",
		Tooltip: "Plots accuracy from current confusion probs to test trial log for each category (diagonal of confusion matrix).",
		Active:  egui.ActiveStopped,
		Func: func() {
			giv.CallMethod(ss, "ConfusionTstPlot", ss.GUI.ViewPort)
		},
	})

	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("log")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Reset RunLog",
		Icon:    "reset",
		Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.ResetLog(etime.Train, etime.Run)
			ss.GUI.UpdatePlot(etime.Train, etime.Run)
		},
	})
	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("misc")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "New Seed",
		Icon:    "new",
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.NewRndSeed()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "README",
		Icon:    "file-markdown",
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			gi.OpenURL("https://github.com/lvis/blob/main/sims/lvis_cu3d100_te16deg/README.md")
		},
	})
	ss.GUI.FinalizeGUI(false)
	return ss.GUI.Win
}

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWts", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts,.wts.gz",
				}},
			},
		}},
		{"ConfusionTstPlot", ki.Props{
			"desc": "plot current confusion matrix probs in TstTrlPlot -- enter Cat for confusion row for that category, else if blank, diagonal accuracy for all categories",
			"icon": "file-sheet",
			"Args": ki.PropSlice{
				{"Cat", ki.Props{
					"desc": "category name to show",
				}},
			},
		}},
	},
}

func (ss *Sim) CmdArgs() {
	ss.NoGui = true
	var nogui bool
	var saveEpcLog bool
	var saveRunLog bool
	var saveTrnTrlLog bool
	var saveTstTrlLog bool
	var saveNetData bool
	var note string
	flag.StringVar(&ss.Params.ExtraSets, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.StringVar(&note, "note", "", "user note -- describe the run params etc")
	flag.IntVar(&ss.StartRun, "run", 0, "starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1")
	flag.IntVar(&ss.MaxEpcs, "epcs", 150, "number of epochs per run")
	flag.IntVar(&ss.MaxRuns, "runs", 1, "number of runs to do")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", false, "if true, save run epoch log to file")
	flag.BoolVar(&saveTrnTrlLog, "trntrllog", false, "if true, save training trial log to file")
	flag.BoolVar(&saveTstTrlLog, "tsttrllog", false, "if true, save testing trial log to file")
	flag.BoolVar(&saveNetData, "netdata", false, "if true, save network activation etc data from testing trials, for later viewing in netview")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.BoolVar(&ss.UseMPI, "mpi", false, "if set, use MPI for distributed computation")
	flag.Parse()

	if ss.UseMPI {
		ss.MPIInit()
	}

	// key for Config and Init to be after MPIInit
	ss.Config()
	ss.Init()

	if note != "" {
		mpi.Printf("note: %s\n", note)
	}
	if ss.Params.ExtraSets != "" {
		mpi.Printf("Using Extra Params: %s\n", ss.Params.ExtraSets)
	}

	if saveEpcLog && (ss.SaveProcLog || mpi.WorldRank() == 0) {
		fnm := ss.LogFileName("trn_epc")
		ss.Logs.SetLogFile(etime.Train, etime.Epoch, fnm)
		fnm = ss.LogFileName("tst_epc")
		ss.Logs.SetLogFile(etime.Test, etime.Epoch, fnm)
	}
	if saveTrnTrlLog && (ss.SaveProcLog || mpi.WorldRank() == 0) {
		fnm := ss.LogFileName("trn_trl")
		ss.Logs.SetLogFile(etime.Train, etime.Trial, fnm)
	}
	if saveTstTrlLog && (ss.SaveProcLog || mpi.WorldRank() == 0) {
		fnm := ss.LogFileName("tst_trl")
		ss.Logs.SetLogFile(etime.Test, etime.Trial, fnm)
	}
	if saveRunLog && (ss.SaveProcLog || mpi.WorldRank() == 0) {
		fnm := ss.LogFileName("run")
		ss.Logs.SetLogFile(etime.Train, etime.Run, fnm)
	}
	if ss.SaveWts {
		if mpi.WorldRank() != 0 {
			ss.SaveWts = false
		}
		mpi.Printf("Saving final weights per run\n")
	}
	if saveNetData && mpi.WorldRank() == 0 {
		fmt.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	mpi.Printf("Running %d Runs starting at %d\n", ss.MaxRuns, ss.StartRun)
	ss.TrainEnv.Run.Set(ss.StartRun)
	ss.TrainEnv.Run.Max = ss.StartRun + ss.MaxRuns
	ss.Train()

	ss.Logs.CloseLogFiles()

	if saveNetData && mpi.WorldRank() == 0 {
		ss.GUI.SaveNetData(ss.RunName())
	}

	ss.MPIFinalize()
}

////////////////////////////////////////////////////////////////////
//  MPI code

// MPIInit initializes MPI
func (ss *Sim) MPIInit() {
	mpi.Init()
	var err error
	ss.Comm, err = mpi.NewComm(nil) // use all procs
	if err != nil {
		log.Println(err)
		ss.UseMPI = false
	} else {
		mpi.Printf("MPI running on %d procs\n", mpi.WorldSize())
	}
}

// MPIFinalize finalizes MPI
func (ss *Sim) MPIFinalize() {
	if ss.UseMPI {
		mpi.Finalize()
	}
}

// CollectDWts collects the weight changes from all synapses into AllDWts
// includes all other long adapting factors too: DTrgAvg, ActAvg, etc
func (ss *Sim) CollectDWts(net *axon.Network) {
	net.CollectDWts(&ss.AllDWts)
}

// MPIWtFmDWt updates weights from weight changes, using MPI to integrate
// DWt changes across parallel nodes, each of which are learning on different
// sequences of inputs.
func (ss *Sim) MPIWtFmDWt() {
	if ss.UseMPI {
		ss.CollectDWts(ss.Net)
		ndw := len(ss.AllDWts)
		if len(ss.SumDWts) != ndw {
			ss.SumDWts = make([]float32, ndw)
		}
		ss.Comm.AllReduceF32(mpi.OpSum, ss.SumDWts, ss.AllDWts)
		ss.Net.SetDWts(ss.SumDWts, mpi.WorldSize())
	}
	ss.Net.WtFmDWt(&ss.Time)
}
