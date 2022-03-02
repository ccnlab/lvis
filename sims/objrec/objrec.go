// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
objrec explores how a hierarchy of areas in the ventral stream of visual
processing (up to inferotemporal (IT) cortex) can produce robust object
recognition that is invariant to changes in position, size, etc of retinal
input images.
*/
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/split"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

func main() {
	TheSim.New()
	TheSim.Config()
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			guirun()
		})
	}
}

func guirun() {
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "needs some special inhibition and learning params",
				Params: params.Params{
					"Layer.Act.Dt.IntTau":   "30",  // trying 30 now per lvis
					"Layer.Act.Decay.Act":   "0.0", // 0.2 with glong .6 best in lvis, slows learning here
					"Layer.Act.Decay.Glong": "0.6", // 0.6 def
					// .2, 3 sig better for both Neur and Syn
					"Layer.Act.Dend.GbarExp":    "0.2", // 0.2 > 0.5 > 0.1 > 0
					"Layer.Act.Dend.GbarR":      "3",   // 3 > 6 > 2 good for 0.2 -- too low rel to ExpGbar causes fast ini learning, but then unravels
					"Layer.Act.Dt.GeTau":        "5",   // 5 = 4 (bit slower) > 6 > 7 @176
					"Layer.Act.Dt.LongAvgTau":   "20",  // 20 > 50 > 100
					"Layer.Act.Dt.VmDendTau":    "5",   // 5 much better in fsa!
					"Layer.Act.NMDA.MgC":        "1.4", // mg1, voff0, gbarexp.2, gbarr3 = better
					"Layer.Act.NMDA.Voff":       "5",   // mg1, voff0 = mg1.4, voff5 w best params
					"Layer.Learn.NeurCa.SynTau": "40",  // 40 >= 30 > 20 > 60 (worse start, pca) > 15 (dies) -- some inc in pca top5
					"Layer.Learn.NeurCa.MTau":   "10",  // 40, 10 same as 10, 40 for Neur
					"Layer.Learn.NeurCa.PTau":   "40",
					"Layer.Learn.NeurCa.DTau":   "40",
					"Layer.Learn.NeurCa.LrnThr": "0.01", // .01 faster & better > .02 > .05 (bad) > .1 (very bad)
					"Layer.Learn.NeurCa.VGCCCa": "10",   // 20 seems reasonable, but not obviously better than 0
					"Layer.Learn.NeurCa.CaMax":  "140",
					"Layer.Learn.NeurCa.CaThr":  "0.01",
					"Layer.Learn.LrnNMDA.ITau":  "1",  // urakubo = 100, does not work here..
					"Layer.Learn.LrnNMDA.Tau":   "30", // urakubo = 30 > 20 but no major effect on PCA
				}},
			{Sel: "#V1", Desc: "pool inhib (not used), initial activity",
				Params: params.Params{
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Layer.Gi":    "0.9",  //
					"Layer.Inhib.Pool.Gi":     "0.9",  //
					"Layer.Inhib.ActAvg.Init": "0.08", // .1 for hard clamp, .06 for Ge clamp
					"Layer.Act.Clamp.Ge":      "1.0",  // 1 > .6 lvis
				}},
			{Sel: "#V4", Desc: "pool inhib, sparse activity",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1.0",  // 1.0 == 0.9 == 0.8 > 0.7 > 1.1 (vry bad -- still!!)
					"Layer.Inhib.Pool.Gi":     "1.0",  // 1.0 == 0.9 > 0.8 > 1.1 (vry bad)
					"Layer.Inhib.Pool.On":     "true", // needs pool-level
					"Layer.Inhib.ActAvg.Init": "0.05",
				}},
			{Sel: "#IT", Desc: "initial activity",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1.1",  // 1.1 > 1.0, 1.2
					"Layer.Inhib.ActAvg.Init": "0.05", // .05 > .04 with adapt
					"Layer.Act.GABAB.Gbar":    "0.2",  // .2 > lower (small dif)
				}},
			{Sel: "#Output", Desc: "high inhib for one-hot output",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1.5", // 1.5 = 1.6 > 1.4 adapt
					"Layer.Inhib.ActAvg.Init": "0.05",
					"Layer.Act.Clamp.Ge":      "0.6", // .6 generally = .5
				}},
			{Sel: "Prjn", Desc: "yes extra learning factors",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":      "0.1",   // SynSpk: .1 > .15 > 0.05 > 0.08 > .2 (.2 for NeurSpk)
					"Prjn.SWt.Adapt.Lrate":       "0.005", // 0.005 > others maybe?  0.02 > 0.05 > .1
					"Prjn.SWt.Init.SPct":         "1",     // 1 >= lower
					"Prjn.Com.PFail":             "0.0",
					"Prjn.Learn.Kinase.SpikeG":   "12", // 8 is target for SynSpk, SynNMDA
					"Prjn.Learn.Kinase.Rule":     "SynSpkCa",
					"Prjn.Learn.Kinase.OptInteg": "true",
					"Prjn.Learn.Kinase.MTau":     "5", // 5 > 2 > 1 for PCA Top5, no perf diff
					"Prjn.Learn.Kinase.PTau":     "40",
					"Prjn.Learn.Kinase.DTau":     "40",
					"Prjn.Learn.Kinase.DScale":   "1",
					"Prjn.Learn.Kinase.MaxISI":   "100", // 50 = 80 = 100, but 50 slightly faster
					"Prjn.Learn.XCal.On":         "true",
					"Prjn.Learn.XCal.PThrMin":    "0.05", // .1 (at end) > 0.05 > 0.02 > 0.01
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates -- smaller as network gets bigger",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2",  // .2 >= .3 > .15 > .1 > .05 @176
					"Prjn.Learn.Learn":   "true", // keep random weights to enable exploration
					// "Prjn.Learn.Lrate.Base":      "0.04", // lrate = 0 allows syn scaling still
				}},
			{Sel: ".Forward", Desc: "special forward-only params: com prob",
				Params: params.Params{}},
			{Sel: ".Inhib", Desc: "inhibitory projection",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base": "0.01", // 0.0001 best for lvis
					"Prjn.SWt.Adapt.On":     "false",
					"Prjn.SWt.Init.Var":     "0.0",
					"Prjn.SWt.Init.Mean":    "0.1",
					"Prjn.PrjnScale.Abs":    "0.1", // .1 from lvis
					"Prjn.PrjnScale.Adapt":  "false",
					"Prjn.IncGain":          "0.5",
				}},
			{Sel: "#ITToOutput", Desc: "no random sampling here",
				Params: params.Params{
					"Prjn.Com.PFail": "0.0",
				}},
		},
	}},
	{Name: "NovelLearn", Desc: "learning for novel objects case -- IT, Output connections learn", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "lr = 0",
				Params: params.Params{
					"Prjn.Learn.Lrate":     "0",
					"Prjn.Learn.LrateInit": "0", // make sure for sched
				}},
			{Sel: ".NovLearn", Desc: "lr = 0.04",
				Params: params.Params{
					"Prjn.Learn.Lrate":     "0.04",
					"Prjn.Learn.LrateInit": "0.04", // double sure
				}},
		},
	}},
	{Name: "NeurSpkCa", Desc: "NeurSpkCa version", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "yes extra learning factors",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":   "0.2", // SynSpk: .1 > .15 > 0.05 > 0.08 > .2 (.2 for NeurSpk)
					"Prjn.Learn.Kinase.Rule":  "NeurSpkCa",
					"Prjn.Learn.XCal.On":      "true",
					"Prjn.Learn.XCal.PThrMin": "0.05", // .1 (at end) > 0.05 > 0.02 > 0.01
				}},
		},
	}},
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net           *axon.Network   `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Params        emer.Params     `view:"inline" desc:"all parameter management"`
	Tag           string          `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	Stats         estats.Stats    `desc:"contains computed statistic values"`
	Logs          elog.Logs       `desc:"Contains all the logs and information about the logs.'"`
	V1V4Prjn      *prjn.PoolTile  `view:"projection from V1 to V4 which is tiled 4x4 skip 2 with topo scale values"`
	StartRun      int             `desc:"starting run number -- typically 0 but can be set in command args for parallel runs on a cluster"`
	MaxRuns       int             `desc:"maximum number of model runs to perform"`
	MaxEpcs       int             `desc:"maximum number of epochs to run per model run"`
	MaxTrls       int             `desc:"maximum number of training trials per epoch"`
	RepsInterval  int             `desc:"how often to analyze the representations"`
	NZeroStop     int             `desc:"if a positive number, training will stop after this many epochs with zero UnitErr"`
	MiniBatches   int             `desc:"number of trials to aggregate into DWt before applying"`
	TrainEnv      LEDEnv          `desc:"Training environment -- LED training"`
	PNovel        float32         `desc:"proportion of novel training items to use -- set this to 0.5 after initial training"`
	NovelTrainEnv LEDEnv          `desc:"Novel items training environment -- LED training"`
	TestEnv       LEDEnv          `desc:"Testing environment -- LED testing"`
	Time          axon.Time       `desc:"axon timing parameters and state"`
	ViewOn        bool            `desc:"whether to update the network view while running"`
	TrainUpdt     axon.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt      axon.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`

	MiniBatchCtr int `inactive:"+" desc:"counter for mini-batch learning"`

	// internal state - view:"-"
	GUI          egui.GUI `view:"-" desc:"manages all the gui elements"`
	SaveWts      bool     `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui        bool     `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams bool     `view:"-" desc:"if true, print message for all params that are set"`
	IsRunning    bool     `view:"-" desc:"true if sim is running"`
	StopNow      bool     `view:"-" desc:"flag to stop running"`
	NeedsNewRun  bool     `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeeds     []int64  `view:"-" desc:"a list of random seeds to use for each run"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	ss.Net = &axon.Network{}
	ss.Params.Params = ParamSets
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()
	ss.V1V4Prjn = prjn.NewPoolTile()
	ss.V1V4Prjn.Size.Set(4, 4)
	ss.V1V4Prjn.Skip.Set(2, 2)
	ss.V1V4Prjn.Start.Set(-1, -1)
	ss.V1V4Prjn.TopoRange.Min = 0.8 // note: none of these make a very big diff
	// but using a symmetric scale range .8 - 1.2 seems like it might be good -- otherwise
	// weights are systematicaly smaller.
	// ss.V1V4Prjn.GaussFull.DefNoWrap()
	// ss.V1V4Prjn.GaussInPool.DefNoWrap()
	ss.RndSeeds = make([]int64, 100) // make enough for plenty of runs
	for i := 0; i < 100; i++ {
		ss.RndSeeds[i] = int64(i) + 1 // exclude 0
	}
	ss.ViewOn = true
	ss.TrainUpdt = axon.GammaCycle
	ss.TestUpdt = axon.GammaCycle
	ss.PNovel = 0
	ss.MiniBatches = 1 // 1 > 16
	ss.RepsInterval = 10

	ss.Time.Defaults()
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
		ss.MaxEpcs = 50
		ss.NZeroStop = -1
	}
	if ss.MaxTrls == 0 { // allow user override
		ss.MaxTrls = 100
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Defaults()
	ss.TrainEnv.MinLED = 0
	ss.TrainEnv.MaxLED = 17 // exclude last 2 by default
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually
	ss.TrainEnv.Trial.Max = ss.MaxTrls

	ss.NovelTrainEnv.Nm = "NovelTrainEnv"
	ss.NovelTrainEnv.Dsc = "novel items training params and state"
	ss.NovelTrainEnv.Defaults()
	ss.NovelTrainEnv.MinLED = 18
	ss.NovelTrainEnv.MaxLED = 19 // only last 2 items
	ss.NovelTrainEnv.Validate()
	ss.NovelTrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually
	ss.NovelTrainEnv.Trial.Max = ss.MaxTrls
	ss.NovelTrainEnv.XFormRand.TransX.Set(-0.125, 0.125)
	ss.NovelTrainEnv.XFormRand.TransY.Set(-0.125, 0.125)
	ss.NovelTrainEnv.XFormRand.Scale.Set(0.775, 0.925) // 1/2 around midpoint
	ss.NovelTrainEnv.XFormRand.Rot.Set(-2, 2)

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Defaults()
	ss.TestEnv.MinLED = 0
	ss.TestEnv.MaxLED = 19    // all by default
	ss.TestEnv.Trial.Max = 50 // 0 // 1000 is too long!
	ss.TestEnv.Validate()

	ss.TrainEnv.Init(0)
	ss.NovelTrainEnv.Init(0)
	ss.TestEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "Objrec")
	v1 := net.AddLayer4D("V1", 10, 10, 5, 4, emer.Input)
	v4 := net.AddLayer4D("V4", 5, 5, 10, 10, emer.Hidden) // 10x10 == 16x16 > 7x7 (orig)
	it := net.AddLayer2D("IT", 16, 16, emer.Hidden)       // 16x16 == 20x20 > 10x10 (orig)
	out := net.AddLayer4D("Output", 4, 5, ss.TrainEnv.NOutPer, 1, emer.Target)

	v1.SetRepIdxs(emer.CenterPoolIdxs(v1, 2))
	v4.SetRepIdxs(emer.CenterPoolIdxs(v4, 2))

	full := prjn.NewFull()
	_ = full
	rndprjn := prjn.NewUnifRnd() // no advantage
	rndprjn.PCon = 0.5           // 0.2 > .1
	_ = rndprjn

	pool1to1 := prjn.NewPoolOneToOne()
	_ = pool1to1

	net.ConnectLayers(v1, v4, ss.V1V4Prjn, emer.Forward)
	v4IT, _ := net.BidirConnectLayers(v4, it, full)
	itOut, outIT := net.BidirConnectLayers(it, out, full)

	// net.LateralConnectLayerPrjn(v4, pool1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
	// net.LateralConnectLayerPrjn(it, full, &axon.HebbPrjn{}).SetType(emer.Inhib)

	it.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "V4", YAlign: relpos.Front, Space: 2})
	out.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "IT", YAlign: relpos.Front, Space: 2})

	v4IT.SetClass("NovLearn")
	itOut.SetClass("NovLearn")
	outIT.SetClass("NovLearn")

	// about the same on mac with and without threading
	// v4.SetThread(1)
	// it.SetThread(2)

	net.Defaults()
	ss.Params.SetObject("Network")
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	ss.InitWts(net)
}

func (ss *Sim) InitWts(net *axon.Network) {
	net.InitWts()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.InitRndSeed()
	ss.TrainEnv.Run.Max = ss.MaxRuns
	ss.StopNow = false
	ss.Params.SetAll()
	ss.Net.SlowInterval = 100 // / ss.MiniBatches
	ss.NewRun()
	ss.GUI.UpdateNetView()
}

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

func (ss *Sim) UpdateViewTime(train bool, viewUpdt axon.TimeScales) {
	switch viewUpdt {
	case axon.Cycle:
		ss.GUI.UpdateNetView()
	case axon.FastSpike:
		if ss.Time.Cycle%10 == 0 {
			ss.GUI.UpdateNetView()
		}
	case axon.GammaCycle:
		if ss.Time.Cycle%25 == 0 {
			ss.GUI.UpdateNetView()
		}
	case axon.AlphaCycle:
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
	if !train {
		viewUpdt = ss.TestUpdt
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	if train {
		ss.MiniBatchCtr++
		if ss.MiniBatchCtr >= ss.MiniBatches {
			ss.MiniBatchCtr = 0
			ss.Net.WtFmDWt(&ss.Time)
		}
	}

	minusCyc := 150
	plusCyc := 50

	ss.Net.NewState()
	ss.Time.NewState(train)
	for cyc := 0; cyc < minusCyc; cyc++ { // do the minus phase
		ss.Net.Cycle(&ss.Time)
		ss.StatCounters(train)
		// if !train {
		// ss.Log(elog.Test, elog.Cycle)
		// }
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
			ss.UpdateViewTime(train, viewUpdt)
		}
	}
	ss.Time.NewPhase()
	ss.StatCounters(train)
	if viewUpdt == axon.Phase {
		ss.GUI.UpdateNetView()
	}
	for cyc := 0; cyc < plusCyc; cyc++ { // do the plus phase
		ss.Net.Cycle(&ss.Time)
		ss.StatCounters(train)
		// if !train {
		// ss.Log(elog.Test, elog.Cycle)
		// }
		if !ss.NoGui {
			ss.RasterRec(ss.Time.Cycle)
		}
		ss.Time.CycleInc()

		if cyc == plusCyc-1 { // do before view update
			ss.Net.PlusPhase(&ss.Time)
		}
		if ss.ViewOn {
			ss.UpdateViewTime(train, viewUpdt)
		}
	}
	ss.TrialStats(train)
	ss.StatCounters(train)

	if train {
		ss.Net.DWt(&ss.Time)
	}
	if viewUpdt == axon.Phase || viewUpdt == axon.AlphaCycle || viewUpdt == axon.ThetaCycle {
		ss.GUI.UpdateNetView()
	}

	// if ss.TstCycPlot != nil && !train {
	// ss.GUI.UpdatePlot(elog.Test, elog.Cycle) // make sure always updated at end
	// }
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"V1", "Output"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// TrainTrial runs one trial of training using TrainEnv
func (ss *Sim) TrainTrial() {
	if ss.NeedsNewRun {
		ss.NewRun()
	}

	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state
	if ss.PNovel > 0 {
		ss.NovelTrainEnv.Step() // keep in sync
	}

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		if (ss.RepsInterval > 0) && ((epc-1)%ss.RepsInterval == 0) { // -1 so runs on first epc
			ss.PCAStats()
		}
		ss.Log(elog.Train, elog.Epoch)
		ss.LrateSched(epc)
		if ss.ViewOn && ss.TrainUpdt > axon.AlphaCycle {
			ss.GUI.UpdateNetView()
		}
		// if (ss.TestInterval > 0) && (epc%ss.TestInterval == 0) { // note: epc is *next* so won't trigger first time
		// 	ss.TestAll()
		// }
		if epc >= ss.MaxEpcs || (ss.NZeroStop > 0 && ss.Stats.Int("NZero") >= ss.NZeroStop) {
			// done with training..
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.StopNow = true
				return
			} else {
				ss.NeedsNewRun = true
				return
			}
		}
	}

	// note: type must be in place before apply inputs
	ss.Net.LayerByName("Output").SetType(emer.Target)
	if erand.BoolP(ss.PNovel) {
		ss.ApplyInputs(&ss.NovelTrainEnv)
	} else {
		ss.ApplyInputs(&ss.TrainEnv)
	}
	ss.ThetaCyc(true) // train
	ss.Log(elog.Train, elog.Trial)
	if ss.RepsInterval > 0 && epc%ss.RepsInterval == 0 {
		ss.Log(elog.Analyze, elog.Trial)
	}
	if ss.GUI.IsRunning {
		ss.GUI.Grid("Image").SetTensor(&ss.TrainEnv.Vis.ImgTsr)
	}
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.Log(elog.Train, elog.Run)
	if ss.SaveWts {
		fnm := ss.WeightsFileName()
		fmt.Printf("Saving Weights to: %s\n", fnm)
		ss.Net.SaveWtsJSON(gi.FileName(fnm))
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	ss.MiniBatchCtr = 0
	run := ss.TrainEnv.Run.Cur
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
	ss.Time.Reset()
	ss.InitWts(ss.Net)
	ss.InitStats()
	ss.StatCounters(true)
	ss.Logs.ResetLog(elog.Train, elog.Trial)
	ss.Logs.ResetLog(elog.Train, elog.Epoch)
	ss.Logs.ResetLog(elog.Test, elog.Epoch)
	ss.NeedsNewRun = false
}

// TrainEpoch runs training trials for remainder of this epoch
func (ss *Sim) TrainEpoch() {
	ss.StopNow = false
	curEpc := ss.TrainEnv.Epoch.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Epoch.Cur != curEpc {
			break
		}
	}
	ss.Stopped()
}

// TrainRun runs training trials for remainder of run
func (ss *Sim) TrainRun() {
	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()
}

// Train runs the full training from this point onward
func (ss *Sim) Train() {
	ss.StopNow = false
	for {
		ss.TrainTrial()
		if ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.GUI.Stopped()
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

// LrateSched implements the learning rate schedule
func (ss *Sim) LrateSched(epc int) {
	switch epc {
	case 70:
		ss.Net.LrateSched(0.5)
		fmt.Printf("dropped lrate 0.5 at epoch: %d\n", epc)
	case 80:
		ss.Net.LrateSched(0.2)
		fmt.Printf("dropped lrate 0.2 at epoch: %d\n", epc)
	}
}

// OpenTrainedWts opens trained weights
func (ss *Sim) OpenTrainedWts() {
	// ab, err := Asset("objrec_train1.wts") // embedded in executable
	// if err != nil {
	// 	log.Println(err)
	// }
	// ss.Net.ReadWtsJSON(bytes.NewBuffer(ab))
	ss.Net.OpenWtsJSON("objrec_train1.wts.gz")
}

// TrainNovel prepares network for training novel items: loads saved weights
// changes PNovel -- just do Step Run after this.
func (ss *Sim) TrainNovel() {
	ss.NewRun()
	ss.OpenTrainedWts()
	ss.Params.SetAllSet("NovelLearn")
	ss.TrainEnv.Epoch.Cur = 40
	ss.LrateSched(40)
	ss.PNovel = 0.5
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("TrlErr", 0.0)
	ss.Stats.SetFloat("TrlErr2", 0.0)
	ss.Stats.SetFloat("TrlUnitErr", 0.0)
	ss.Stats.SetFloat("TrlCosDiff", 0.0)
	ss.Stats.SetFloat("TrlTrgAct", 0.0)
	ss.Stats.SetString("TrlOut", "")
	ss.Stats.SetInt("FirstZero", -1) // critical to reset to -1
	ss.Stats.SetInt("NZero", 0)
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

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) {
	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()
	ss.Stats.SetFloat("TrlCosDiff", float64(out.CosDiff.Cos))
	ss.Stats.SetFloat("TrlUnitErr", out.PctUnitErr())

	ev := ss.TrainEnv
	if !accum {
		ev = ss.TestEnv
	}

	ovt := ss.Stats.SetLayerTensor(ss.Net, "Output", "ActM")
	rsp, trlErr, trlErr2 := ev.OutErr(ovt)
	ss.Stats.SetFloat("TrlErr", trlErr)
	ss.Stats.SetFloat("TrlErr2", trlErr2)
	ss.Stats.SetString("TrlOut", fmt.Sprintf("%d", rsp))
	ss.Stats.SetFloat("TrlTrgAct", float64(out.Pools[0].ActP.Avg))

	ss.Stats.SetString("Cat", fmt.Sprintf("%d", ev.CurLED))
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial(returnOnChg bool) {
	ss.TestEnv.Step()

	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdt > axon.AlphaCycle {
			ss.GUI.UpdateNetView()
		}
		ss.Log(elog.Test, elog.Epoch)
		if returnOnChg {
			return
		}
	}

	// note: type must be in place before apply inputs
	ss.Net.LayerByName("Output").SetType(emer.Compare)
	ss.ApplyInputs(&ss.TestEnv)
	ss.ThetaCyc(false) // !train
	ss.Log(elog.Test, elog.Trial)
	if ss.GUI.IsRunning {
		ss.GUI.Grid("Image").SetTensor(&ss.TestEnv.Vis.ImgTsr)
	}
	ss.Stats.UpdateActRFs(ss.Net, "ActM", 0.01)
	ss.GUI.NetDataRecord()
}

// TestItem tests given item which is at given index in test item list
func (ss *Sim) TestItem(idx int) {
	cur := ss.TestEnv.Trial.Cur
	ss.TestEnv.Trial.Cur = idx
	ss.TestEnv.DoObject(idx)
	ss.ApplyInputs(&ss.TestEnv)
	ss.ThetaCyc(false) // !train
	ss.TestEnv.Trial.Cur = cur
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	ss.Stats.ActRFs.Reset()
	for {
		ss.TestTrial(true) // return on chg, don't present
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}
	ss.Stats.ActRFsAvgNorm()
	ss.GUI.ViewActRFs(&ss.Stats.ActRFs)
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.ConfigLogItems()
	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(elog.Test, elog.Cycle)
	ss.Logs.NoPlot(elog.Train, elog.Cycle)
	ss.Logs.NoPlot(elog.Test, elog.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(elog.Train, elog.Run, "LegendCol", "Params")
	ss.Stats.ConfigRasters(ss.Net, ss.Net.LayersByClass()) // all

	ss.Stats.SetF32Tensor("Image", &ss.TestEnv.Vis.ImgTsr) // image used for actrfs, must be there first
	ss.Stats.InitActRFs(ss.Net, []string{"V4:Image", "V4:Output", "IT:Image", "IT:Output"}, "ActM")

	// reshape v4 tensor for inner 2x2 set of representative units
	v4 := ss.Net.LayerByName("V4").(axon.AxonLayer).AsAxon()
	ss.Stats.F32Tensor("V4").SetShape([]int{2, 2, v4.Shp.Dim(2), v4.Shp.Dim(3)}, nil, nil)
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode elog.EvalModes, time elog.Times) {
	dt := ss.Logs.Table(mode, time)
	row := dt.Rows
	switch {
	case mode == elog.Test && time == elog.Epoch:
		ss.LogTestErrors()
	case mode == elog.Train && time == elog.Epoch:
		ss.LogTrainErrStats()
	case time == elog.Cycle:
		row = ss.Stats.Int("Cycle")
	case time == elog.Trial:
		row = ss.Stats.Int("Trial")
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
	if time == elog.Cycle {
		ss.GUI.UpdateCyclePlot(elog.Test, ss.Time.Cycle)
	} else {
		ss.GUI.UpdatePlot(mode, time)
	}

	switch {
	case mode == elog.Train && time == elog.Run:
		ss.LogRunStats()
	}
}

// LogTrainErrorStats summarizes train errors
func (ss *Sim) LogTrainErrStats() {
	sk := elog.Scope(elog.Train, elog.Trial)
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
	sk := elog.Scope(elog.Test, elog.Trial)
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
	sk := elog.Scope(elog.Train, elog.Run)
	lt := ss.Logs.TableDetailsScope(sk)
	ix, _ := lt.NamedIdxView("RunStats")

	spl := split.GroupBy(ix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "PctCor")
	ss.Logs.MiscTables["RunStats"] = spl.AggsToTable(etable.AddAggName)
}

// HogDead computes the proportion of units in given layer name with ActAvg over hog thr
// and under dead threshold.  Also reports max Gnmda and GgabaB
func (ss *Sim) HogDead(lnm string) (hog, dead, gnmda, ggabab float64) {
	ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
	n := len(ly.Neurons)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.ActAvg > 0.3 {
			hog += 1
		} else if nrn.ActAvg < 0.01 {
			dead += 1
		}
		gnmda = math.Max(gnmda, float64(nrn.Gnmda))
		ggabab = math.Max(ggabab, float64(nrn.GgabaB))
	}
	hog /= float64(n)
	dead /= float64(n)
	return
}

// PCAStats computes PCA statistics on recorded hidden activation patterns
// from Analyze, Trial log data
func (ss *Sim) PCAStats() {
	ss.Stats.PCAStats(ss.Logs.IdxView(elog.Analyze, elog.Trial), "ActM", ss.Net.LayersByClass("Hidden", "Target"))
	ss.Logs.ResetLog(elog.Analyze, elog.Trial)
}

// RasterRec updates spike raster record for given cycle
func (ss *Sim) RasterRec(cyc int) {
	ss.Stats.RasterRec(ss.Net, cyc, "Spike", ss.Net.LayersByClass())
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
	return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".tsv"
}

/*
func (ss *Sim) LogTstEpc(dt *etable.Table) {
	trl := ss.TstTrlLog
	tix := etable.NewIdxView(trl)
	// epc := ss.TrainEnv.Epoch.Prv // ?

	spl := split.GroupBy(tix, []string{"Obj"})
	_, err := split.AggTry(spl, "Err", agg.AggMean)
	if err != nil {
		log.Println(err)
	}
	objs := spl.AggsToTable(etable.AddAggName)
	no := objs.Rows
	dt.SetNumRows(no)
	for i := 0; i < no; i++ {
		dt.SetCellFloat("Obj", i, float64(i))
		dt.SetCellFloat("PctErr", i, objs.Cols[1].FloatVal1D(i))
	}
	ss.TstEpcPlot.GoUpdate()
}

func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Obj", etensor.INT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Object Recognition Testing Epoch Plot"
	plt.Params.XAxisCol = "Obj"
	plt.Params.Type = eplot.Bar
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Obj", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}
*/

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	cam := &(nv.Scene().Camera)
	cam.Pose.Pos.Set(0.0, 1.733, 2.3)
	cam.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
	// cam.Pose.Quat.SetFromAxisAngle(mat32.Vec3{-1, 0, 0}, 0.4077744)
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	title := "Object Recognition"
	ss.GUI.MakeWindow(ss, "objrec", title, `This simulation explores how a hierarchy of areas in the ventral stream of visual processing (up to inferotemporal (IT) cortex) can produce robust object recognition that is invariant to changes in position, size, etc of retinal input images. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch6/objrec/README.md">README.md on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10
	ss.GUI.NetView.SetNet(ss.Net)

	ss.GUI.NetView.Scene().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	ss.GUI.NetView.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
	ss.ConfigNetView(ss.GUI.NetView)

	ss.GUI.AddPlots(title, &ss.Logs)

	stb := ss.GUI.TabView.AddNewTab(gi.KiT_Layout, "Spike Rasters").(*gi.Layout)
	stb.Lay = gi.LayoutVert
	stb.SetStretchMax()
	layers := ss.Net.LayersByClass() // all
	for _, lnm := range layers {
		sr := ss.Stats.F32Tensor("Raster_" + lnm)
		ss.GUI.ConfigRasterGrid(stb, lnm, sr)
	}

	tg := ss.GUI.TabView.AddNewTab(etview.KiT_TensorGrid, "Image").(*etview.TensorGrid)
	tg.SetStretchMax()
	ss.GUI.SetGrid("Image", tg)
	tg.SetTensor(&ss.TrainEnv.Vis.ImgTsr)

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

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Train Novel",
		Icon:    "update",
		Tooltip: "prepares network for training novel items: loads saved weight, changes PNovel -- just do Step Run after this..",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.TrainNovel()
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

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test Item",
		Icon:    "step-fwd",
		Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.",
		Active:  egui.ActiveStopped,
		Func: func() {
			gi.StringPromptDialog(ss.GUI.ViewPort, "", "Test Item",
				gi.DlgOpts{Title: "Test Item", Prompt: "Enter the Name of a given input pattern to test (case insensitive, contains given string."},
				ss.GUI.Win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
					dlg := send.(*gi.Dialog)
					if sig == int64(gi.DialogAccepted) {
						val := gi.StringPromptDialogValue(dlg)
						idx, _ := strconv.Atoi(val)
						if !ss.GUI.IsRunning {
							ss.GUI.IsRunning = true
							fmt.Printf("testing index: %d\n", idx)
							ss.TestItem(idx)
							ss.GUI.IsRunning = false
							ss.GUI.UpdateWindow()
						}
					}
				})
		},
	})

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test All",
		Icon:    "step-fwd",
		Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				go ss.RunTestAll()
			}
		},
	})

	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("log")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Reset RunLog",
		Icon:    "reset",
		Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.ResetLog(elog.Train, elog.Run)
			ss.GUI.UpdatePlot(elog.Train, elog.Run)
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
			gi.OpenURL("https://github.com/ccnlab/lvis/blob/master/sims/objrec/README.md")
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
	},
}

func (ss *Sim) CmdArgs() {
	ss.NoGui = true
	var nogui bool
	var saveEpcLog bool
	var saveRunLog bool
	var saveNetData bool
	var note string
	flag.StringVar(&ss.Params.ExtraSets, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.StringVar(&note, "note", "", "user note -- describe the run params etc")
	flag.IntVar(&ss.StartRun, "run", 0, "starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1")
	flag.IntVar(&ss.MaxRuns, "runs", 1, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.IntVar(&ss.MaxEpcs, "epcs", 50, "number of epochs per run")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", false, "if true, save run epoch log to file")
	flag.BoolVar(&saveNetData, "netdata", false, "if true, save network activation etc data from testing trials, for later viewing in netview")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.Parse()
	ss.Init()

	if note != "" {
		fmt.Printf("note: %s\n", note)
	}
	if ss.Params.ExtraSets != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.Params.ExtraSets)
	}

	if saveEpcLog {
		fnm := ss.LogFileName("epc")
		ss.Logs.SetLogFile(elog.Train, elog.Epoch, fnm)
	}
	if saveRunLog {
		fnm := ss.LogFileName("run")
		ss.Logs.SetLogFile(elog.Train, elog.Run, fnm)
	}
	if ss.SaveWts {
		fmt.Printf("Saving final weights per run\n")
	}
	if saveNetData {
		fmt.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}
	fmt.Printf("Running Runs: %d - %d\n", ss.StartRun, ss.MaxRuns)
	ss.TrainEnv.Run.Set(ss.StartRun)
	ss.TrainEnv.Run.Max = ss.MaxRuns
	ss.NewRun()
	ss.Train()

	ss.Logs.CloseLogFiles()

	if saveNetData {
		ss.GUI.SaveNetData(ss.RunName())
	}
}
