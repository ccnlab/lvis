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
	"fmt"
	"log"
	"os"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/ecmd"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/etable"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/mat32"
)

// Debug triggers various messages etc
var Debug = false

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

// see params.go for params

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net          *axon.Network    `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Params       emer.Params      `view:"inline" desc:"all parameter management"`
	Loops        *looper.Manager  `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats        estats.Stats     `desc:"contains computed statistic values"`
	Logs         elog.Logs        `desc:"Contains all the logs and information about the logs.'"`
	Pats         *etable.Table    `view:"no-inline" desc:"the training patterns to use"`
	Envs         env.Envs         `view:"no-inline" desc:"Environments"`
	Time         axon.Time        `desc:"axon timing parameters and state"`
	ViewUpdt     netview.ViewUpdt `view:"inline" desc:"netview update parameters"`
	TestInterval int              `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	PCAInterval  int              `desc:"how frequently (in epochs) to compute PCA on hidden representations to measure variance?"`
	V1V4Prjn     *prjn.PoolTile   `view:"projection from V1 to V4 which is tiled 4x4 skip 2 with topo scale values"`
	NOutPer      int              `desc:"number of units per localist output unit"`

	GUI      egui.GUI    `view:"-" desc:"manages all the gui elements"`
	Args     ecmd.Args   `view:"no-inline" desc:"command line args"`
	RndSeeds erand.Seeds `view:"-" desc:"a list of random seeds to use for each run"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	ss.Params.Params = ParamSets
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()
	ss.Pats = &etable.Table{}
	ss.RndSeeds.Init(100) // max 100 runs
	ss.NOutPer = 5
	ss.TestInterval = 5
	ss.PCAInterval = 5
	ss.Time.Defaults()
	ss.ConfigArgs() // do this first, has key defaults
	ss.NewPrjns()
}

// New creates new blank elements and initializes defaults
func (ss *Sim) NewPrjns() {
	ss.V1V4Prjn = prjn.NewPoolTile()
	ss.V1V4Prjn.Size.Set(4, 4)
	ss.V1V4Prjn.Skip.Set(2, 2)
	ss.V1V4Prjn.Start.Set(-1, -1)
	ss.V1V4Prjn.TopoRange.Min = 0.8 // note: none of these make a very big diff
	// but using a symmetric scale range .8 - 1.2 seems like it might be good -- otherwise
	// weights are systematicaly smaller.
	// ss.V1V4Prjn.GaussFull.DefNoWrap()
	// ss.V1V4Prjn.GaussInPool.DefNoWrap()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn, novTrn, tst *LEDEnv
	if len(ss.Envs) == 0 {
		trn = &LEDEnv{}
		novTrn = &LEDEnv{}
		tst = &LEDEnv{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*LEDEnv)
		novTrn = ss.Envs.ByMode(etime.Analyze).(*LEDEnv)
		tst = ss.Envs.ByMode(etime.Test).(*LEDEnv)
	}

	trn.Nm = etime.Train.String()
	trn.Dsc = "training params and state"
	trn.Defaults()
	trn.MinLED = 0
	trn.MaxLED = 17 // exclude last 2 by default
	trn.NOutPer = ss.NOutPer
	trn.Validate()
	trn.Trial.Max = 100

	novTrn.Nm = etime.Analyze.String()
	novTrn.Dsc = "novel items training params and state"
	novTrn.Defaults()
	novTrn.MinLED = 18
	novTrn.MaxLED = 19 // only last 2 items
	novTrn.NOutPer = ss.NOutPer
	novTrn.Validate()
	novTrn.Trial.Max = 100
	novTrn.XFormRand.TransX.Set(-0.125, 0.125)
	novTrn.XFormRand.TransY.Set(-0.125, 0.125)
	novTrn.XFormRand.Scale.Set(0.775, 0.925) // 1/2 around midpoint
	novTrn.XFormRand.Rot.Set(-2, 2)

	tst.Nm = etime.Test.String()
	tst.Dsc = "testing params and state"
	tst.Defaults()
	tst.MinLED = 0
	tst.MaxLED = 19 // all by default
	tst.NOutPer = ss.NOutPer
	tst.Trial.Max = 50 // 0 // 1000 is too long!
	tst.Validate()

	trn.Init(0)
	novTrn.Init(0)
	tst.Init(0)

	ss.Envs.Add(trn, novTrn, tst)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "Objrec")
	v1 := net.AddLayer4D("V1", 10, 10, 5, 4, emer.Input)
	v4 := net.AddLayer4D("V4", 5, 5, 10, 10, emer.Hidden) // 10x10 == 16x16 > 7x7 (orig)
	it := net.AddLayer2D("IT", 16, 16, emer.Hidden)       // 16x16 == 20x20 > 10x10 (orig)
	out := net.AddLayer4D("Output", 4, 5, ss.NOutPer, 1, emer.Target)

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
	net.InitWts()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Loops.ResetCounters()
	ss.InitRndSeed()
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.Params.SetAll()
	ss.NewRun()
	ss.ViewUpdt.Update()
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed() {
	run := ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur
	ss.RndSeeds.Set(run)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	man.AddStack(etime.Train).AddTime(etime.Run, 1).AddTime(etime.Epoch, 50).AddTime(etime.Trial, 100).AddTime(etime.Cycle, 200)

	man.AddStack(etime.Test).AddTime(etime.Epoch, 1).AddTime(etime.Trial, 100).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Time, ss.Net.AsAxon(), 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net.AsAxon(), &ss.Time, &ss.ViewUpdt) // std algo code

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("Sim:Env:Step", func() {
			// note: OnStart for env.Env, others may happen OnEnd
			ss.Envs[mode.String()].Step()
		})
		stack.Loops[etime.Trial].OnStart.Add("Sim:ApplyInputs", func() {
			ss.ApplyInputs()
			// axon.EnvApplyInputs(ss.Net, ss.Envs[ss.Time.Mode])
		})
		stack.Loops[etime.Trial].OnEnd.Add("Sim:StatCounters", ss.StatCounters)
		stack.Loops[etime.Trial].OnEnd.Add("Sim:TrialStats", ss.TrialStats)
	}

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("Sim:NewRun", ss.NewRun)

	// Train stop early condition
	man.GetLoop(etime.Train, etime.Epoch).IsDone["Epoch:NZeroStop"] = func() bool {
		// This is calculated in TrialStats
		stopNz := ss.Args.Int("nzero")
		if stopNz <= 0 {
			stopNz = 2
		}
		curNZero := ss.Stats.Int("NZero")
		stop := curNZero >= stopNz
		return stop
	}

	// Add Testing
	trainEpoch := man.GetLoop(etime.Train, etime.Epoch)
	trainEpoch.OnStart.Add("Log:Train:TestAtInterval", func() {
		if (ss.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.TestInterval == 0) {
			// Note the +1 so that it doesn't occur at the 0th timestep.
			ss.TestAll()
		}
	})

	/////////////////////////////////////////////
	// Logging

	man.GetLoop(etime.Test, etime.Epoch).OnEnd.Add("Test:Epoch:LogTestErrors", func() {
		axon.LogTestErrors(&ss.Logs)
	})
	man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("Train:Epoch:PCAStats", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if ss.PCAInterval > 0 && trnEpc%ss.PCAInterval == 0 {
			axon.PCAStats(ss.Net.AsAxon(), &ss.Logs, &ss.Stats)
		}
	})

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)

	man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("Train:Trial:LogAnalyze", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.PCAInterval > 0) && (trnEpc%ss.PCAInterval == 0) {
			ss.Log(etime.Analyze, etime.Trial)
		}
	})

	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("Train:Run:RunStats", func() {
		ss.Logs.RunStats("PctCor", "FirstZero", "LastZero")
	})

	// Save weights to file, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("Log:Train:SaveWeights", func() {
		ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		axon.SaveWeightsIfArgSet(ss.Net.AsAxon(), &ss.Args, ctrString, ss.Stats.String("RunName"))
	})

	////////////////////////////////////////////
	// GUI
	if ss.Args.Bool("nogui") == false {
		axon.LooperUpdtNetView(man, &ss.ViewUpdt)
		axon.LooperUpdtPlots(man, &ss.GUI)
		// man.GetLoop(etime.Test, etime.Trial).Main.Add("Log:Test:Trial", func() {
		// 	ss.GUI.NetDataRecord()
		// })
	}

	if Debug {
		fmt.Println(man.DocString())
	}
	ss.Loops = man
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	net := ss.Net
	ev := ss.Envs[ss.Time.Mode]
	net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway
	lays := net.LayersByClass("Input", "Target")
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		pats := ev.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	ss.Envs.ByMode(etime.Train).Init(0)
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Time.Reset()
	ss.Time.Mode = etime.Train.String()
	ss.Net.InitWts()
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Loops.Mode = etime.Test
	ss.Loops.Run()
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("TrlUnitErr", 0.0)
	ss.Stats.SetFloat("TrlCorSim", 0.0)
	ss.Logs.InitErrStats() // inits TrlErr, FirstZero, LastZero, NZero
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdt.Text
func (ss *Sim) StatCounters() {
	var mode etime.Modes
	mode.FromString(ss.Time.Mode)
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	ss.Stats.SetInt("Cycle", ss.Time.Cycle)
	ev := ss.Envs[ss.Time.Mode]
	ss.Stats.SetString("TrialName", ev.(*LEDEnv).String())
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle", "TrlUnitErr", "TrlErr", "TrlCorSim"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()

	ss.Stats.SetFloat("TrlCorSim", float64(out.CorSim.Cor))
	ss.Stats.SetFloat("TrlUnitErr", out.PctUnitErr())

	if ss.Stats.Float("TrlUnitErr") > 0 {
		ss.Stats.SetFloat("TrlErr", 1)
	} else {
		ss.Stats.SetFloat("TrlErr", 0)
	}
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging
func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems([]etime.Times{etime.Run, etime.Epoch, etime.Trial, etime.Cycle}, []string{"TrialName", "RunName"})

	ss.Logs.AddStatAggItem("CorSim", "TrlCorSim", elog.DTrue, etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("UnitErr", "TrlUnitErr", elog.DFalse, etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	axon.LogAddDiagnosticItems(&ss.Logs, ss.Net.AsAxon(), etime.Epoch, etime.Trial)
	axon.LogAddPCAItems(&ss.Logs, ss.Net.AsAxon(), etime.Run, etime.Epoch, etime.Trial)

	axon.LogAddLayerGeActAvgItems(&ss.Logs, ss.Net.AsAxon(), etime.Test, etime.Cycle)
	axon.LogAddLayerActTensorItems(&ss.Logs, ss.Net.AsAxon(), etime.Test, etime.Trial)

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net.AsAxon())
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	if mode.String() != "Analyze" {
		ss.Time.Mode = mode.String() // Also set specifically in a Loop callback.
	}
	ss.StatCounters()
	dt := ss.Logs.Table(mode, time)
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		row = ss.Stats.Int("Cycle")
	case time == etime.Trial:
		row = ss.Stats.Int("Trial")
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

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

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ViewUpdt.Config(nv, etime.AlphaCycle, etime.AlphaCycle)
	ss.GUI.ViewUpdt = &ss.ViewUpdt
	ss.ConfigNetView(nv)

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddLooperCtrl(ss.Loops, []etime.Modes{etime.Train, etime.Test})

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
			ss.RndSeeds.NewSeeds()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "README",
		Icon:    "file-markdown",
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/ra25/README.md")
		},
	})
	ss.GUI.FinalizeGUI(false)
	return ss.GUI.Win
}

func (ss *Sim) ConfigArgs() {
	ss.Args.Init()
	ss.Args.AddStd()
	ss.Args.AddInt("nzero", 2, "number of zero error epochs in a row to count as full training")
	ss.Args.AddInt("iticycles", 0, "number of cycles to run between trials (inter-trial-interval)")
	ss.Args.SetInt("epochs", 100)
	ss.Args.SetInt("runs", 5)
	ss.Args.Parse() // always parse
}

func (ss *Sim) CmdArgs() {
	ss.Args.ProcStd(&ss.Logs, &ss.Params, ss.Net.Name())
	ss.Args.SetBool("nogui", true)                                       // by definition if here
	ss.Stats.SetString("RunName", ss.Params.RunName(ss.Args.Int("run"))) // used for naming logs, stats, etc

	netdata := ss.Args.Bool("netdata")
	if netdata {
		fmt.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	runs := ss.Args.Int("runs")
	run := ss.Args.Int("run")
	fmt.Printf("Running %d Runs starting at %d\n", runs, run)
	rc := &ss.Loops.GetLoop(etime.Train, etime.Run).Counter
	rc.Set(run)
	rc.Max = run + runs
	ss.NewRun()
	ss.Loops.Run()

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}
}
