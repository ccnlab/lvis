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
	"strings"
	"time"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/actrf"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/metric"
	"github.com/emer/etable/norm"
	"github.com/emer/etable/pca"
	"github.com/emer/etable/split"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
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
					"Layer.Act.Dt.IntTau":                "40", // 40 > 30 > 20 > 10
					"Layer.Inhib.Layer.Gi":               "1.0",
					"Layer.Inhib.Layer.FBTau":            "1.4", // 1.4 def
					"Layer.Inhib.Pool.FBTau":             "1.4",
					"Layer.Act.Decay.Act":                "0.0",  // 0.2 with glong .6 best in lvis, slows learning here
					"Layer.Act.Decay.Glong":              "1",    // 0, 1 best here
					"Layer.Act.Decay.KNa":                "0.0",  // 0 > 0.5 interesting..
					"Layer.Act.Gbar.L":                   "0.2",  // .2 > .1 @176
					"Layer.Act.Gbar.E":                   "1.0",  // 1.2 maybe better % cor but not cosdiff
					"Layer.Act.Gbar.K":                   "1.0",  // 1.0 > 0.5 > 2.0 -- interactive test
					"Layer.Act.NMDA.Gbar":                "0.03", // 0.03 > .04 @176 > .02 -- massive effects for .02
					"Layer.Act.NMDA.Tau":                 "100",  // 100 def
					"Layer.Act.GABAB.Gbar":               "0.2",  // .1 == .2 pretty much
					"Layer.Act.GABAB.Gbase":              "0.2",  // .1 == .2
					"Layer.Act.GABAB.DecayTau":           "50",   // 50 def
					"Layer.Act.GABAB.RiseTau":            "45",   // 45 def
					"Layer.Act.Spike.Exp":                "true", // true > false @176
					"Layer.Act.Spike.Tr":                 "3",    // 3 def -- 2 tiny bit worse, needs out ge .5
					"Layer.Learn.ActAvg.SpikeG":          "8",    // 8 for sure..
					"Layer.Learn.ActAvg.SSTau":           "40",   // 40 > 35 def > 30
					"Layer.Learn.ActAvg.STau":            "10",   // 10 >= 8 def (10 better early) > 6
					"Layer.Learn.ActAvg.MTau":            "40",   // for 50 cyc qtr: SS = 40, 50, 45 faster then die
					"Layer.Learn.ActAvg.MinLrn":          "0.02", // lvis: sig improves "top5" hogging in pca strength -- slower start here but same asymp.
					"Layer.Act.Dt.GeTau":                 "5",    // 5 = 4 (bit slower) > 6 > 7 @176
					"Layer.Act.KNa.On":                   "true", // true > false @176
					"Layer.Act.KNa.Fast.Max":             "0.1",  // 0.1 > 0.1 -- 122 best
					"Layer.Act.KNa.Med.Max":              "0.2",  // 0.2 > 0.1 def
					"Layer.Act.KNa.Slow.Max":             "0.2",  // 0.2 > 0.1 > 0.5
					"Layer.Act.Noise.Dist":               "Gaussian",
					"Layer.Act.Noise.Mean":               "0.0",
					"Layer.Act.Noise.Var":                "0.01",    // 0.01 > 0.005 > 0.02
					"Layer.Act.Noise.Type":               "NoNoise", // no diff -- maybe tiny bit better
					"Layer.Act.Clamp.Rate":               "180",     // 180 == 200 > 150 > 120 > 100 -- major effect on 100, 120
					"Layer.Act.Clamp.ErrThr":             "0.5",     // 0.5 best
					"Layer.Act.Dt.LongAvgTau":            "20",      // 20 > 50 > 100
					"Layer.Act.GTarg.GeMax":              "1",       // 1 > .8 here
					"Layer.Learn.TrgAvgAct.ErrLrate":     "0.02",    // .02 > .01 > .005 > .05
					"Layer.Learn.TrgAvgAct.SynScaleRate": "0.005",   // .002 >= .005 > .01
					"Layer.Learn.TrgAvgAct.TrgRange.Min": "0.2",     // .2 > .5 > .1
					"Layer.Learn.TrgAvgAct.TrgRange.Max": "2.0",     // 2 > 2.5 > 1.8
					"Layer.Learn.RLrate.On":              "true",
					"Layer.Learn.RLrate.ActThr":          "0.1",
					"Layer.Learn.RLrate.ActDifThr":       "0.02", // .02 > .05
					"Layer.Learn.RLrate.Min":             "0.001",
				}},
			{Sel: "#V1", Desc: "pool inhib (not used), initial activity",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1.0",  //
					"Layer.Inhib.Pool.Gi":     "1.0",  //
					"Layer.Inhib.Pool.On":     "true", // key for soft clamped
					"Layer.Inhib.ActAvg.Init": "0.06", // .1 for hard clamp, .06 for Ge clamp
					"Layer.Inhib.ActAvg.Targ": "0.06",
					"Layer.Act.Clamp.Type":    "GeClamp", // GeClamp better in every other case..
					"Layer.Act.Clamp.Ge":      "0.6",     // .6 generally = .5
				}},
			{Sel: "#V4", Desc: "pool inhib, sparse activity",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":       "1.0",  // 1.0 == 0.9 == 0.8 > 0.7 > 1.1 (vry bad)
					"Layer.Inhib.Pool.Gi":        "1.0",  // 1.0 == 0.9 > 0.8 > 1.1 (vry bad)
					"Layer.Inhib.Pool.On":        "true", // needs pool-level
					"Layer.Inhib.ActAvg.Init":    "0.05",
					"Layer.Inhib.ActAvg.Targ":    "0.05",
					"Layer.Inhib.ActAvg.AdaptGi": "false", // no advantage, no cost at .05
				}},
			{Sel: "#IT", Desc: "initial activity",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":       "1.1",  // 1.1 > 1.0, 1.2
					"Layer.Inhib.ActAvg.Init":    "0.05", // .05 > .04 with adapt
					"Layer.Inhib.ActAvg.Targ":    "0.05", // .05 > .04 with adapt
					"Layer.Inhib.ActAvg.AdaptGi": "true", // this is only adapt that is key
					"Layer.Act.GABAB.Gbar":       "0.2",  // .2 > lower (small dif)
				}},
			{Sel: "#Output", Desc: "high inhib for one-hot output",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":       "1.5", // 1.5 = 1.6 > 1.4 adapt
					"Layer.Inhib.ActAvg.Init":    "0.05",
					"Layer.Inhib.ActAvg.Targ":    "0.05",  // this has to be exact for adapt
					"Layer.Inhib.ActAvg.AdaptGi": "false", // true >= false (small, only cosdif)
					"Layer.Inhib.ActAvg.LoTol":   "0.8",   // essential to keep low
					"Layer.Act.Clamp.Rate":       "180",   // 180 best here too
					"Layer.Act.Clamp.Type":       "GeClamp",
					"Layer.Act.Clamp.Ge":         "0.6",   // .6 generally = .5
					"Layer.Act.Clamp.Burst":      "false", // effective for boosting errors but no overall effect
					"Layer.Act.Clamp.BurstThr":   "0.5",   //
					"Layer.Act.Clamp.BurstGe":    "2",     // 2, 20cyc with tr 2 or 3, ge .6 all about same; 2 = 1.5 = 1 more or less -- tiny bit of extra err diff progressively
					"Layer.Act.Clamp.BurstCyc":   "20",    // 20 > 15 > 10 -- maybe refractory?  25, 30 = 20
					"Layer.Act.Spike.Tr":         "3",     // 2 >= 3 > 1 > 0
					"Layer.Act.GTarg.GeMax":      "0.8",   // 0.8 >= 1 -- tiny diff
					"Layer.Act.GABAB.Gbar":       "0.005", // .005 > .01 > .02 > .05 > .1 > .2
					"Layer.Act.NMDA.Gbar":        "0.03",  // .03 > .02 > .01 > .1
				}},
			{Sel: "Prjn", Desc: "yes extra learning factors",
				Params: params.Params{
					"Prjn.PrjnScale.ScaleLrate": "0.02",   // .1 > higher
					"Prjn.PrjnScale.AvgTau":     "500",    // slower default
					"Prjn.Learn.Lrate.Base":     "0.2",    // lower progressively worse.. gain 1, lr .35 or .4 pretty close to 6/.04
					"Prjn.Learn.XCal.SubMean":   "1",      // 1 > .9
					"Prjn.Learn.XCal.DWtThr":    "0.0001", // 0.0001 > 0.001
					"Prjn.SWt.Adapt.Lrate":      "0.005",  // 0.005 > others maybe?  0.02 > 0.05 > .1
					"Prjn.SWt.Adapt.SigGain":    "6",
					"Prjn.SWt.Init.SPct":        "1",   // 1 >= lower
					"Prjn.SWt.Init.Mean":        "0.5", // .4 better on pca, .5 starts faster
					"Prjn.SWt.Limit.Min":        "0.2", // .3-.7 better constraint, but not clear better than no SWt
					"Prjn.SWt.Limit.Max":        "0.8",
					"Prjn.Com.PFail":            "0.0",
					"Prjn.Com.PFailWtMax":       "0.0", // 0.8 default
					// "Prjn.WtInit.Sym":        "false", // slows first couple of epochs but then no diff
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
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net            *axon.Network                 `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	TrnTrlLog      *etable.Table                 `view:"no-inline" desc:"training trial-level log data"`
	TrnTrlRepLog   *etable.Table                 `view:"no-inline" desc:"training trial-level reps log data"`
	TrnEpcLog      *etable.Table                 `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog      *etable.Table                 `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog      *etable.Table                 `view:"no-inline" desc:"testing trial-level log data"`
	TrnErrStats    *etable.Table                 `view:"no-inline" desc:"training error stats"`
	ActRFs         actrf.RFs                     `view:"no-inline" desc:"activation-based receptive fields"`
	RunLog         *etable.Table                 `view:"no-inline" desc:"summary log of each run"`
	RunStats       *etable.Table                 `view:"no-inline" desc:"aggregate stats on all runs"`
	ErrLrMod       axon.LrateMod                 `view:"inline" desc:"learning rate modulation as function of error"`
	Params         params.Sets                   `view:"no-inline" desc:"full collection of param sets"`
	ParamSet       string                        `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don't put spaces in ParamSet names!)"`
	Tag            string                        `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	V1V4Prjn       *prjn.PoolTile                `view:"projection from V1 to V4 which is tiled 4x4 skip 2 with topo scale values"`
	StartRun       int                           `desc:"starting run number -- typically 0 but can be set in command args for parallel runs on a cluster"`
	MaxRuns        int                           `desc:"maximum number of model runs to perform"`
	MaxEpcs        int                           `desc:"maximum number of epochs to run per model run"`
	MaxTrls        int                           `desc:"maximum number of training trials per epoch"`
	RepsInterval   int                           `desc:"how often to analyze the representations"`
	NZeroStop      int                           `desc:"if a positive number, training will stop after this many epochs with zero UnitErr"`
	MiniBatches    int                           `desc:"number of trials to aggregate into DWt before applying"`
	TrainEnv       LEDEnv                        `desc:"Training environment -- LED training"`
	PNovel         float32                       `desc:"proportion of novel training items to use -- set this to 0.5 after initial training"`
	NovelTrainEnv  LEDEnv                        `desc:"Novel items training environment -- LED training"`
	TestEnv        LEDEnv                        `desc:"Testing environment -- LED testing"`
	Time           axon.Time                     `desc:"axon timing parameters and state"`
	ViewOn         bool                          `desc:"whether to update the network view while running"`
	TrainUpdt      axon.TimeScales               `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt       axon.TimeScales               `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	LayStatNms     []string                      `desc:"names of layers to collect more detailed stats on (avg act, etc)"`
	ActRFNms       []string                      `desc:"names of layers to compute activation rfields on"`
	SpikeRasters   map[string]*etensor.Float32   `desc:"spike raster data for different layers"`
	SpikeRastGrids map[string]*etview.TensorGrid `desc:"spike raster plots for different layers"`

	// statistics: note use float64 as that is best for etable.Table
	TrlErr        float64 `inactive:"+" desc:"1 if trial was error, 0 if correct -- based on max out unit"`
	TrlErr2       float64 `inactive:"+" desc:"1 if trial was error, 0 if correct -- correct if in top 2"`
	TrlTrgAct     float64 `inactive:"+" desc:"activity of target output unit on this trial"`
	TrlUnitErr    float64 `inactive:"+" desc:"current trial's sum squared error"`
	TrlCosDiff    float64 `inactive:"+" desc:"current trial's cosine difference"`
	TrlOut        string  `inactive:"+" desc:"output response for current trial"`
	EpcUnitErr    float64 `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcPctErr     float64 `inactive:"+" desc:"last epoch's average TrlErr"`
	EpcPctCor     float64 `inactive:"+" desc:"1 - last epoch's average TrlErr"`
	EpcPctErr2    float64 `inactive:"+" desc:"last epoch's average TrlErr2"`
	EpcCosDiff    float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	EpcErrTrgAct  float64 `inactive:"+" desc:"avg activity of target output unit on err trials"`
	EpcCorTrgAct  float64 `inactive:"+" desc:"avg activity of target output unit on correct trials"`
	EpcPerTrlMSec float64 `inactive:"+" desc:"how long did the epoch take per trial in wall-clock milliseconds"`
	FirstZero     int     `inactive:"+" desc:"epoch at when UnitErr first went to zero"`
	NZero         int     `inactive:"+" desc:"number of epochs in a row with zero UnitErr"`
	MiniBatchCtr  int     `inactive:"+" desc:"counter for mini-batch learning"`
	PCA           pca.PCA `view:"-" desc:"pca obj"`

	// internal state - view:"-"
	Win          *gi.Window                    `view:"-" desc:"main GUI window"`
	NetView      *netview.NetView              `view:"-" desc:"the network viewer"`
	ToolBar      *gi.ToolBar                   `view:"-" desc:"the master toolbar"`
	CurImgGrid   *etview.TensorGrid            `view:"-" desc:"the current image grid view"`
	ActRFGrids   map[string]*etview.TensorGrid `view:"-" desc:"the act rf grid views"`
	TrnTrlPlot   *eplot.Plot2D                 `view:"-" desc:"the training trial plot"`
	TrnEpcPlot   *eplot.Plot2D                 `view:"-" desc:"the training epoch plot"`
	TstEpcPlot   *eplot.Plot2D                 `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot   *eplot.Plot2D                 `view:"-" desc:"the test-trial plot"`
	RunPlot      *eplot.Plot2D                 `view:"-" desc:"the run plot"`
	TrnTrlFile   *os.File                      `view:"-" desc:"log file"`
	TrnEpcFile   *os.File                      `view:"-" desc:"log file"`
	RunFile      *os.File                      `view:"-" desc:"log file"`
	ValsTsrs     map[string]*etensor.Float32   `view:"-" desc:"for holding layer values"`
	SaveWts      bool                          `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui        bool                          `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams bool                          `view:"-" desc:"if true, print message for all params that are set"`
	IsRunning    bool                          `view:"-" desc:"true if sim is running"`
	StopNow      bool                          `view:"-" desc:"flag to stop running"`
	NeedsNewRun  bool                          `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeeds     []int64                       `view:"-" desc:"a list of random seeds to use for each run"`
	LastEpcTime  time.Time                     `view:"-" desc:"timer for last epoch"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	ss.TrnTrlLog = &etable.Table{}
	ss.TrnTrlRepLog = &etable.Table{}
	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.Params = ParamSets
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
	ss.LayStatNms = []string{"V4", "IT", "Output"}
	ss.ActRFNms = []string{"V4:Image", "V4:Output", "IT:Image", "IT:Output"}
	ss.PNovel = 0
	ss.MiniBatches = 1 // 1 > 16
	ss.RepsInterval = 10

	ss.Time.Defaults()
	ss.ErrLrMod.Defaults()
	ss.ErrLrMod.Base = 0.5 // .5 > .2 > .1 > .05 (not good at end)
	ss.ErrLrMod.Range.Set(0.2, 0.8)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTrnTrlLog(ss.TrnTrlLog)
	ss.ConfigTrnTrlRepLog(ss.TrnTrlRepLog)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigSpikeRasts()
	ss.ConfigRunLog(ss.RunLog)
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
	ss.TestEnv.MaxLED = 19     // all by default
	ss.TestEnv.Trial.Max = 500 // 1000 is too long!
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
	ss.SetParams("Network", false) // only set Network params
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
	ss.SetParams("", false)   // all sheets
	ss.Net.SlowInterval = 100 // / ss.MiniBatches
	ss.NewRun()
	ss.UpdateView(true)
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

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters(train bool) string {
	if train {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.String())
	} else {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.String())
	}
}

func (ss *Sim) UpdateView(train bool) {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(train))
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

func (ss *Sim) UpdateViewTime(train bool, viewUpdt axon.TimeScales) {
	switch viewUpdt {
	case axon.Cycle:
		ss.UpdateView(train)
	case axon.FastSpike:
		if ss.Time.Cycle%10 == 0 {
			ss.UpdateView(train)
		}
	case axon.GammaCycle:
		if ss.Time.Cycle%25 == 0 {
			ss.UpdateView(train)
		}
	case axon.AlphaCycle:
		if ss.Time.Cycle%100 == 0 {
			ss.UpdateView(train)
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
			ss.Net.WtFmDWt()
		}
	}

	minusCyc := 150
	plusCyc := 50

	ss.Net.NewState()
	ss.Time.NewState()
	for cyc := 0; cyc < minusCyc; cyc++ { // do the minus phase
		ss.Net.Cycle(&ss.Time)
		// if !train {
		// 	ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
		// }
		if !ss.NoGui {
			ss.RecSpikes(ss.Time.Cycle)
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
			ss.Net.ClearTargExt()
			ss.Time.PlusPhase = false
		}

		if cyc == minusCyc-1 { // do before view update
			ss.Net.MinusPhase(&ss.Time)
		}
		if ss.ViewOn {
			ss.UpdateViewTime(train, viewUpdt)
		}
	}
	ss.Time.NewPhase()
	if viewUpdt == axon.Phase {
		ss.UpdateView(train)
	}
	for cyc := 0; cyc < plusCyc; cyc++ { // do the plus phase
		ss.Net.Cycle(&ss.Time)
		// if !train {
		// 	ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
		// }
		if !ss.NoGui {
			ss.RecSpikes(ss.Time.Cycle)
		}
		ss.Time.CycleInc()

		if cyc == plusCyc-1 { // do before view update
			ss.Net.PlusPhase(&ss.Time)
		}
		if ss.ViewOn {
			ss.UpdateViewTime(train, viewUpdt)
		}
	}
	if viewUpdt == axon.Phase || viewUpdt == axon.AlphaCycle || viewUpdt == axon.ThetaCycle {
		ss.UpdateView(train)
	}

	ss.TrialStats(train)

	if train {
		// ss.ErrLrMod.LrateMod(ss.Net, float32(1-ss.TrlCosDiff))
		ss.Net.DWt()
	}
	if ss.ViewOn && viewUpdt == axon.AlphaCycle {
		ss.UpdateView(train)
	}

	// if ss.TstCycPlot != nil && !train {
	// 	ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
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
		ss.LogTrnEpc(ss.TrnEpcLog)
		ss.LrateSched(epc)
		if ss.ViewOn && ss.TrainUpdt > axon.AlphaCycle {
			ss.UpdateView(true)
		}
		if epc >= ss.MaxEpcs || (ss.NZeroStop > 0 && ss.NZero >= ss.NZeroStop) {
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
	ss.LogTrnTrl(ss.TrnTrlLog)
	if ss.RepsInterval > 0 && epc%ss.RepsInterval == 0 {
		ss.LogTrnRepTrl(ss.TrnTrlRepLog)
	}
	if ss.CurImgGrid != nil {
		ss.CurImgGrid.UpdateSig()
	}
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.LogRun(ss.RunLog)
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
	ss.TrnTrlLog.SetNumRows(0)
	ss.TrnEpcLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
	ss.NeedsNewRun = false
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	ss.FirstZero = -1
	ss.NZero = 0
	// clear rest just to make Sim look initialized
	ss.TrlErr = 0
	ss.TrlErr2 = 0
	ss.TrlOut = ""
	ss.TrlTrgAct = 0
	ss.TrlUnitErr = 0
	ss.TrlCosDiff = 0
	ss.EpcUnitErr = 0
	ss.EpcPctErr = 0
	ss.EpcCosDiff = 0
	ss.EpcErrTrgAct = 0
	ss.EpcCorTrgAct = 0
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) {
	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()
	ss.TrlCosDiff = float64(out.CosDiff.Cos)
	ss.TrlUnitErr = out.PctUnitErr()

	ovt := ss.ValsTsr("Output")
	out.UnitValsTensor(ovt, "ActM")
	rsp := 0
	rsp, ss.TrlErr, ss.TrlErr2 = ss.TrainEnv.OutErr(ovt)
	ss.TrlOut = fmt.Sprintf("%d", rsp)

	ss.TrlTrgAct = float64(out.Pools[0].ActP.Avg)

	if !accum { // testing
		ss.UpdtActRFs()
	}
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
	ss.IsRunning = false
	if ss.Win != nil {
		vp := ss.Win.WinViewport2D()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.SetNeedsFullRender()
	}
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

// LrateSched implements the learning rate schedule
func (ss *Sim) LrateSched(epc int) {
	switch epc {
	case 40:
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
	ss.SetParamsSet("NovelLearn", "Network", true)
	ss.TrainEnv.Epoch.Cur = 40
	ss.LrateSched(40)
	ss.PNovel = 0.5
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
			ss.UpdateView(false)
		}
		ss.LogTstEpc(ss.TstEpcLog)
		if returnOnChg {
			return
		}
	}

	// note: type must be in place before apply inputs
	ss.Net.LayerByName("Output").SetType(emer.Compare)
	ss.ApplyInputs(&ss.TestEnv)
	ss.ThetaCyc(false) // !train
	ss.LogTstTrl(ss.TstTrlLog)
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
	ss.ActRFs.Reset()
	for {
		ss.TestTrial(true) // return on chg, don't present
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}
	ss.ActRFs.Avg()
	ss.ActRFs.Norm()
	ss.ViewActRFs()
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

// UpdtActRFs updates activation rf's -- only called during testing
func (ss *Sim) UpdtActRFs() {
	oly := ss.Net.LayerByName("Output")
	ovt := ss.ValsTsr("Output")
	oly.UnitValsTensor(ovt, "ActM")
	if _, ok := ss.ValsTsrs["Image"]; !ok {
		ss.ValsTsrs["Image"] = &ss.TestEnv.Vis.ImgTsr
	}
	naf := len(ss.ActRFNms)
	if len(ss.ActRFs.RFs) != naf {
		for _, anm := range ss.ActRFNms {
			sp := strings.Split(anm, ":")
			lnm := sp[0]
			ly := ss.Net.LayerByName(lnm)
			if ly == nil {
				continue
			}
			lvt := ss.ValsTsr(lnm)
			ly.UnitValsTensor(lvt, "ActM")
			tnm := sp[1]
			tvt := ss.ValsTsr(tnm)
			ss.ActRFs.AddRF(anm, lvt, tvt)
			// af.NormRF.SetMetaData("min", "0")
		}
	}
	for _, anm := range ss.ActRFNms {
		sp := strings.Split(anm, ":")
		lnm := sp[0]
		ly := ss.Net.LayerByName(lnm)
		if ly == nil {
			continue
		}
		lvt := ss.ValsTsr(lnm)
		ly.UnitValsTensor(lvt, "ActM")
		tnm := sp[1]
		tvt := ss.ValsTsr(tnm)
		ss.ActRFs.Add(anm, lvt, tvt, 0.01) // thr prevent weird artifacts
	}
}

// ViewActRFs displays act rfs
func (ss *Sim) ViewActRFs() {
	if ss.ActRFGrids == nil {
		return
	}
	for _, nm := range ss.ActRFNms {
		tg := ss.ActRFGrids[nm]
		if tg.Tensor == nil {
			rf := ss.ActRFs.RFByName(nm)
			tg.SetTensor(&rf.NormRF)
		} else {
			tg.UpdateSig()
		}
	}
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

// ParamsName returns name of current set of parameters
func (ss *Sim) ParamsName() string {
	if ss.ParamSet == "" {
		return "Base"
	}
	return ss.ParamSet
}

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) error {
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		ss.Params.ValidateSheets([]string{"Network", "Sim"})
	}
	err := ss.SetParamsSet("Base", sheet, setMsg)
	if ss.ParamSet != "" && ss.ParamSet != "Base" {
		sps := strings.Fields(ss.ParamSet)
		for _, ps := range sps {
			err = ss.SetParamsSet(ps, sheet, setMsg)
		}
	}
	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParamsSet(setNm string, sheet string, setMsg bool) error {
	pset, err := ss.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			ss.Net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// ValsTsr gets value tensor of given name, creating if not yet made
func (ss *Sim) ValsTsr(name string) *etensor.Float32 {
	if ss.ValsTsrs == nil {
		ss.ValsTsrs = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.ValsTsrs[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.ValsTsrs[name] = tsr
	}
	return tsr
}

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func (ss *Sim) RunName() string {
	rn := ""
	if ss.Tag != "" {
		rn += ss.Tag + "_"
	}
	rn += ss.ParamsName()
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

//////////////////////////////////////////////
//  TrnTrlLog

// LogTrnTrl adds data from current trial to the TrnTrlLog table.
func (ss *Sim) LogTrnTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Cur
	trl := ss.TrainEnv.Trial.Cur
	row := dt.Rows

	if row > 1 { // reset at new epoch
		lstepc := int(dt.CellFloat("Epoch", row-1))
		if lstepc != epc {
			dt.SetNumRows(0)
			row = 0
		}
	}
	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellFloat("Idx", row, float64(row))
	dt.SetCellString("Cat", row, fmt.Sprintf("%d", ss.TrainEnv.CurLED))
	dt.SetCellString("TrialName", row, ss.TrainEnv.String())

	dt.SetCellFloat("Err", row, ss.TrlErr)
	dt.SetCellFloat("Err2", row, ss.TrlErr2)
	dt.SetCellFloat("TrgAct", row, ss.TrlTrgAct)
	dt.SetCellFloat("UnitErr", row, ss.TrlUnitErr)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		dt.SetCellFloat(lnm+"_MaxGeM", row, float64(ly.Pools[0].GeM.Max))

		lvt := ss.ValsTsr(lnm)
		ly.UnitValsTensor(lvt, "ActDif")
		mad := 0.0
		for i := range lvt.Values {
			mad += math.Abs(float64(lvt.Values[i]))
		}
		mad /= float64(len(lvt.Values))

		dt.SetCellFloat(lnm+"_CosDiff", row, float64(1-ly.CosDiff.Cos))
		dt.SetCellFloat(lnm+"_ActDif", row, mad)
	}

	// if ss.TrnTrlFile != nil && (!ss.UseMPI || ss.SaveProcLog) { // otherwise written at end of epoch, integrated
	// 	if ss.TrainEnv.Run.Cur == ss.StartRun && epc == 0 && row == 0 {
	// 		dt.WriteCSVHeaders(ss.TrnTrlFile, etable.Tab)
	// 	}
	// 	dt.WriteCSVRow(ss.TrnTrlFile, row, etable.Tab)
	// }

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTrnTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnTrlLog")
	dt.SetMetaData("desc", "Record of training per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"Idx", etensor.INT64, nil, nil},
		{"Cat", etensor.STRING, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Err", etensor.FLOAT64, nil, nil},
		{"Err2", etensor.FLOAT64, nil, nil},
		{"TrgAct", etensor.FLOAT64, nil, nil},
		{"UnitErr", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + "_MaxGeM", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_CosDiff", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_ActDif", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Object Recognition Train Trial Plot"
	plt.Params.XAxisCol = "Idx"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Idx", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Cat", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

	plt.SetColParams("Err", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Err2", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("TrgAct", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("UnitErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+"_MaxGeM", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.5)
		plt.SetColParams(lnm+"_CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams(lnm+"_ActDif", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	}
	return plt
}

//////////////////////////////////////////////
//  TrnTrlRepLog

// CenterPoolsIdxs returns the indexes for 2 center pools:
// nu = number of units per pool, si = starting index
func (ss *Sim) CenterPoolsIdxs(ly *axon.Layer) (nu, py, px, si int) {
	py = (ly.Shp.Dim(0) - 1) / 2
	px = (ly.Shp.Dim(1) - 1) / 2
	nu = ly.Shp.Dim(2) * ly.Shp.Dim(3)
	si = (py*ly.Shp.Dim(1) + px) * nu
	return
}

// CopyCenterPools copy 2 center pools of ActM to tensor
func (ss *Sim) CopyCenterPools(ly *axon.Layer, vl *etensor.Float32) {
	nu, _, _, si := ss.CenterPoolsIdxs(ly)
	vl.SetShape([]int{2 * nu}, nil, nil)
	for ni := 0; ni < 2*nu; ni++ {
		vl.Values[ni] = ly.Neurons[si+ni].ActM
	}
}

// LogTrnRepTrl adds data from current trial to the TrnTrlRepLog table.
func (ss *Sim) LogTrnRepTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Cur
	trl := ss.TrainEnv.Trial.Cur
	row := dt.Rows

	if row > 1 { // reset at new epoch
		lstepc := int(dt.CellFloat("Epoch", row-1))
		if lstepc != epc {
			dt.SetNumRows(0)
			row = 0
		}
	}
	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellFloat("Idx", row, float64(row))
	dt.SetCellString("Cat", row, fmt.Sprintf("%d", ss.TrainEnv.CurLED))
	dt.SetCellString("TrialName", row, ss.TrainEnv.String())

	dt.SetCellFloat("Err", row, ss.TrlErr)
	dt.SetCellFloat("Err2", row, ss.TrlErr2)
	dt.SetCellFloat("TrgAct", row, ss.TrlTrgAct)
	dt.SetCellFloat("UnitErr", row, ss.TrlUnitErr)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		lvt := ss.ValsTsr(lnm)
		if ly.Is4D() && ly.Nm != "Output" {
			ss.CopyCenterPools(ly, lvt)
			dt.SetCellTensor(lnm, row, lvt)
		} else {
			ly.UnitValsTensor(lvt, "ActM")
			dt.SetCellTensor(lnm, row, lvt)
		}
	}

	// if ss.TrnTrlFile != nil && (!ss.UseMPI || ss.SaveProcLog) { // otherwise written at end of epoch, integrated
	// 	if ss.TrainEnv.Run.Cur == ss.StartRun && epc == 0 && row == 0 {
	// 		dt.WriteCSVHeaders(ss.TrnTrlFile, etable.Tab)
	// 	}
	// 	dt.WriteCSVRow(ss.TrnTrlFile, row, etable.Tab)
	// }

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTrnTrlRepLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnTrlRepLog")
	dt.SetMetaData("desc", "Record of training per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"Idx", etensor.INT64, nil, nil},
		{"Cat", etensor.STRING, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Err", etensor.FLOAT64, nil, nil},
		{"Err2", etensor.FLOAT64, nil, nil},
		{"TrgAct", etensor.FLOAT64, nil, nil},
		{"UnitErr", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		if ly.Is4D() && ly.Nm != "Output" {
			nu, _, _, _ := ss.CenterPoolsIdxs(ly)
			sch = append(sch, etable.Column{lnm, etensor.FLOAT64, []int{2 * nu}, nil})
		} else {
			sch = append(sch, etable.Column{lnm, etensor.FLOAT64, ly.Shp.Shp, nil})
		}
	}
	dt.SetFromSchema(sch, 0)
}

//////////////////////////////////////////////
//  TrnEpcLog

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value

	trl := ss.TrnTrlLog
	nt := float64(trl.Rows)
	tix := etable.NewIdxView(trl)

	ss.EpcUnitErr = agg.Mean(tix, "UnitErr")[0]
	ss.EpcPctErr = agg.Mean(tix, "Err")[0]
	ss.EpcPctCor = 1 - ss.EpcPctErr
	ss.EpcPctErr2 = agg.Mean(tix, "Err2")[0]
	ss.EpcCosDiff = agg.Mean(tix, "CosDiff")[0]

	spl := split.GroupBy(tix, []string{"Err"})
	split.Desc(spl, "TrgAct")
	for _, lnm := range ss.LayStatNms {
		split.Desc(spl, lnm+"_CosDiff")
		split.Desc(spl, lnm+"_ActDif")
	}
	ss.TrnErrStats = spl.AggsToTable(etable.AddAggName)

	if ss.EpcPctErr > 0 && ss.EpcPctErr < 1 {
		ss.EpcCorTrgAct = ss.TrnErrStats.CellFloat("TrgAct:Mean", 0)
		ss.EpcErrTrgAct = ss.TrnErrStats.CellFloat("TrgAct:Mean", 1)
	}

	if ss.FirstZero < 0 && ss.EpcPctErr == 0 {
		ss.FirstZero = epc
	}
	if ss.EpcPctErr == 0 {
		ss.NZero++
	} else {
		ss.NZero = 0
	}

	if ss.LastEpcTime.IsZero() {
		ss.EpcPerTrlMSec = 0
	} else {
		iv := time.Now().Sub(ss.LastEpcTime)
		ss.EpcPerTrlMSec = float64(iv) / (nt * float64(time.Millisecond))
	}
	ss.LastEpcTime = time.Now()

	if ss.RepsInterval > 0 && epc%ss.RepsInterval == 0 {
		reps := etable.NewIdxView(ss.TrnTrlRepLog)
		reps.SortColName("Cat", true)
		for _, lnm := range ss.LayStatNms {
			ss.PCA.TableCol(reps, lnm, metric.Covariance64)
			var nstr float64
			ln := len(ss.PCA.Values)
			for i, v := range ss.PCA.Values {
				// fmt.Printf("%s\t\t %d  %g\n", lnm, i, v)
				if v >= 0.01 { // .01 gives appropriate number for output
					nstr = float64(ln - i)
					break
				}
			}
			mn := norm.Mean64(ss.PCA.Values)
			dt.SetCellFloat(lnm+"_PCA_NStrong", row, nstr)
			dt.SetCellFloat(lnm+"_PCA_Mean", row, mn)
		}
	} else {
		if row > 0 {
			for _, lnm := range ss.LayStatNms {
				dt.SetCellFloat(lnm+"_PCA_NStrong", row, dt.CellFloat(lnm+"_PCA_NStrong", row-1))
				dt.SetCellFloat(lnm+"_PCA_Mean", row, dt.CellFloat(lnm+"_PCA_Mean", row-1))
			}
		}
	}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("UnitErr", row, ss.EpcUnitErr)
	dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
	dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
	dt.SetCellFloat("PctErr2", row, ss.EpcPctErr2)
	dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)
	dt.SetCellFloat("ErrTrgAct", row, ss.EpcErrTrgAct)
	dt.SetCellFloat("CorTrgAct", row, ss.EpcCorTrgAct)
	dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		ffpj := ly.RecvPrjn(0).(*axon.Prjn)
		dt.SetCellFloat(ly.Nm+"_FF_AvgMaxG", row, float64(ffpj.GScale.AvgMax))
		dt.SetCellFloat(ly.Nm+"_FF_Scale", row, float64(ffpj.GScale.Scale))
		if ly.NRecvPrjns() > 1 {
			fbpj := ly.RecvPrjn(1).(*axon.Prjn)
			dt.SetCellFloat(ly.Nm+"_FB_AvgMaxG", row, float64(fbpj.GScale.AvgMax))
			dt.SetCellFloat(ly.Nm+"_FB_Scale", row, float64(fbpj.GScale.Scale))
		}
		dt.SetCellFloat(lnm+"_MaxGeM", row, float64(ly.ActAvg.AvgMaxGeM))
		dt.SetCellFloat(lnm+"_ActAvg", row, float64(ly.ActAvg.ActMAvg))
		dt.SetCellFloat(lnm+"_AvgDifAvg", row, float64(ly.Pools[0].AvgDif.Avg))
		dt.SetCellFloat(lnm+"_AvgDifMax", row, float64(ly.Pools[0].AvgDif.Max))
		dt.SetCellFloat(lnm+"_GiMult", row, float64(ly.ActAvg.GiMult))
		if ss.EpcPctErr > 0 && ss.EpcPctErr < 1 {
			dt.SetCellFloat(lnm+"_CorCosDiff", row, ss.TrnErrStats.CellFloat(lnm+"_CosDiff:Mean", 0))
			dt.SetCellFloat(lnm+"_ErrCosDiff", row, ss.TrnErrStats.CellFloat(lnm+"_CosDiff:Mean", 1))
			dt.SetCellFloat(lnm+"_CorActDif", row, ss.TrnErrStats.CellFloat(lnm+"_ActDif:Mean", 0))
			dt.SetCellFloat(lnm+"_ErrActDif", row, ss.TrnErrStats.CellFloat(lnm+"_ActDif:Mean", 1))
		}
		hog, dead, gnmda, ggabab := ss.HogDead(lnm)
		dt.SetCellFloat(lnm+"_Hog", row, hog)
		dt.SetCellFloat(lnm+"_Dead", row, dead)
		dt.SetCellFloat(lnm+"_Gnmda", row, gnmda)
		dt.SetCellFloat(lnm+"_GgabaB", row, ggabab)
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnEpcPlot.GoUpdate()
	if ss.TrnEpcFile != nil {
		if ss.TrainEnv.Run.Cur == ss.StartRun && epc == 0 {
			dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigTrnEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"UnitErr", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"PctErr2", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"ErrTrgAct", etensor.FLOAT64, nil, nil},
		{"CorTrgAct", etensor.FLOAT64, nil, nil},
		{"PerTrlMSec", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + "_FF_AvgMaxG", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_FF_Scale", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_FB_AvgMaxG", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_FB_Scale", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_MaxGeM", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_ActAvg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_AvgDifAvg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_AvgDifMax", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_GiMult", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_CorCosDiff", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_ErrCosDiff", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_CorActDif", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_ErrActDif", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_Hog", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_Dead", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_Gnmda", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_GgabaB", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_PCA_NStrong", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_PCA_Mean", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Object Recognition Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("UnitErr", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctErr2", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("ErrTrgAct", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CorTrgAct", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PerTrlMSec", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+"_FF_AvgMaxG", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .5)
		plt.SetColParams(lnm+"_FF_Scale", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .5)
		plt.SetColParams(lnm+"_FB_AvgMaxG", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .5)
		plt.SetColParams(lnm+"_FB_Scale", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .5)
		plt.SetColParams(lnm+"_MaxGeM", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_ActAvg", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0.5)
		plt.SetColParams(lnm+"_AvgDifAvg", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_AvgDifMax", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_GiMult", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_CorCosDiff", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_ErrCosDiff", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_CorActDif", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_ErrActDif", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_Hog", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_Dead", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_Gnmda", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_GgabaB", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_PCA_NStrong", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_PCA_Mean", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	}
	return plt
}

//////////////////////////////////////////////
//  SpikeRasters

// SpikeRastTsr gets spike raster tensor of given name, creating if not yet made
func (ss *Sim) SpikeRastTsr(name string) *etensor.Float32 {
	if ss.SpikeRasters == nil {
		ss.SpikeRasters = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.SpikeRasters[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.SpikeRasters[name] = tsr
	}
	return tsr
}

// SpikeRastGrid gets spike raster grid of given name, creating if not yet made
func (ss *Sim) SpikeRastGrid(name string) *etview.TensorGrid {
	if ss.SpikeRastGrids == nil {
		ss.SpikeRastGrids = make(map[string]*etview.TensorGrid)
	}
	tsr, ok := ss.SpikeRastGrids[name]
	if !ok {
		tsr = &etview.TensorGrid{}
		ss.SpikeRastGrids[name] = tsr
	}
	return tsr
}

// SetSpikeRastCol sets column of given spike raster from data
func (ss *Sim) SetSpikeRastCol(sr, vl *etensor.Float32, col int) {
	for ni, v := range vl.Values {
		sr.Set([]int{ni, col}, v)
	}
}

// ConfigSpikeGrid configures the spike grid
func (ss *Sim) ConfigSpikeGrid(tg *etview.TensorGrid, sr *etensor.Float32) {
	tg.SetStretchMax()
	sr.SetMetaData("grid-fill", "1")
	sr.SetMetaData("grid-min", "2")
	tg.SetTensor(sr)
}

// ConfigSpikeRasts configures spike rasters
func (ss *Sim) ConfigSpikeRasts() {
	ncy := 200 // max cycles
	// spike rast
	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		sr := ss.SpikeRastTsr(lnm)
		sr.SetShape([]int{ly.Shp.Len(), ncy}, nil, []string{"Nrn", "Cyc"})
	}
}

// RecSpikes records spikes
func (ss *Sim) RecSpikes(cyc int) {
	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		tv := ss.ValsTsr(lnm)
		ly.UnitValsTensor(tv, "Spike")
		sr := ss.SpikeRastTsr(lnm)
		ss.SetSpikeRastCol(sr, tv, cyc)
	}
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	// inp := ss.Net.LayerByName("V1").(axon.AxonLayer).AsAxon()
	// out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()

	trl := ss.TestEnv.Trial.Cur
	row := trl

	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellFloat("Obj", row, float64(ss.TestEnv.CurLED))
	dt.SetCellString("TrialName", row, ss.TestEnv.String())
	dt.SetCellFloat("Err", row, ss.TrlErr)
	dt.SetCellFloat("Err2", row, ss.TrlErr2)
	dt.SetCellFloat("UnitErr", row, ss.TrlUnitErr)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		dt.SetCellFloat(lnm+"_ActM.Avg", row, float64(ly.Pools[0].ActM.Avg))
	}
	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	// inp := ss.Net.LayerByName("V1").(axon.AxonLayer).AsAxon()
	// out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()

	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Trial.Max
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"Obj", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Err", etensor.FLOAT64, nil, nil},
		{"UnitErr", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + "_ActM.Avg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Object Recognition Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Obj", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Err", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("UnitErr", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0) // default plot
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+"_ActM.Avg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.5)
	}
	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

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

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) LogRun(dt *etable.Table) {
	run := ss.TrainEnv.Run.Cur // this is NOT triggered by increment yet -- use Cur
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epclog := ss.TrnEpcLog
	epcix := etable.NewIdxView(epclog)
	// compute mean over last N epochs for run level
	nlast := 5
	if nlast > epcix.Len()-1 {
		nlast = epcix.Len() - 1
	}
	epcix.Idxs = epcix.Idxs[epcix.Len()-nlast:]

	// params := ss.Params.Name
	params := "params"

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Params", row, params)
	dt.SetCellFloat("FirstZero", row, float64(ss.FirstZero))
	dt.SetCellFloat("UnitErr", row, agg.Mean(epcix, "UnitErr")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "PctCor")
	ss.RunStats = spl.AggsToTable(etable.AddAggName)

	// note: essential to use Go version of update when called from another goroutine
	ss.RunPlot.GoUpdate()
	if ss.RunFile != nil {
		if row == 0 {
			dt.WriteCSVHeaders(ss.RunFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.RunFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigRunLog(dt *etable.Table) {
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Record of performance at end of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Params", etensor.STRING, nil, nil},
		{"FirstZero", etensor.FLOAT64, nil, nil},
		{"UnitErr", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Object Recognition Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("FirstZero", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0) // default plot
	plt.SetColParams("UnitErr", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
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
	width := 1600
	height := 1200

	gi.SetAppName("objrec")
	gi.SetAppAbout(`This simulation explores how a hierarchy of areas in the ventral stream of visual processing (up to inferotemporal (IT) cortex) can produce robust object recognition that is invariant to changes in position, size, etc of retinal input images. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch6/objrec/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewMainWindow("objrec", "Object Recognition", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = mat32.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	nv.SetNet(ss.Net)
	ss.NetView = nv
	ss.ConfigNetView(nv)

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnTrlPlot").(*eplot.Plot2D)
	ss.TrnTrlPlot = ss.ConfigTrnTrlPlot(plt, ss.TrnTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

	tg := tv.AddNewTab(etview.KiT_TensorGrid, "Image").(*etview.TensorGrid)
	tg.SetStretchMax()
	ss.CurImgGrid = tg
	tg.SetTensor(&ss.TrainEnv.Vis.ImgTsr)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

	stb := tv.AddNewTab(gi.KiT_Layout, "Spike Rasters").(*gi.Layout)
	stb.Lay = gi.LayoutVert
	stb.SetStretchMax()
	for _, lnm := range ss.LayStatNms {
		sr := ss.SpikeRastTsr(lnm)
		tg := ss.SpikeRastGrid(lnm)
		tg.SetName(lnm + "Spikes")
		gi.AddNewLabel(stb, lnm, lnm+":")
		stb.AddChild(tg)
		gi.AddNewSpace(stb, lnm+"_spc")
		ss.ConfigSpikeGrid(tg, sr)
	}

	ss.ActRFGrids = make(map[string]*etview.TensorGrid)
	for _, nm := range ss.ActRFNms {
		tg := tv.AddNewTab(etview.KiT_TensorGrid, nm).(*etview.TensorGrid)
		tg.SetStretchMax()
		ss.ActRFGrids[nm] = tg
	}

	split.SetSplits(.2, .8)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Train", Icon: "run", Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.IsRunning)
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.Train()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Trial", Icon: "step-fwd", Tooltip: "Advances one training trial at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TrainTrial()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Epoch", Icon: "fast-fwd", Tooltip: "Advances one epoch (complete set of training patterns) at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainEpoch()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Run", Icon: "fast-fwd", Tooltip: "Advances one full training Run at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainRun()
		}
	})

	tbar.AddSeparator("spcl")

	tbar.AddAction(gi.ActOpts{Label: "Open Trained Wts", Icon: "update", Tooltip: "open weights trained on first phase of training (excluding 'novel' objects)", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.OpenTrainedWts()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Train Novel", Icon: "update", Tooltip: "prepares network for training novel items: loads saved weight, changes PNovel -- just do Step Run after this..", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.TrainNovel()
		vp.SetNeedsFullRender()
	})

	tbar.AddSeparator("test")

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrial(false) // don't break on chg
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Item", Icon: "step-fwd", Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		gi.StringPromptDialog(vp, "", "Test Item",
			gi.DlgOpts{Title: "Test Item", Prompt: "Enter the Name of a given input pattern to test (case insensitive, contains given string."},
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				dlg := send.(*gi.Dialog)
				if sig == int64(gi.DialogAccepted) {
					val := gi.StringPromptDialogValue(dlg)
					idx, _ := strconv.Atoi(val)
					if !ss.IsRunning {
						ss.IsRunning = true
						fmt.Printf("testing index: %v\n", idx)
						ss.TestItem(idx)
						ss.IsRunning = false
						vp.SetNeedsFullRender()
					}
				}
			})
	})

	tbar.AddAction(gi.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAll()
		}
	})

	tbar.AddSeparator("log")

	tbar.AddAction(gi.ActOpts{Label: "Reset RunLog", Icon: "update", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch6/objrec/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	// note: Command in shortcuts is automatically translated into Control for
	// Linux, Windows or Meta for MacOS
	// fmen := win.MainMenu.ChildByName("File", 0).(*gi.Action)
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Open", Shortcut: "Command+O"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		FileViewOpenSVG(vp)
	// 	})
	// fmen.Menu.AddSeparator("csep")
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		win.Close()
	// 	})

	/*
		inQuitPrompt := false
		gi.SetQuitReqFunc(func() {
			if inQuitPrompt {
				return
			}
			inQuitPrompt = true
			gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
				Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, gi.AddOk, gi.AddCancel,
				win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
					if sig == int64(gi.DialogAccepted) {
						gi.Quit()
					} else {
						inQuitPrompt = false
					}
				})
		})

		// gi.SetQuitCleanFunc(func() {
		// 	fmt.Printf("Doing final Quit cleanup here..\n")
		// })

		inClosePrompt := false
		win.SetCloseReqFunc(func(w *gi.Window) {
			if inClosePrompt {
				return
			}
			inClosePrompt = true
			gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Close Window?",
				Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, gi.AddOk, gi.AddCancel,
				win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
					if sig == int64(gi.DialogAccepted) {
						gi.Quit()
					} else {
						inClosePrompt = false
					}
				})
		})
	*/

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
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
	var note string
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.StringVar(&note, "note", "", "user note -- describe the run params etc")
	flag.IntVar(&ss.StartRun, "run", 0, "starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1")
	flag.IntVar(&ss.MaxRuns, "runs", 1, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.IntVar(&ss.MaxEpcs, "epcs", 50, "number of epochs per run")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", false, "if true, save run epoch log to file")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.Parse()
	ss.Init()

	if note != "" {
		fmt.Printf("note: %s\n", note)
	}
	if ss.ParamSet != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	if saveEpcLog {
		var err error
		fnm := ss.LogFileName("epc")
		ss.TrnEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnEpcFile = nil
		} else {
			fmt.Printf("Saving epoch log to: %s\n", fnm)
			defer ss.TrnEpcFile.Close()
		}
	}
	if saveRunLog {
		var err error
		fnm := ss.LogFileName("run")
		ss.RunFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.RunFile = nil
		} else {
			fmt.Printf("Saving run log to: %s\n", fnm)
			defer ss.RunFile.Close()
		}
	}
	if ss.SaveWts {
		fmt.Printf("Saving final weights per run\n")
	}
	fmt.Printf("Running Runs: %d - %d\n", ss.StartRun, ss.MaxRuns)
	ss.TrainEnv.Run.Set(ss.StartRun)
	ss.TrainEnv.Run.Max = ss.MaxRuns
	ss.NewRun()
	ss.Train()
}
