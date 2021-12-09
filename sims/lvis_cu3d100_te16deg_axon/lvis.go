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
	"strconv"
	"strings"
	"time"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/actrf"
	"github.com/emer/emergent/confusion"
	"github.com/emer/emergent/decoder"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/empi/empi"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/convolve"
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

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "needs some special inhibition and learning params",
				Params: params.Params{
					"Layer.Inhib.Inhib.AvgTau":           "30",   // 30 > 20 >> 1 definitively
					"Layer.Inhib.Inhib.GiSynThr":         "0.0",  // 0.01 shows effects
					"Layer.Act.Dt.IntTau":                "40",   // 40 > 20
					"Layer.Inhib.Layer.Gi":               "1.1",  // 1.1 > 1.0 > 1.2 -- all layers
					"Layer.Inhib.Pool.Gi":                "1.1",  // 1.1 > 1.0 -- universal for all layers
					"Layer.Inhib.Pool.FFEx0":             "0.15", // .15 > .18; Ex .05 -- .2/.1, .2/.2, .3/.5 all blow up
					"Layer.Inhib.Pool.FFEx":              "0.05", // .05 best so far
					"Layer.Inhib.Layer.FFEx0":            "0.15",
					"Layer.Inhib.Layer.FFEx":             "0.05", // .05 best so far
					"Layer.Act.Gbar.L":                   "0.2",  // 0.2 orig > 0.1 new def
					"Layer.Act.Decay.Act":                "0.2",  // 0.2 > 0 > 0.5 w/ glong.7 459
					"Layer.Act.Decay.Glong":              "0.6",  // 0.6 > 0.7 > 0.8
					"Layer.Act.KNa.Fast.Max":             "0.1",  // fm both .2 worse
					"Layer.Act.KNa.Med.Max":              "0.2",  // 0.2 > 0.1 def
					"Layer.Act.KNa.Slow.Max":             "0.2",  // 0.2 > higher
					"Layer.Act.Noise.Dist":               "Gaussian",
					"Layer.Act.Noise.Mean":               "0.0",     // .05 max for blowup
					"Layer.Act.Noise.Var":                "0.01",    // .01 a bit worse
					"Layer.Act.Noise.Type":               "NoNoise", // off for now
					"Layer.Act.GTarg.GeMax":              "1.2",     // 1 > .8 -- rescaling not very useful.
					"Layer.Act.Dt.LongAvgTau":            "20",      // 50 > 20 in terms of stability, but weird effect late
					"Layer.Learn.ActAvg.MinLrn":          "0.02",    // sig improves "top5" hogging in pca strength
					"Layer.Learn.ActAvg.SSTau":           "40",
					"Layer.Inhib.ActAvg.AdaptRate":       "0.5",   // 0.5 default for layers, except output
					"Layer.Learn.TrgAvgAct.ErrLrate":     "0.01",  // 0.01 orig > 0.005
					"Layer.Learn.TrgAvgAct.SynScaleRate": "0.005", // 0.005 orig > 0.01
					"Layer.Learn.TrgAvgAct.TrgRange.Min": "0.5",   // .5 > .2 overall
					"Layer.Learn.TrgAvgAct.TrgRange.Max": "2.0",   // objrec 2 > 1.8
					"Layer.Learn.RLrate.On":              "true",  // true = essential -- prevents over rep of
					"Layer.Learn.RLrate.ActThr":          "0.1",   // 0.1 > 0.15 > 0.05 > 0.2
					"Layer.Learn.RLrate.ActDifThr":       "0.02",  // 0.02 > 0.05 in other models
					"Layer.Learn.RLrate.Min":             "0.001", // .001 best, adifthr.05
				}},
			{Sel: ".Input", Desc: "all V1 input layers",
				Params: params.Params{
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Layer.Gi":    "0.9",     // 0.9 >= 1.1 def -- more activity -- clamp.Ge more important
					"Layer.Inhib.Pool.Gi":     "0.9",     // 0.9 >= 1.1 def -- more activity
					"Layer.Inhib.ActAvg.Init": "0.06",    // .06 for !SepColor actuals: V1m8: .04, V1m16: .03
					"Layer.Act.Clamp.Type":    "GeClamp", // GeClamp better
					"Layer.Act.Clamp.Ge":      "1.0",     // 1.0 > .6 -- more activity
					"Layer.Act.Decay.Act":     "1",       // these make no diff
					"Layer.Act.Decay.Glong":   "1",
				}},
			{Sel: ".V2", Desc: "pool inhib, sparse activity",
				Params: params.Params{
					"Layer.Inhib.Topo.On":    "false",
					"Layer.Inhib.Topo.Width": "4",
					"Layer.Inhib.Topo.Sigma": "1.0",
					"Layer.Inhib.Topo.Gi":    "0.002", // 0.002 best -- reduces Top5, keeps NStrong
					"Layer.Inhib.Topo.FF0":   "0.2",   // 0.2 best -- test more
				}},
			{Sel: ".V2m", Desc: "pool inhib, sparse activity",
				Params: params.Params{
					"Layer.Inhib.Pool.On":     "true", // needs pool-level
					"Layer.Inhib.Layer.FB":    "1",    // 0 possibly causes blowup at some point, no bene
					"Layer.Inhib.ActAvg.Init": "0.02",
				}},
			{Sel: ".V2l", Desc: "pool inhib, sparse activity",
				Params: params.Params{
					"Layer.Inhib.Pool.On":     "true", // needs pool-level
					"Layer.Inhib.Layer.FB":    "1",
					"Layer.Inhib.ActAvg.Init": "0.02",
					"Layer.Inhib.Topo.Width":  "2", // smaller
				}},
			{Sel: "#V2l16", Desc: "this layer is too active, drives V4f16 too strongly",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.02", // not clear if needed now..
				}},
			{Sel: ".V2h", Desc: "pool inhib, sparse activity",
				Params: params.Params{
					"Layer.Inhib.Pool.On":     "true", // needs pool-level
					"Layer.Inhib.Layer.FB":    "1",    // 0 possibly causes blowup at some point, no bene
					"Layer.Inhib.ActAvg.Init": "0.02",
				}},
			{Sel: ".V3h", Desc: "pool inhib, sparse activity -- only for h16",
				Params: params.Params{
					"Layer.Inhib.Pool.On":     "true", // needs pool-level
					"Layer.Inhib.Layer.FB":    "1",    // 0 possibly causes blowup at some point, no bene
					"Layer.Inhib.ActAvg.Init": "0.02", // .02 > .04
					"Layer.Act.GTarg.GeMax":   "1.2",  // these need to get stronger?
				}},
			{Sel: ".V4", Desc: "pool inhib, sparse activity",
				Params: params.Params{
					"Layer.Inhib.Pool.On":     "true", // needs pool-level
					"Layer.Inhib.Layer.FB":    "1",    // 1 >= 0 in lba
					"Layer.Inhib.ActAvg.Init": "0.04", // .04 >= .03 > .05
					"Layer.Inhib.Layer.Gi":    "1.1",  // was 1.1
					"Layer.Inhib.Pool.Gi":     "1.1",  // was 1.1
					"Layer.Inhib.Topo.On":     "false",
					"Layer.Inhib.Topo.Width":  "4", // was 4
					"Layer.Inhib.Topo.Sigma":  "1.0",
					"Layer.Inhib.Topo.Gi":     "0.002", // 0.002 best -- reduces Top5, keeps NStrong
					"Layer.Inhib.Topo.FF0":    "0.2",   // 0.2 best -- test more
				}},
			{Sel: ".TEO", Desc: "initial activity",
				Params: params.Params{
					"Layer.Inhib.Pool.On":     "true",  // needs pool-level
					"Layer.Inhib.Layer.On":    "false", // no layer!
					"Layer.Inhib.ActAvg.Init": "0.06",  // .06 > .05 = .04
					"Layer.Inhib.Pool.Gi":     "1.1",   // was 1.1
				}},
			{Sel: "#TE", Desc: "initial activity",
				Params: params.Params{
					"Layer.Inhib.Pool.On":     "true",  // needs pool-level
					"Layer.Inhib.Layer.On":    "false", // no layer!
					"Layer.Inhib.ActAvg.Init": "0.06",  // .03 actual with gi 1.2, was .06
					"Layer.Inhib.Pool.Gi":     "1.1",   // was 1.1
				}},
			{Sel: "#Output", Desc: "general output, Localist default -- see RndOutPats, LocalOutPats",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":         "1.3",   // 1.3 adapt > fixed: 1.2, 1.23 too low, 1.25, 1.3 too high
					"Layer.Inhib.ActAvg.Init":      "0.005", // .005 > .008 > .01 -- prevents loss of Ge over time..
					"Layer.Inhib.ActAvg.Targ":      "0.01",  // .01 -- .005, .008 too low -- maybe not nec?
					"Layer.Inhib.ActAvg.AdaptGi":   "true",  // true: it is essential -- too hard to balance manually
					"Layer.Inhib.ActAvg.LoTol":     "0.5",
					"Layer.Inhib.ActAvg.AdaptRate": "0.02", // 0.01 >= 0.02 best in range 0.01..0.1
					// "Layer.Act.Decay.Act":        "0.5", // 0.5 makes no diff
					// "Layer.Act.Decay.Glong":      "1", // 1 makes no diff
					"Layer.Act.Clamp.Type":     "GeClamp",
					"Layer.Act.Clamp.Ge":       "0.6", // .6 = .7 > .5 (tiny diff) -- input has 1.0 now
					"Layer.Act.Clamp.Burst":    "false",
					"Layer.Act.Clamp.BurstThr": "0.5",   //
					"Layer.Act.Clamp.BurstGe":  "2",     // 2, 20cyc with tr 2 or 3, ge .6 all about same
					"Layer.Act.Clamp.BurstCyc": "20",    // 20 > 15 > 10 -- maybe refractory?
					"Layer.Act.Spike.Tr":       "3",     // 2 >= 3 > 1 > 0
					"Layer.Act.GABAB.Gbar":     "0.005", // .005 > .01 > .02 > .05 > .1 > .2
					"Layer.Act.NMDA.Gbar":      "0.03",  // was .02
					"Layer.Learn.RLrate.On":    "true",  // todo: try false
					"Layer.Inhib.Pool.FFEx":    "0.0",   // no
					"Layer.Inhib.Layer.FFEx":   "0.0",   //
				}},
			{Sel: "#Claustrum", Desc: "testing -- not working",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.8",
					"Layer.Inhib.Pool.On":     "false", // needs pool-level
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Inhib.ActAvg.Init": ".06",
				}},
			///////////////////////////////
			// projections
			{Sel: "Prjn", Desc: "exploring",
				Params: params.Params{
					"Prjn.PrjnScale.ScaleLrate": "2",      // 2 = fast response, effective
					"Prjn.PrjnScale.LoTol":      "0.8",    // good now...
					"Prjn.PrjnScale.AvgTau":     "500",    // slower default
					"Prjn.PrjnScale.Adapt":      "false",  // no adapt better?
					"Prjn.SWt.Adapt.On":         "true",   // true > false, esp in cosdiff
					"Prjn.SWt.Adapt.Lrate":      "0.0002", // .0002, .001 > .01 > .1 after 250epc in NStrong
					"Prjn.SWt.Adapt.SigGain":    "6",
					"Prjn.SWt.Adapt.DreamVar":   "0.02",   // 0.02 good overall, no ToOut
					"Prjn.SWt.Init.SPct":        "1",      // 1 > lower
					"Prjn.SWt.Init.Mean":        "0.5",    // .5 > .4 -- key, except v2?
					"Prjn.SWt.Limit.Min":        "0.2",    // .2-.8 == .1-.9; .3-.7 not better -- 0-1 minor worse
					"Prjn.SWt.Limit.Max":        "0.8",    //
					"Prjn.Learn.Lrate.Base":     "0.04",   // 0.01 > 0.015 > 0.02 459 -- .04 for RLrate
					"Prjn.Learn.XCal.SubMean":   "1",      // testing..
					"Prjn.Learn.XCal.DWtThr":    "0.0001", // 0.0001 > 0.001
					"Prjn.Com.PFail":            "0.0",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates -- smaller as network gets bigger",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2",
					// "Prjn.Learn.Lrate.Base": "0",
				}},
			{Sel: ".Forward", Desc: "use pfail only on forward cons?",
				Params: params.Params{
					// .2 max 1 = no diff, .5 max .8 = no diff
					"Prjn.Com.PFail": "0.0", // 0 > .05 > .1 > .2
					// "Prjn.SWt.Adapt.DreamVar": "0.02", // 0.01 big pca effects, no perf bene; 0.05 impairs perf
				}},
			{Sel: ".ToOut", Desc: "to output -- some things should be different..",
				Params: params.Params{
					"Prjn.Com.PFail":          "0.0",
					"Prjn.SWt.Adapt.DreamVar": "0.0",   // nope
					"Prjn.SWt.Adapt.On":       "false", // off > on
					"Prjn.SWt.Init.SPct":      "0",     // when off, 0
					"Prjn.PrjnScale.LoTol":    "0.5",   // .5 > .8 -- needs extra kick at start!
					"Prjn.PrjnScale.Adapt":    "true",  // was essential here
					"Prjn.Learn.XCal.SubMean": "0",
				}},
			{Sel: ".FmOut", Desc: "from output -- some things should be different..",
				Params: params.Params{
					"Prjn.Learn.XCal.SubMean": "1",
				}},
			{Sel: ".Inhib", Desc: "inhibitory projection",
				Params: params.Params{
					"Prjn.Learn.Learn":      "true",   // learned decorrel is good
					"Prjn.Learn.Lrate.Base": "0.0001", // .0001 > .001 -- slower better!
					"Prjn.SWt.Init.Var":     "0.0",
					"Prjn.SWt.Init.Mean":    "0.1",
					"Prjn.SWt.Init.Sym":     "false",
					"Prjn.SWt.Adapt.On":     "false",
					"Prjn.PrjnScale.Abs":    "0.2", // .2 > .1 for controlling PCA; .3 or.4 with GiSynThr .01
					"Prjn.PrjnScale.Adapt":  "false",
					"Prjn.IncGain":          "1", // .5 def
				}},
			{Sel: ".V1V2", Desc: "special SWt params",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.4", // .4 here is key!
					"Prjn.SWt.Limit.Min": "0.1", // .1-.7
					"Prjn.SWt.Limit.Max": "0.7", //
					"Prjn.PrjnScale.Abs": "1.4", // 1.4 > 2.0 for color -- extra boost to get more v2 early on
				}},
			{Sel: ".V1V2fmSm", Desc: "weaker",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2",
				}},
			{Sel: ".V2V4", Desc: "extra boost",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.0", // 1.0 prev, 1.2 not better
					"Prjn.SWt.Init.Mean": "0.4", // .4 a tiny bit better overall
					"Prjn.SWt.Limit.Min": "0.1", // .1-.7 def
					"Prjn.SWt.Limit.Max": "0.7", //
				}},
			{Sel: ".V2V4sm", Desc: "extra boost",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.0", // 1.0 prev, 1.2 not better
				}},
			{Sel: "#V2m16ToV4f16", Desc: "weights into V416 getting too high",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.0", // was 0.8, but as of #680 1.0 better
				}},
			{Sel: "#V2l16ToV4f16", Desc: "weights into V416 getting too high",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.0", // see above
				}},
			{Sel: ".V4TEO", Desc: "stronger",
				Params: params.Params{
					// "Prjn.PrjnScale.Abs": "1.2", // trying bigger -- was low
				}},
			{Sel: ".V4TEOoth", Desc: "weaker rel",
				Params: params.Params{
					// "Prjn.PrjnScale.Abs": "1.2", // trying bigger -- was low
					"Prjn.PrjnScale.Rel": "0.5",
				}},
			{Sel: ".V4Out", Desc: "NOT weaker",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1", // 1 > 0.5 > .2 -- v53 still
				}},
			{Sel: ".TEOTE", Desc: "too weak at start",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1", // 1.2 not better
				}},

			// back projections
			{Sel: ".V4V2", Desc: "weaker",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.05", // .05 > .02 > .1 v70
					"Prjn.SWt.Init.Mean": "0.4",  // .4 matches V2V4 -- not that big a diff on its own
					"Prjn.SWt.Limit.Min": "0.1",  // .1-.7 def
					"Prjn.SWt.Limit.Max": "0.7",  //
				}},
			{Sel: ".TEOV2", Desc: "weaker -- not used",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.05", // .05 > .02 > .1
				}},
			{Sel: ".TEOV4", Desc: "weaker",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1", // .1 == .2
				}},
			{Sel: ".TETEO", Desc: "std",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1", // .1 orig
				}},
			{Sel: ".OutTEO", Desc: "weaker",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3", // .3 > .2 v53 in long run
				}},
			{Sel: ".OutV4", Desc: "weaker",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1", // .1 > .2 v53
				}},
			{Sel: "#OutputToTE", Desc: "weaker",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3", // 0.3 > .2 v53 in long run
				}},
			{Sel: "#TEToOutput", Desc: "weaker",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1.0", // turn off for TE testing
				}},

			// shortcuts -- .5 > .2 (v32 still) -- all tested together
			{Sel: "#V1l16ToClaustrum", Desc: "random fixed -- not useful",
				Params: params.Params{
					"Prjn.Learn.Learn":   "false",
					"Prjn.PrjnScale.Rel": "0.5",   // .5 > .8 > 1 > .4 > .3 etc
					"Prjn.SWt.Adapt.On":  "false", // seems better
				}},
			{Sel: ".V1SC", Desc: "v1 shortcut",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base": "0.001", //
					// "Prjn.Learn.Learn":      "false",
					"Prjn.PrjnScale.Rel": "0.5",   // .5 > .8 > 1 > .4 > .3 etc
					"Prjn.SWt.Adapt.On":  "false", // seems better
					// "Prjn.SWt.Init.Var":  "0.05",
				}},
		},
	}},
	{Name: "RndOutPats", Desc: "random output pattern", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "#Output", Desc: "high inhib for one-hot output",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.9", // 0.9 > 1.0
					"Layer.Inhib.ActAvg.Init": "0.1", // 0.1 seems good
				}},
		},
	}},
	{Name: "LocalOutPats", Desc: "localist output pattern", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "#Output", Desc: "high inhib for one-hot output",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1.5", // 1.5 = 1.6 > 1.4
					"Layer.Inhib.ActAvg.Init": "0.01",
				}},
		},
	}},
	{Name: "ToOutTol", Desc: "delayed enforcement of low tolerance on .ToOut", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".ToOut", Desc: "to output -- some things should be different..",
				Params: params.Params{
					"Prjn.PrjnScale.LoTol": "0.5", // activation dropping off a cliff there at the end..
				}},
		},
	}},
	{Name: "OutAdapt", Desc: "delayed enforcement of output adaptation", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "#Output", Desc: "general output, Localist default -- see RndOutPats, LocalOutPats",
				Params: params.Params{
					"Layer.Inhib.ActAvg.AdaptGi": "true", // true = definitely worse
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
	Net             *axon.Network    `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	TrnTrlLog       *etable.Table    `view:"no-inline" desc:"training trial-level log data"`
	TrnCycLog       *etable.Table    `view:"no-inline" desc:"training cycle-level log data"`
	TrnTrlLogAll    *etable.Table    `view:"no-inline" desc:"all training trial-level log data (aggregated from MPI)"`
	TrnTrlRepLog    *etable.Table    `view:"no-inline" desc:"training trial-level reps log data"`
	TrnTrlRepLogAll *etable.Table    `view:"no-inline" desc:"training trial-level reps log data"`
	TrnEpcLog       *etable.Table    `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog       *etable.Table    `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog       *etable.Table    `view:"no-inline" desc:"testing trial-level log data"`
	TstTrlLogAll    *etable.Table    `view:"no-inline" desc:"all testing trial-level log data (aggregated from MPI)"`
	TrnErrStats     *etable.Table    `view:"no-inline" desc:"training error stats"`
	ActRFs          actrf.RFs        `view:"no-inline" desc:"activation-based receptive fields"`
	RunLog          *etable.Table    `view:"no-inline" desc:"summary log of each run"`
	RunStats        *etable.Table    `view:"no-inline" desc:"aggregate stats on all runs"`
	Confusion       confusion.Matrix `view:"no-inline" desc:"confusion matrix"`
	ConfusionEpc    int              `desc:"epoch to start recording confusion matrix"`
	MinusCycles     int              `desc:"number of minus-phase cycles"`
	PlusCycles      int              `desc:"number of plus-phase cycles"`
	SubPools        bool             `desc:"if true, organize layers and connectivity with 2x2 sub-pools within each topological pool"`
	RndOutPats      bool             `desc:"if true, use random output patterns -- else localist"`
	PostCycs        int              `desc:"number of cycles to run after main alphacyc cycles, between stimuli"`
	PostDecay       float32          `desc:"decay to apply at start of PostCycs"`
	ErrLrMod        axon.LrateMod    `view:"inline" desc:"learning rate modulation as function of error"`
	Params          params.Sets      `view:"no-inline" desc:"full collection of param sets"`
	ParamSet        string           `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don't put spaces in ParamSet names!)"`
	Tag             string           `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`

	Prjn4x4Skp2              *prjn.PoolTile    `view:"-" desc:"Standard feedforward topographic projection, recv = 1/2 send size"`
	Prjn4x4Skp2Recip         *prjn.PoolTile    `view:"-" desc:"Reciprocal"`
	Prjn4x4Skp2Sub2          *prjn.PoolTileSub `view:"-" desc:"Standard feedforward topographic projection, recv = 1/2 send size"`
	Prjn4x4Skp2Sub2Recip     *prjn.PoolTileSub `view:"-" desc:"Reciprocal"`
	Prjn4x4Skp2Sub2Send      *prjn.PoolTileSub `view:"-" desc:"Standard feedforward topographic projection, recv = 1/2 send size"`
	Prjn4x4Skp2Sub2SendRecip *prjn.PoolTileSub `view:"-" desc:"Standard feedforward topographic projection, recv = 1/2 send size"`
	Prjn2x2Skp1              *prjn.PoolTile    `view:"-" desc:"same-size prjn"`
	Prjn2x2Skp1Recip         *prjn.PoolTile    `view:"-" desc:"same-size prjn reciprocal"`
	Prjn2x2Skp1Sub2          *prjn.PoolTileSub `view:"-" desc:"same-size prjn"`
	Prjn2x2Skp1Sub2Recip     *prjn.PoolTileSub `view:"-" desc:"same-size prjn reciprocal"`
	Prjn2x2Skp1Sub2Send      *prjn.PoolTileSub `view:"-" desc:"same-size prjn"`
	Prjn2x2Skp1Sub2SendRecip *prjn.PoolTileSub `view:"-" desc:"same-size prjn reciprocal"`
	Prjn2x2Skp2              *prjn.PoolTileSub `view:"-" desc:"lateral inhib projection"`
	Prjn4x4Skp0              *prjn.PoolTile    `view:"-" desc:"for V4 <-> TEO"`
	Prjn4x4Skp0Recip         *prjn.PoolTile    `view:"-" desc:"for V4 <-> TEO"`
	Prjn4x4Skp0Sub2          *prjn.PoolTileSub `view:"-" desc:"for V4 <-> TEO"`
	Prjn4x4Skp0Sub2Recip     *prjn.PoolTileSub `view:"-" desc:"for V4 <-> TEO"`
	Prjn1x1Skp0              *prjn.PoolTile    `view:"-" desc:"for TE <-> TEO"`
	Prjn1x1Skp0Recip         *prjn.PoolTile    `view:"-" desc:"for TE <-> TEO"`
	Prjn6x6Skp2Lat           *prjn.PoolTileSub `view:"-" desc:"lateral inhibitory connectivity for subpools"`

	StartRun       int                           `desc:"starting run number -- typically 0 but can be set in command args for parallel runs on a cluster"`
	MaxRuns        int                           `desc:"maximum number of model runs to perform"`
	MaxEpcs        int                           `desc:"maximum number of epochs to run per model run"`
	MaxTrls        int                           `desc:"maximum number of training trials per epoch"`
	RepsInterval   int                           `desc:"how often to analyze the representations"`
	NZeroStop      int                           `desc:"if a positive number, training will stop after this many epochs with zero Err"`
	TrainEnv       ImagesEnv                     `desc:"Training environment"`
	TestEnv        ImagesEnv                     `desc:"Testing environment"`
	Decoder        decoder.SoftMax               `desc:"decoder for better output"`
	Time           axon.Time                     `desc:"axon timing parameters and state"`
	TestInterval   int                           `desc:"how often to run through the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	ViewOn         bool                          `desc:"whether to update the network view while running"`
	TrainUpdt      axon.TimeScales               `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt       axon.TimeScales               `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	InLays         []string                      `view:"-" desc:"input layers -- for stats"`
	OutLays        []string                      `view:"-" desc:"output layers -- for stats"`
	HidLays        []string                      `view:"-" desc:"hidden layers -- for all main stats"`
	ActRFNms       []string                      `desc:"names of layers to compute activation rfields on"`
	SpikeRastNms   []string                      `view:"-" desc:"spike raster layers"`
	SpikeRasters   map[string]*etensor.Float32   `desc:"spike raster data for different layers"`
	SpikeRastGrids map[string]*etview.TensorGrid `desc:"spike raster plots for different layers"`

	// statistics: note use float64 as that is best for etable.Table
	TrlCatIdx      int       `inactive:"+" desc:"current category index"`
	TrlCat         string    `inactive:"+" desc:"current category name"`
	TrlRespIdx     int       `inactive:"+" desc:"output response index of the network"`
	TrlResp        string    `inactive:"+" desc:"output response name of the network"`
	TrlErr         float64   `inactive:"+" desc:"1 if trial was error, 0 if correct -- based on max out unit"`
	TrlErr2        float64   `inactive:"+" desc:"1 if trial was error, 0 if correct -- based on second max out"`
	TrlUnitErr     float64   `inactive:"+" desc:"current trial's average sum squared error"`
	TrlTrgAct      float64   `inactive:"+" desc:"activity of target output unit on this trial"`
	TrlCosDiff     float64   `inactive:"+" desc:"current trial's cosine difference"`
	TrlOutFirstCyc int       `inactive:"+" desc:"first activation cycle of output"`
	TrlFirstResp   int       `inactive:"+" desc:"response at OutFirstCyc"`
	TrlFirstErr    float64   `inactive:"+" desc:"error on first output response"`
	TrlFirstErr2   float64   `inactive:"+" desc:"error2 on first output response"`
	TrlDecRespIdx  int       `inactive:"+" desc:"output response index of the network"`
	TrlDecErr      float64   `inactive:"+" desc:"1 if trial was error, 0 if correct -- based on decoder max"`
	TrlDecErr2     float64   `inactive:"+" desc:"1 if trial was error, 0 if correct -- based on decoder second max"`
	EpcUnitErr     float64   `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcPctErr      float64   `inactive:"+" desc:"last epoch's average TrlErr"`
	EpcPctCor      float64   `inactive:"+" desc:"1 - last epoch's average TrlErr"`
	EpcPctErr2     float64   `inactive:"+" desc:"last epoch's average TrlErr2"`
	EpcPctDecErr   float64   `inactive:"+" desc:"last epoch's average TrlDecErr"`
	EpcPctDecErr2  float64   `inactive:"+" desc:"last epoch's average TrlDecErr2"`
	EpcCosDiff     float64   `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	EpcErrTrgAct   float64   `inactive:"+" desc:"avg activity of target output unit on err trials"`
	EpcCorTrgAct   float64   `inactive:"+" desc:"avg activity of target output unit on correct trials"`
	EpcPerTrlMSec  float64   `inactive:"+" desc:"how long did the epoch take per trial in wall-clock milliseconds"`
	FirstZero      int       `inactive:"+" desc:"epoch at when Err first went to zero"`
	NZero          int       `inactive:"+" desc:"number of epochs in a row with zero Err"`
	PCA            pca.PCA   `view:"-" desc:"pca obj"`
	GaussKernel    []float64 `view:"-" desc:"gaussian smoothing kernel"`
	SmoothData     []float64 `view:"-" desc:"data for smoothing"`

	// internal state - view:"-"
	Win          *gi.Window                    `view:"-" desc:"main GUI window"`
	NetView      *netview.NetView              `view:"-" desc:"the network viewer"`
	ToolBar      *gi.ToolBar                   `view:"-" desc:"the master toolbar"`
	CurImgGrid   *etview.TensorGrid            `view:"-" desc:"the current image grid view"`
	ActRFGrids   map[string]*etview.TensorGrid `view:"-" desc:"the act rf grid views"`
	TrnTrlPlot   *eplot.Plot2D                 `view:"-" desc:"the training trial plot"`
	TrnCycPlot   *eplot.Plot2D                 `view:"-" desc:"the training cycle plot"`
	TrnEpcPlot   *eplot.Plot2D                 `view:"-" desc:"the training epoch plot"`
	TstEpcPlot   *eplot.Plot2D                 `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot   *eplot.Plot2D                 `view:"-" desc:"the test-trial plot"`
	RunPlot      *eplot.Plot2D                 `view:"-" desc:"the run plot"`
	TrnEpcFile   *os.File                      `view:"-" desc:"log file"`
	TrnTrlFile   *os.File                      `view:"-" desc:"log file"`
	TstEpcFile   *os.File                      `view:"-" desc:"log file"`
	TstTrlFile   *os.File                      `view:"-" desc:"log file"`
	RunFile      *os.File                      `view:"-" desc:"log file"`
	ValsTsrs     map[string]*etensor.Float32   `view:"-" desc:"for holding layer values"`
	SaveWts      bool                          `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui        bool                          `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams bool                          `view:"-" desc:"if true, print message for all params that are set"`
	IsRunning    bool                          `view:"-" desc:"true if sim is running"`
	StopNow      bool                          `view:"-" desc:"flag to stop running"`
	NeedsNewRun  bool                          `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeeds     []int64                       `view:"-" desc:"the current random seeds to use for each run"`
	LastEpcTime  time.Time                     `view:"-" desc:"timer for last epoch"`

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
	ss.TrnTrlLog = &etable.Table{}
	ss.TrnCycLog = &etable.Table{}
	ss.TrnTrlLogAll = &etable.Table{}
	ss.TrnTrlRepLog = &etable.Table{}
	ss.TrnTrlRepLogAll = &etable.Table{}
	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.TstTrlLogAll = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.Params = ParamSets
	ss.TestInterval = 20 // maybe causing issues?

	ss.Time.Defaults()
	ss.MinusCycles = 180
	ss.PlusCycles = 50
	ss.RepsInterval = 10
	ss.SubPools = true    // true
	ss.RndOutPats = false // change here
	if ss.RndOutPats {
		ss.ParamSet = "RndOutPats"
	}
	ss.PostCycs = 0
	ss.PostDecay = .5
	ss.ErrLrMod.Defaults()
	ss.ErrLrMod.Base = 0.05 // 0.05 > .02? > .1 in long-term output layer health
	ss.ErrLrMod.Range.Set(0.2, 0.8)

	ss.Prjn4x4Skp2 = prjn.NewPoolTile()
	ss.Prjn4x4Skp2.Size.Set(4, 4)
	ss.Prjn4x4Skp2.Skip.Set(2, 2)
	ss.Prjn4x4Skp2.Start.Set(-1, -1)
	ss.Prjn4x4Skp2.TopoRange.Min = 0.8
	ss.Prjn4x4Skp2Recip = prjn.NewPoolTileRecip(ss.Prjn4x4Skp2)

	ss.Prjn4x4Skp2Sub2 = prjn.NewPoolTileSub()
	ss.Prjn4x4Skp2Sub2.Size.Set(4, 4)
	ss.Prjn4x4Skp2Sub2.Skip.Set(2, 2)
	ss.Prjn4x4Skp2Sub2.Start.Set(-1, -1)
	ss.Prjn4x4Skp2Sub2.Subs.Set(2, 2)
	ss.Prjn4x4Skp2Sub2.TopoRange.Min = 0.8
	ss.Prjn4x4Skp2Sub2Recip = prjn.NewPoolTileSubRecip(ss.Prjn4x4Skp2Sub2)

	ss.Prjn4x4Skp2Sub2Send = prjn.NewPoolTileSub()
	*ss.Prjn4x4Skp2Sub2Send = *ss.Prjn4x4Skp2Sub2
	ss.Prjn4x4Skp2Sub2Send.SendSubs = true
	ss.Prjn4x4Skp2Sub2SendRecip = prjn.NewPoolTileSubRecip(ss.Prjn4x4Skp2Sub2Send)

	ss.Prjn2x2Skp1 = prjn.NewPoolTile()
	ss.Prjn2x2Skp1.Size.Set(2, 2)
	ss.Prjn2x2Skp1.Skip.Set(1, 1)
	ss.Prjn2x2Skp1.Start.Set(0, 0)
	ss.Prjn2x2Skp1.TopoRange.Min = 0.8
	ss.Prjn2x2Skp1Recip = prjn.NewPoolTileRecip(ss.Prjn2x2Skp1)

	ss.Prjn2x2Skp1Sub2 = prjn.NewPoolTileSub()
	ss.Prjn2x2Skp1Sub2.Size.Set(2, 2)
	ss.Prjn2x2Skp1Sub2.Skip.Set(1, 1)
	ss.Prjn2x2Skp1Sub2.Start.Set(0, 0)
	ss.Prjn2x2Skp1Sub2.Subs.Set(2, 2)
	ss.Prjn2x2Skp1Sub2.TopoRange.Min = 0.8

	ss.Prjn2x2Skp1Sub2Recip = prjn.NewPoolTileSubRecip(ss.Prjn2x2Skp1Sub2)

	ss.Prjn2x2Skp1Sub2Send = prjn.NewPoolTileSub()
	ss.Prjn2x2Skp1Sub2Send.Size.Set(2, 2)
	ss.Prjn2x2Skp1Sub2Send.Skip.Set(1, 1)
	ss.Prjn2x2Skp1Sub2Send.Start.Set(0, 0)
	ss.Prjn2x2Skp1Sub2Send.Subs.Set(2, 2)
	ss.Prjn2x2Skp1Sub2Send.SendSubs = true
	ss.Prjn2x2Skp1Sub2Send.TopoRange.Min = 0.8

	ss.Prjn2x2Skp1Sub2SendRecip = prjn.NewPoolTileSub()
	*ss.Prjn2x2Skp1Sub2SendRecip = *ss.Prjn2x2Skp1Sub2Send
	ss.Prjn2x2Skp1Sub2SendRecip.Recip = true

	ss.Prjn2x2Skp2 = prjn.NewPoolTileSub()
	ss.Prjn2x2Skp2.Size.Set(2, 2)
	ss.Prjn2x2Skp2.Skip.Set(2, 2)
	ss.Prjn2x2Skp2.Start.Set(0, 0)
	ss.Prjn2x2Skp2.Subs.Set(2, 2)

	ss.Prjn4x4Skp0 = prjn.NewPoolTile()
	ss.Prjn4x4Skp0.Size.Set(4, 4)
	ss.Prjn4x4Skp0.Skip.Set(0, 0)
	ss.Prjn4x4Skp0.Start.Set(0, 0)
	ss.Prjn4x4Skp0.GaussFull.Sigma = 1.5
	ss.Prjn4x4Skp0.GaussInPool.Sigma = 1.5
	ss.Prjn4x4Skp0.TopoRange.Min = 0.8
	ss.Prjn4x4Skp0Recip = prjn.NewPoolTileRecip(ss.Prjn4x4Skp0)

	ss.Prjn4x4Skp0Sub2 = prjn.NewPoolTileSub()
	ss.Prjn4x4Skp0Sub2.Size.Set(4, 4)
	ss.Prjn4x4Skp0Sub2.Skip.Set(0, 0)
	ss.Prjn4x4Skp0Sub2.Start.Set(0, 0)
	ss.Prjn4x4Skp0Sub2.Subs.Set(2, 2)
	ss.Prjn4x4Skp0Sub2.SendSubs = true
	ss.Prjn4x4Skp0Sub2.GaussFull.Sigma = 1.5
	ss.Prjn4x4Skp0Sub2.GaussInPool.Sigma = 1.5
	ss.Prjn4x4Skp0Sub2.TopoRange.Min = 0.8
	ss.Prjn4x4Skp0Sub2Recip = prjn.NewPoolTileSubRecip(ss.Prjn4x4Skp0Sub2)

	ss.Prjn1x1Skp0 = prjn.NewPoolTile()
	ss.Prjn1x1Skp0.Size.Set(1, 1)
	ss.Prjn1x1Skp0.Skip.Set(0, 0)
	ss.Prjn1x1Skp0.Start.Set(0, 0)
	ss.Prjn1x1Skp0.GaussFull.Sigma = 1.5
	ss.Prjn1x1Skp0.GaussInPool.Sigma = 1.5
	ss.Prjn1x1Skp0.TopoRange.Min = 0.8
	ss.Prjn1x1Skp0Recip = prjn.NewPoolTileRecip(ss.Prjn1x1Skp0)

	ss.Prjn6x6Skp2Lat = prjn.NewPoolTileSub()
	ss.Prjn6x6Skp2Lat.Size.Set(6, 6)
	ss.Prjn6x6Skp2Lat.Skip.Set(2, 2)
	ss.Prjn6x6Skp2Lat.Start.Set(-2, -2)
	ss.Prjn6x6Skp2Lat.Subs.Set(2, 2)
	ss.Prjn6x6Skp2Lat.TopoRange.Min = 0.8

	ss.RndSeeds = make([]int64, 100) // make enough for plenty of runs
	for i := 0; i < 100; i++ {
		ss.RndSeeds[i] = int64(i) + 1 // exclude 0
	}
	ss.ViewOn = true
	ss.TrainUpdt = axon.AlphaCycle
	ss.TestUpdt = axon.GammaCycle
	ss.ActRFNms = []string{"V4f16:Image", "V4f8:Output", "TEO8:Image", "TEO8:Output", "TEO16:Image", "TEO16:Output"}
	ss.SpikeRastNms = []string{"V1l16", "V2l16", "V4f16", "TEOf16", "TE", "Output"}
	ss.GaussKernel = convolve.GaussianKernel64(3, .5)
	// fmt.Printf("%v\n", ss.GaussKernel)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTrnTrlLog(ss.TrnTrlLog)
	ss.ConfigTrnTrlLog(ss.TrnTrlLogAll)
	ss.ConfigTrnTrlRepLog(ss.TrnTrlRepLog)
	ss.ConfigTrnTrlRepLog(ss.TrnTrlRepLogAll)
	ss.ConfigTrnCycLog(ss.TrnCycLog)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigTstTrlLog(ss.TstTrlLogAll)
	ss.ConfigSpikeRasts()
	ss.ConfigRunLog(ss.RunLog)
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
	v1m8.SetClass("V1m")
	v1l16.SetClass("V1l")
	v1l8.SetClass("V1l")

	// not useful so far..
	// clst := net.AddLayer2D("Claustrum", 5, 5, emer.Hidden)

	var v1cm16, v1cl16, v1cm8, v1cl8 emer.Layer
	if cdog {
		v1cm16 = net.AddLayer4D("V1Cm16", 16, 16, 2, 2, emer.Input)
		v1cl16 = net.AddLayer4D("V1Cl16", 8, 8, 2, 2, emer.Input)
		v1cm8 = net.AddLayer4D("V1Cm8", 16, 16, 2, 2, emer.Input)
		v1cl8 = net.AddLayer4D("V1Cl8", 8, 8, 2, 2, emer.Input)
		v1cm16.SetClass("V1Cm")
		v1cm8.SetClass("V1Cm")
		v1cl16.SetClass("V1Cl")
		v1cl8.SetClass("V1Cl")
	}

	v2m16 := net.AddLayer4D("V2m16", v2mNp, v2mNp, v2Nu, v2Nu, emer.Hidden)
	v2l16 := net.AddLayer4D("V2l16", v2lNp, v2lNp, v2Nu, v2Nu, emer.Hidden)
	v2m8 := net.AddLayer4D("V2m8", v2mNp, v2mNp, v2Nu, v2Nu, emer.Hidden)
	v2l8 := net.AddLayer4D("V2l8", v2lNp, v2lNp, v2Nu, v2Nu, emer.Hidden)
	v2m16.SetClass("V2m V2")
	v2m8.SetClass("V2m V2")
	v2l16.SetClass("V2l V2")
	v2l8.SetClass("V2l V2")

	var v1h16, v2h16, v3h16 emer.Layer
	if hi16 {
		v1h16 = net.AddLayer4D("V1h16", 32, 32, 5, 4, emer.Input)
		v2h16 = net.AddLayer4D("V2h16", 32, 32, v2Nu, v2Nu, emer.Hidden)
		v3h16 = net.AddLayer4D("V3h16", 16, 16, v2Nu, v2Nu, emer.Hidden)
		v1h16.SetClass("V1h")
		v2h16.SetClass("V2h V2")
		v3h16.SetClass("V3h")
	}

	v4f16 := net.AddLayer4D("V4f16", v4Np, v4Np, v4Nu, v4Nu, emer.Hidden)
	v4f8 := net.AddLayer4D("V4f8", v4Np, v4Np, v4Nu, v4Nu, emer.Hidden)
	v4f16.SetClass("V4")
	v4f8.SetClass("V4")

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

	var p4x4s2, p2x2s1, p4x4s2send, p2x2s1send, p4x4s2recip, p2x2s1recip, v4toteo, teotov4 prjn.Pattern
	p4x4s2 = ss.Prjn4x4Skp2
	p2x2s1 = ss.Prjn2x2Skp1
	p4x4s2send = ss.Prjn4x4Skp2
	p2x2s1send = ss.Prjn2x2Skp1
	p4x4s2recip = ss.Prjn4x4Skp2Recip
	p2x2s1recip = ss.Prjn2x2Skp1Recip
	v4toteo = full
	teotov4 = full

	if ss.SubPools {
		p4x4s2 = ss.Prjn4x4Skp2Sub2
		p2x2s1 = ss.Prjn2x2Skp1Sub2
		p4x4s2send = ss.Prjn4x4Skp2Sub2Send
		p2x2s1send = ss.Prjn2x2Skp1Sub2Send
		p4x4s2recip = ss.Prjn4x4Skp2Sub2SendRecip
		p2x2s1recip = ss.Prjn2x2Skp1Sub2SendRecip
		v4toteo = ss.Prjn4x4Skp0Sub2
		teotov4 = ss.Prjn4x4Skp0Sub2Recip
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
		v2inhib = ss.Prjn2x2Skp2 // ss.Prjn6x6Skp2Lat
		v4inhib = ss.Prjn2x2Skp2
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

	ss.InLays = []string{}
	ss.OutLays = []string{}
	ss.HidLays = []string{}
	for _, ly := range net.Layers {
		if ly.IsOff() {
			continue
		}
		switch ly.Type() {
		case emer.Input:
			ss.InLays = append(ss.InLays, ly.Name())
		case emer.Target:
			ss.OutLays = append(ss.OutLays, ly.Name())
			fallthrough
		case emer.Hidden:
			ss.HidLays = append(ss.HidLays, ly.Name())
		}
	}

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
	ss.SetParams("Network", false) // only set Network params
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

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.InitRndSeed()
	ss.StopNow = false
	ss.SetParams("", false) // all sheets
	// note: in general shortening the time constants based on MPI is not useful
	ss.Net.SlowInterval = 100 // 100 > 20
	ss.NewRun()
	ss.UpdateView(true)
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
		ss.MPIWtFmDWt()
	}

	minusCyc := ss.MinusCycles
	plusCyc := ss.PlusCycles

	ss.Net.NewState()
	ss.Time.NewState()
	for cyc := 0; cyc < minusCyc; cyc++ { // do the minus phase
		ss.Net.Cycle(&ss.Time)
		ss.LogTrnCyc(ss.TrnCycLog, ss.Time.Cycle)
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
		ss.LogTrnCyc(ss.TrnCycLog, ss.Time.Cycle)
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

	ss.TrialStats(train)

	if train {
		// not using this anymore, in favor of neuron-level RLrate
		// ss.ErrLrMod.LrateMod(ss.Net, float32(1-ss.TrlCosDiff))
		ss.Net.DWt()
	}

	if viewUpdt == axon.Phase || viewUpdt == axon.AlphaCycle || viewUpdt == axon.ThetaCycle {
		ss.UpdateView(train)
	}

	// include extra off cycles at end
	if ss.PostCycs > 0 {
		ss.Net.InitExt()
		ss.Net.DecayState(ss.PostDecay)
		mxcyc := ss.PostCycs
		for cyc := 0; cyc < mxcyc; cyc++ {
			ss.Net.Cycle(&ss.Time)
			ss.Time.CycleInc()
			if ss.ViewOn {
				ss.UpdateViewTime(train, viewUpdt)
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

	lays := []string{"V1m16", "V1l16", "V1m8", "V1l8", "Output"}
	if ss.TrainEnv.High16 {
		lays = append(lays, "V1h16")
	}
	if ss.TrainEnv.ColorDoG {
		lays = append(lays, []string{"V1Cm16", "V1Cl16", "V1Cm8", "V1Cl8"}...)
	}

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
		ss.LogTrnEpc(ss.TrnEpcLog)
		ss.EpochSched(epc)
		if ss.ViewOn && ss.TrainUpdt > axon.AlphaCycle {
			ss.UpdateView(true)
		}
		if ss.TestInterval > 0 && epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
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
	ss.ApplyInputs(&ss.TrainEnv)
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
	ss.TrnEpcLog.SetNumRows(0)
	ss.TrnTrlLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
	ss.TstTrlLog.SetNumRows(0)
	ss.NeedsNewRun = false
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	// accumulators
	ss.FirstZero = -1
	ss.NZero = 0
	// clear rest just to make Sim look initialized
	ss.TrlErr = 0
	ss.TrlErr2 = 0
	ss.TrlTrgAct = 0
	ss.TrlUnitErr = 0
	ss.EpcUnitErr = 0
	ss.EpcPctErr = 0
	ss.EpcPctErr2 = 0
	ss.EpcCosDiff = 0
	ss.EpcErrTrgAct = 0
	ss.EpcCorTrgAct = 0
	ncat := len(ss.TrainEnv.Images.Cats)
	ss.Confusion.Init(ncat)
	ss.ConfusionEpc = 500
	ss.Confusion.SetLabels(ss.TrainEnv.Images.Cats)
	ss.Confusion.Prob.SetMetaData("font-size", "12")
	// ss.Confusion.Prob.SetMetaData("grid-fill", "1") // not good for tracing rows /cols
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
	ss.TrlCosDiff = float64(out.CosDiff.Cos)
	ss.TrlUnitErr = out.PctUnitErr()

	ovt := ss.ValsTsr("Output")
	out.UnitValsTensor(ovt, "ActM")

	ncats := len(env.Images.Cats)

	ss.TrlRespIdx, ss.TrlErr, ss.TrlErr2 = env.OutErr(ovt)
	if ss.TrlRespIdx >= 0 && ss.TrlRespIdx < ncats {
		ss.TrlResp = env.Images.Cats[ss.TrlRespIdx]
	}
	ss.TrlCatIdx = env.CurCatIdx
	ss.TrlCat = env.CurCat

	epc := env.Epoch.Cur
	if epc > ss.ConfusionEpc {
		ss.Confusion.Incr(ss.TrlCatIdx, ss.TrlRespIdx)
	}

	ss.TrlTrgAct = float64(out.Pools[0].ActP.Avg / 0.01)

	ss.TrlDecRespIdx = ss.Decoder.Decode("ActM")
	if train {
		ss.Decoder.Train(env.CurCatIdx)
	}
	if ss.TrlDecRespIdx == env.CurCatIdx {
		ss.TrlDecErr = 0
	} else {
		ss.TrlDecErr = 1
	}
	ss.TrlDecErr2 = ss.TrlDecErr
	if ss.Decoder.Sorted[1] == env.CurCatIdx {
		ss.TrlDecErr2 = 0
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
		// 	ss.SetParamsSet("WeakShorts", "Network", true)
		// 	mpi.Printf("weaker shortcut cons at epoch: %d\n", epc)
		// case 200: // these have no effect anymore -- with dopamine modulator!
		// ss.Net.LrateSched(0.5)
		// mpi.Printf("dropped lrate to 0.5 at epoch: %d\n", epc)
	case 500:
		ss.SaveWeights()
		// ss.Net.LrateSched(0.2)
		// mpi.Printf("dropped lrate to 0.2 at epoch: %d\n", epc)
		ss.SetParamsSet("ToOutTol", "Network", true) // increase LoTol
		// ss.SetParamsSet("OutAdapt", "Network", true) // increase LoTol
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

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	for {
		ss.TestTrial(true) // return on chg, don't present
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

// ConfusionTstPlot plots the current confusion probability values.
// if cat is empty then it is the diagonal accuracy across all cats
// otherwise it is the confusion row for given category.
// data goes in the TrlErr = Err column.
func (ss *Sim) ConfusionTstPlot(cat string) {
	ss.TstTrlLog.SetNumRows(0)
	nc := ss.Confusion.N.Len()
	ti := -1
	if cat != "" {
		ti = ss.TrainEnv.Images.CatMap[cat]
	}
	for i := 0; i < nc; i++ {
		ss.TestEnv.Trial.Cur = i
		ss.TestEnv.CurCat = ss.TrainEnv.Images.Cats[i]
		if ti >= 0 {
			ss.TrlErr = ss.Confusion.Prob.Value([]int{ti, i})
		} else {
			ss.TrlErr = ss.Confusion.Prob.Value([]int{i, i})
		}
		ss.LogTstTrl(ss.TstTrlLog)
	}
	ss.TstTrlPlot.Params.XAxisCol = "Cat"
	ss.TstTrlPlot.Params.Type = eplot.Bar
	ss.TstTrlPlot.Params.XAxisRot = 45
	ss.TstTrlPlot.GoUpdate()
}

// TestRFs runs test for receptive fields
func (ss *Sim) TestRFs() {
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	ss.ActRFs.Reset()
	for {
		ss.TestTrial(true) // return on chg, don't present
		ss.UpdtActRFs()
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}
	ss.ActRFs.Avg()
	ss.ActRFs.Norm()
	ss.ViewActRFs()
}

// RunTestRFs runs test for receptive fields
func (ss *Sim) RunTestRFs() {
	ss.StopNow = false
	ss.TestRFs()
	ss.Stopped()
}

// UpdtActRFs updates activation rf's -- only called during testing
func (ss *Sim) UpdtActRFs() {
	oly := ss.Net.LayerByName("Output")
	ovt := ss.ValsTsr("Output")
	oly.UnitValsTensor(ovt, "ActM")
	if _, ok := ss.ValsTsrs["Image"]; !ok {
		ss.ValsTsrs["Image"] = &ss.TestEnv.Img.Tsr
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
func (ss *Sim) SetSpikeRastCol(ly *axon.Layer, sr, vl *etensor.Float32, col int) {
	if ly.Is4D() && ly.Nm != "Output" && !strings.HasPrefix(ly.Nm, "TE") {
		nu, sis := ss.CenterPoolsIdxs(ly)
		ti := 0
		for _, si := range sis {
			for ni := 0; ni < nu; ni++ {
				v := vl.Values[si+ni]
				sr.Set([]int{ti, col}, v)
				ti++
			}
		}
	} else {
		for ni, v := range vl.Values {
			sr.Set([]int{ni, col}, v)
		}
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
	ncy := ss.MinusCycles + ss.PlusCycles
	// spike rast
	for _, lnm := range ss.SpikeRastNms {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		sr := ss.SpikeRastTsr(lnm)
		if ly.Is4D() && ly.Nm != "Output" && !strings.HasPrefix(ly.Nm, "TE") {
			nu, sis := ss.CenterPoolsIdxs(ly)
			sr.SetShape([]int{len(sis) * nu, ncy}, nil, []string{"Nrn", "Cyc"})
		} else {
			sr.SetShape([]int{ly.Shp.Len(), ncy}, nil, []string{"Nrn", "Cyc"})
		}
	}
}

// RecSpikes records spikes
func (ss *Sim) RecSpikes(cyc int) {
	for _, lnm := range ss.SpikeRastNms {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		tv := ss.ValsTsr(lnm)
		ly.UnitValsTensor(tv, "Spike")
		sr := ss.SpikeRastTsr(lnm)
		ss.SetSpikeRastCol(ly, sr, tv, cyc)
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
	if ss.Tag != "" {
		return ss.Tag + "_" + ss.ParamsName()
	} else {
		return ss.ParamsName()
	}
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

//////////////////////////////////////////////
//  TrnCycLog

// LogTrnCyc adds data from current trial to the TrnCycLog table.
func (ss *Sim) LogTrnCyc(dt *etable.Table, cyc int) {
	epc := ss.TrainEnv.Epoch.Cur
	trl := ss.TrainEnv.Trial.Cur
	row := cyc

	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellFloat("Cycle", row, float64(cyc))
	dt.SetCellString("Cat", row, ss.TrainEnv.CurCat)
	dt.SetCellString("TrialName", row, ss.TrainEnv.String())

	for _, lnm := range ss.HidLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		dt.SetCellFloat(lnm+"_ActAvg", row, float64(ly.Pools[0].Inhib.Act.Avg))
		dt.SetCellFloat(lnm+"_ActMax", row, float64(ly.Pools[0].Inhib.Act.Max))
	}

	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()
	ovt := ss.ValsTsr("Output")
	out.UnitValsTensor(ovt, "Act")
	dt.SetCellTensor("Output_Acts", row, ovt)

	// note: essential to use Go version of update when called from another goroutine
	if cyc%50 == 0 || cyc == dt.Rows-1 {
		ss.TrnCycPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTrnCycLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnCycLog")
	dt.SetMetaData("desc", "Record of training per cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"Cycle", etensor.INT64, nil, nil},
		{"Cat", etensor.STRING, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
	}
	for _, lnm := range ss.HidLays {
		sch = append(sch, etable.Column{lnm + "_ActAvg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_ActMax", etensor.FLOAT64, nil, nil})
	}

	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()
	sch = append(sch, etable.Column{"Output_Acts", etensor.FLOAT32, out.Shp.Shp, out.Shp.Nms})

	ncy := ss.MinusCycles + ss.PlusCycles
	dt.SetFromSchema(sch, ncy)
}

func (ss *Sim) ConfigTrnCycPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Object Recognition Train Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Cycle", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Cat", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

	for _, lnm := range ss.HidLays {
		plt.SetColParams(lnm+"_ActAvg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.15)
		plt.SetColParams(lnm+"_ActMax", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	}

	return plt
}

//////////////////////////////////////////////
//  TrnTrlLog

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

// FirstActCycle returns the point at which max activity stops going up by more than .01
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
func (ss *Sim) FirstOut(cyclog *etable.Table) (resp int, err float64, err2 float64) {
	fcyc := ss.FirstActStat(cyclog, "Output")
	ss.TrlOutFirstCyc = fcyc
	out := cyclog.CellTensor("Output_Acts", fcyc).(*etensor.Float32)
	resp, err, err2 = ss.TrainEnv.OutErr(out)
	ss.TrlFirstResp = resp
	ss.TrlFirstErr = err
	ss.TrlFirstErr2 = err2
	return
}

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
	dt.SetCellString("Cat", row, ss.TrainEnv.CurCat)
	dt.SetCellString("TrialName", row, ss.TrainEnv.String())

	dt.SetCellString("Resp", row, ss.TrlResp)
	dt.SetCellFloat("Err", row, ss.TrlErr)
	dt.SetCellFloat("Err2", row, ss.TrlErr2)
	dt.SetCellFloat("TrgAct", row, ss.TrlTrgAct)
	dt.SetCellFloat("UnitErr", row, ss.TrlUnitErr)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)
	dt.SetCellFloat("DecResp", row, float64(ss.TrlDecRespIdx))
	dt.SetCellFloat("DecErr", row, ss.TrlDecErr)
	dt.SetCellFloat("DecErr2", row, ss.TrlDecErr2)

	cyclog := ss.TrnCycLog
	ss.FirstOut(cyclog)
	dt.SetCellFloat("FirstErr", row, ss.TrlFirstErr)
	dt.SetCellFloat("FirstErr2", row, ss.TrlFirstErr2)

	for _, lnm := range ss.HidLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		dt.SetCellFloat(lnm+"_MaxGeM", row, float64(ly.Pools[0].GeM.Max))
		dt.SetCellFloat(lnm+"_MaxGiM", row, float64(ly.Pools[0].GiM.Max))
		dt.SetCellFloat(lnm+"_ActDifAvg", row, float64(ly.Pools[0].AvgDif.Avg))
		dt.SetCellFloat(lnm+"_ActDifMax", row, float64(ly.Pools[0].AvgDif.Max))
		dt.SetCellFloat(lnm+"_CosDiff", row, float64(1-ly.CosDiff.Cos))
		fcyc := ss.FirstActStat(cyclog, lnm)
		dt.SetCellFloat(lnm+"_FirstCyc", row, float64(fcyc))
	}

	if ss.TrnTrlFile != nil && (!ss.UseMPI || ss.SaveProcLog) { // otherwise written at end of epoch, integrated
		if ss.TrainEnv.Run.Cur == ss.StartRun && epc == 0 && row == 0 {
			dt.WriteCSVHeaders(ss.TrnTrlFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TrnTrlFile, row, etable.Tab)
	}

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
		{"Resp", etensor.STRING, nil, nil},
		{"Err", etensor.FLOAT64, nil, nil},
		{"Err2", etensor.FLOAT64, nil, nil},
		{"TrgAct", etensor.FLOAT64, nil, nil},
		{"UnitErr", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"DecResp", etensor.INT64, nil, nil},
		{"DecErr", etensor.FLOAT64, nil, nil},
		{"DecErr2", etensor.FLOAT64, nil, nil},
		{"FirstErr", etensor.FLOAT64, nil, nil},
		{"FirstErr2", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.HidLays {
		sch = append(sch, etable.Column{lnm + "_MaxGeM", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_MaxGiM", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_CosDiff", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_ActDifAvg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_ActDifMax", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_FirstCyc", etensor.FLOAT64, nil, nil})
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
	plt.SetColParams("DecErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("DecErr2", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("FirstErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("FirstErr2", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.HidLays {
		plt.SetColParams(lnm+"_MaxGeM", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.5)
		plt.SetColParams(lnm+"_MaxGiM", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.5)
		plt.SetColParams(lnm+"_CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams(lnm+"_ActDifAvg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams(lnm+"_ActDifMax", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams(lnm+"_FirstCyc", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	}
	return plt
}

//////////////////////////////////////////////
//  TrnTrlRepLog

// CenterPoolsIdxs returns the indexes for 2x2 center pools (including sub-pools):
// nu = number of units per pool, sis = starting indexes
func (ss *Sim) CenterPoolsIdxs(ly *axon.Layer) (nu int, sis []int) {
	nu = ly.Shp.Dim(2) * ly.Shp.Dim(3)
	npy := ly.Shp.Dim(0)
	npx := ly.Shp.Dim(1)
	npxact := npx
	nsp := 1
	if ss.SubPools {
		npy /= 2
		npx /= 2
		nsp = 2
	}
	cpy := (npy - 1) / 2
	cpx := (npx - 1) / 2

	for py := 0; py < 2; py++ {
		for px := 0; px < 2; px++ {
			for sy := 0; sy < nsp; sy++ {
				for sx := 0; sx < nsp; sx++ {
					y := (py+cpy)*nsp + sy
					x := (px+cpx)*nsp + sx
					si := (y*npxact + x) * nu
					sis = append(sis, si)
				}
			}
		}
	}
	return
}

// CopyCenterPools copy 2 center pools of ActM to tensor
func (ss *Sim) CopyCenterPools(ly *axon.Layer, vl *etensor.Float32) {
	nu, sis := ss.CenterPoolsIdxs(ly)
	vl.SetShape([]int{len(sis) * nu}, nil, nil)
	ti := 0
	for _, si := range sis {
		for ni := 0; ni < nu; ni++ {
			vl.Values[ti] = ly.Neurons[si+ni].ActM
			ti++
		}
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
	dt.SetCellString("Cat", row, ss.TrainEnv.CurCat)
	dt.SetCellString("TrialName", row, ss.TrainEnv.String())

	dt.SetCellFloat("Err", row, ss.TrlErr)
	dt.SetCellFloat("Err2", row, ss.TrlErr2)
	dt.SetCellFloat("TrgAct", row, ss.TrlTrgAct)
	dt.SetCellFloat("UnitErr", row, ss.TrlUnitErr)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)
	dt.SetCellFloat("DecErr", row, ss.TrlDecErr)
	dt.SetCellFloat("DecErr2", row, ss.TrlDecErr2)

	for _, lnm := range ss.HidLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		lvt := ss.ValsTsr(lnm)
		if ly.Is4D() && ly.Nm != "Output" && !strings.HasPrefix(ly.Nm, "TE") {
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
	// ss.TrnTrlPlot.GoUpdate()
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
		{"DecErr", etensor.FLOAT64, nil, nil},
		{"DecErr2", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.HidLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		if ly.Is4D() && ly.Nm != "Output" && !strings.HasPrefix(ly.Nm, "TE") {
			nu, sis := ss.CenterPoolsIdxs(ly)
			sch = append(sch, etable.Column{lnm, etensor.FLOAT64, []int{len(sis) * nu}, nil})
		} else {
			sch = append(sch, etable.Column{lnm, etensor.FLOAT64, ly.Shp.Shp, nil})
		}
	}
	dt.SetFromSchema(sch, 0)
}

//////////////////////////////////////////////
//  TrnEpcLog

// HogDead computes the proportion of units in given layer name with ActAvg over hog thr
// and under dead threshold
func (ss *Sim) HogDead(lnm string) (hog, dead float64) {
	ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
	n := 0
	if ly.Is4D() {
		npy := ly.Shp.Dim(0)
		npx := ly.Shp.Dim(1)
		nny := ly.Shp.Dim(2)
		nnx := ly.Shp.Dim(3)
		nn := nny * nnx
		if npy == 8 { // exclude periphery
			n = 16 * nn
			for py := 2; py < 6; py++ {
				for px := 2; px < 6; px++ {
					pi := (py*npx + px) * nn
					for ni := 0; ni < nn; ni++ {
						nrn := &ly.Neurons[pi+ni]
						if nrn.ActAvg > 0.3 {
							hog += 1
						} else if nrn.ActAvg < 0.01 {
							dead += 1
						}
					}
				}
			}
		} else if ly.Shp.Dim(0) == 4 && ly.Nm[:2] != "TE" {
			n = 4 * nn
			for py := 1; py < 3; py++ {
				for px := 1; px < 3; px++ {
					pi := (py*npx + px) * nn
					for ni := 0; ni < nn; ni++ {
						nrn := &ly.Neurons[pi+ni]
						if nrn.ActAvg > 0.3 {
							hog += 1
						} else if nrn.ActAvg < 0.01 {
							dead += 1
						}
					}
				}
			}
		}
	}
	if n == 0 {
		n = len(ly.Neurons)
		for ni := range ly.Neurons {
			nrn := &ly.Neurons[ni]
			if nrn.ActAvg > 0.3 {
				hog += 1
			} else if nrn.ActAvg < 0.01 {
				dead += 1
			}
		}
	}
	hog /= float64(n)
	dead /= float64(n)
	return
}

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value

	trl := ss.TrnTrlLog
	if ss.UseMPI {
		empi.GatherTableRows(ss.TrnTrlLogAll, ss.TrnTrlLog, ss.Comm)
		trl = ss.TrnTrlLogAll
	}
	nt := float64(trl.Rows)
	tix := etable.NewIdxView(trl)

	ss.EpcUnitErr = agg.Mean(tix, "UnitErr")[0]
	ss.EpcPctErr = agg.Mean(tix, "Err")[0]
	ss.EpcPctCor = 1 - ss.EpcPctErr
	ss.EpcPctErr2 = agg.Mean(tix, "Err2")[0]
	ss.EpcCosDiff = agg.Mean(tix, "CosDiff")[0]
	ss.EpcPctDecErr = agg.Mean(tix, "DecErr")[0]
	ss.EpcPctDecErr2 = agg.Mean(tix, "DecErr2")[0]

	spl := split.GroupBy(tix, []string{"Err"})
	split.Desc(spl, "TrgAct")
	for _, lnm := range ss.HidLays {
		split.Desc(spl, lnm+"_CosDiff")
		split.Desc(spl, lnm+"_FirstCyc")
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
		if ss.UseMPI {
			empi.GatherTableRows(ss.TrnTrlRepLogAll, ss.TrnTrlRepLog, ss.Comm)
			reps = etable.NewIdxView(ss.TrnTrlRepLogAll)
		}
		reps.SortColName("Cat", true)
		for _, lnm := range ss.HidLays {
			ss.PCA.TableCol(reps, lnm, metric.Covariance64)
			var nstr float64
			ln := len(ss.PCA.Values)
			for i, v := range ss.PCA.Values {
				// fmt.Printf("%s\t\t %d  %g\n", lnm, i, v)
				if v >= 0.01 {
					nstr = float64(ln - i)
					break
				}
			}
			var top5, next5 float64
			for i := 0; i < 5; i++ {
				top5 += ss.PCA.Values[ln-1-i]
				next5 += ss.PCA.Values[ln-6-i]
			}
			sum := norm.Sum64(ss.PCA.Values)
			ravg := (sum - (top5 + next5)) / float64(ln-10)
			dt.SetCellFloat(lnm+"_PCA_NStrong", row, nstr)
			dt.SetCellFloat(lnm+"_PCA_Top5", row, top5/5)
			dt.SetCellFloat(lnm+"_PCA_Next5", row, next5/5)
			dt.SetCellFloat(lnm+"_PCA_Rest", row, ravg)
		}
	} else {
		if row > 0 {
			for _, lnm := range ss.HidLays {
				dt.SetCellFloat(lnm+"_PCA_NStrong", row, dt.CellFloat(lnm+"_PCA_NStrong", row-1))
				dt.SetCellFloat(lnm+"_PCA_Top5", row, dt.CellFloat(lnm+"_PCA_Top5", row-1))
				dt.SetCellFloat(lnm+"_PCA_Next5", row, dt.CellFloat(lnm+"_PCA_Next5", row-1))
				dt.SetCellFloat(lnm+"_PCA_Rest", row, dt.CellFloat(lnm+"_PCA_Rest", row-1))
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
	dt.SetCellFloat("PctDecErr", row, ss.EpcPctDecErr)
	dt.SetCellFloat("PctDecErr2", row, ss.EpcPctDecErr2)
	dt.SetCellFloat("ErrTrgAct", row, ss.EpcErrTrgAct)
	dt.SetCellFloat("CorTrgAct", row, ss.EpcCorTrgAct)
	dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)

	dt.SetCellFloat("PctFirstErr", row, agg.Mean(tix, "FirstErr")[0])
	dt.SetCellFloat("PctFirstErr2", row, agg.Mean(tix, "FirstErr2")[0])

	tst := ss.TstEpcLog
	if tst.Rows > 0 {
		trow := tst.Rows - 1
		dt.SetCellFloat("TstUnitErr", row, tst.CellFloat("UnitErr", trow))
		dt.SetCellFloat("TstPctErr", row, tst.CellFloat("PctErr", trow))
		dt.SetCellFloat("TstPctCor", row, tst.CellFloat("PctCor", trow))
		dt.SetCellFloat("TstPctErr2", row, tst.CellFloat("PctErr2", trow))
		dt.SetCellFloat("TstCosDiff", row, tst.CellFloat("CosDiff", trow))
		dt.SetCellFloat("TstDecErr", row, tst.CellFloat("PctDecErr", trow))
		dt.SetCellFloat("TstDecErr2", row, tst.CellFloat("PctDecErr2", trow))
		dt.SetCellFloat("TstFirstErr", row, tst.CellFloat("PctFirstErr", trow))
		dt.SetCellFloat("TstFirstErr2", row, tst.CellFloat("PctFirstErr2", trow))
	}

	for _, lnm := range ss.HidLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		hog, dead := ss.HogDead(lnm)
		dt.SetCellFloat(lnm+"_Dead", row, dead)
		dt.SetCellFloat(lnm+"_Hog", row, hog)
		ffpj := ly.RecvPrjn(0).(*axon.Prjn)
		dt.SetCellFloat(ly.Nm+"_FF_AvgMaxG", row, float64(ffpj.GScale.AvgMax))
		dt.SetCellFloat(ly.Nm+"_FF_Scale", row, float64(ffpj.GScale.Scale))
		if ly.NRecvPrjns() > 1 {
			fbpj := ly.RecvPrjn(1).(*axon.Prjn)
			dt.SetCellFloat(ly.Nm+"_FB_AvgMaxG", row, float64(fbpj.GScale.AvgMax))
			dt.SetCellFloat(ly.Nm+"_FB_Scale", row, float64(fbpj.GScale.Scale))
		}
		dt.SetCellFloat(lnm+"_MaxGeM", row, float64(ly.ActAvg.AvgMaxGeM))
		dt.SetCellFloat(lnm+"_MaxGiM", row, float64(ly.ActAvg.AvgMaxGiM))
		dt.SetCellFloat(lnm+"_ActAvg", row, float64(ly.ActAvg.ActMAvg))
		dt.SetCellFloat(lnm+"_GiMult", row, float64(ly.ActAvg.GiMult))
		dt.SetCellFloat(lnm+"_ActDifAvg", row, agg.Mean(tix, lnm+"_ActDifAvg")[0])
		dt.SetCellFloat(lnm+"_ActDifMax", row, agg.Max(tix, lnm+"_ActDifMax")[0])
		if ss.EpcPctErr > 0 && ss.EpcPctErr < 1 {
			dt.SetCellFloat(lnm+"_CorCosDiff", row, ss.TrnErrStats.CellFloat(lnm+"_CosDiff:Mean", 0))
			dt.SetCellFloat(lnm+"_ErrCosDiff", row, ss.TrnErrStats.CellFloat(lnm+"_CosDiff:Mean", 1))
			dt.SetCellFloat(lnm+"_CorFirstCyc", row, ss.TrnErrStats.CellFloat(lnm+"_FirstCyc:Mean", 0))
			dt.SetCellFloat(lnm+"_ErrFirstCyc", row, ss.TrnErrStats.CellFloat(lnm+"_FirstCyc:Mean", 1))
		}
	}

	for _, lnm := range ss.InLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		dt.SetCellFloat(lnm+"_ActAvg", row, float64(ly.ActAvg.ActMAvg))
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnEpcPlot.GoUpdate()
	if ss.TrnEpcFile != nil {
		if ss.TrainEnv.Run.Cur == ss.StartRun && row == 0 {
			// note: can't use row=0 b/c reset table each run
			dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab)
		if epc > ss.ConfusionEpc {
			ss.Confusion.Probs()
			fnm := ss.LogFileName("trn_conf")
			ss.Confusion.SaveCSV(gi.FileName(fnm))
		}
	}

	if ss.TrnTrlFile != nil && !(!ss.UseMPI || ss.SaveProcLog) { // saved at trial level otherwise
		if ss.TrainEnv.Run.Cur == ss.StartRun && row == 0 {
			// note: can't just use row=0 b/c reset table each run
			trl.WriteCSVHeaders(ss.TrnTrlFile, etable.Tab)
		}
		for ri := 0; ri < trl.Rows; ri++ {
			trl.WriteCSVRow(ss.TrnTrlFile, ri, etable.Tab)
		}
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
		{"PctDecErr", etensor.FLOAT64, nil, nil},
		{"PctDecErr2", etensor.FLOAT64, nil, nil},
		{"ErrTrgAct", etensor.FLOAT64, nil, nil},
		{"CorTrgAct", etensor.FLOAT64, nil, nil},
		{"PerTrlMSec", etensor.FLOAT64, nil, nil},
		{"PctFirstErr", etensor.FLOAT64, nil, nil},
		{"PctFirstErr2", etensor.FLOAT64, nil, nil},

		{"TstUnitErr", etensor.FLOAT64, nil, nil},
		{"TstPctErr", etensor.FLOAT64, nil, nil},
		{"TstPctCor", etensor.FLOAT64, nil, nil},
		{"TstCosDiff", etensor.FLOAT64, nil, nil},
		{"TstPctErr2", etensor.FLOAT64, nil, nil},
		{"TstDecErr", etensor.FLOAT64, nil, nil},
		{"TstDecErr2", etensor.FLOAT64, nil, nil},
		{"TstFirstErr", etensor.FLOAT64, nil, nil},
		{"TstFirstErr2", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.HidLays {
		sch = append(sch, etable.Column{lnm + "_Dead", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_Hog", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_FF_AvgMaxG", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_FF_Scale", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_FB_AvgMaxG", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_FB_Scale", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_MaxGeM", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_MaxGiM", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_ActAvg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_GiMult", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_ActDifAvg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_ActDifMax", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_CorCosDiff", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_ErrCosDiff", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_CorFirstCyc", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_ErrFirstCyc", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_PCA_NStrong", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_PCA_Top5", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_PCA_Next5", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_PCA_Rest", etensor.FLOAT64, nil, nil})
	}

	for _, lnm := range ss.InLays {
		sch = append(sch, etable.Column{lnm + "_ActAvg", etensor.FLOAT64, nil, nil})
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
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctDecErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("PctDecErr2", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("ErrTrgAct", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CorTrgAct", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PerTrlMSec", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctFirstErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("PctFirstErr2", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)

	plt.SetColParams("TstUnitErr", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TstPctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("TstPctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("TstCosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("TstPctErr2", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("TstDecErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("TstDecErr2", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("TstFirstErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("TstFirstErr2", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.HidLays {
		plt.SetColParams(lnm+"_Dead", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.5)
		plt.SetColParams(lnm+"_Hog", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.5)
		plt.SetColParams(lnm+"_FF_AvgMaxG", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .5)
		plt.SetColParams(lnm+"_FF_Scale", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .5)
		plt.SetColParams(lnm+"_FB_AvgMaxG", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .5)
		plt.SetColParams(lnm+"_FB_Scale", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .5)
		plt.SetColParams(lnm+"_MaxGeM", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_MaxGiM", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_ActAvg", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0.5)
		plt.SetColParams(lnm+"_GiMult", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_ActDifAvg", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_ActDifMax", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_CorCosDiff", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_ErrCosDiff", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_CorFirstCyc", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_ErrFirstCyc", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_PCA_NStrong", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_PCA_Top5", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_PCA_Next5", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_PCA_Rest", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	}

	for _, lnm := range ss.InLays {
		plt.SetColParams(lnm+"_ActAvg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.5)
	}

	return plt
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
	dt.SetCellString("Cat", row, ss.TestEnv.CurCat)
	dt.SetCellString("TrialName", row, ss.TestEnv.String())

	dt.SetCellString("Resp", row, ss.TrlResp)
	dt.SetCellFloat("Err", row, ss.TrlErr)
	dt.SetCellFloat("Err2", row, ss.TrlErr2)
	dt.SetCellFloat("TrgAct", row, ss.TrlTrgAct)
	dt.SetCellFloat("UnitErr", row, ss.TrlUnitErr)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)
	dt.SetCellFloat("DecResp", row, float64(ss.TrlDecRespIdx))
	dt.SetCellFloat("DecErr", row, ss.TrlDecErr)
	dt.SetCellFloat("DecErr2", row, ss.TrlDecErr2)

	cyclog := ss.TrnCycLog
	ss.FirstOut(cyclog)
	dt.SetCellFloat("FirstErr", row, ss.TrlFirstErr)
	dt.SetCellFloat("FirstErr2", row, ss.TrlFirstErr2)

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()

	if ss.TstTrlFile != nil && (!ss.UseMPI || ss.SaveProcLog) { // otherwise written at end of epoch, integrated
		if ss.TrainEnv.Run.Cur == ss.StartRun && ss.TstEpcLog.Rows == 0 && row == 0 {
			dt.WriteCSVHeaders(ss.TstTrlFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TstTrlFile, row, etable.Tab)
	}
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
		{"Cat", etensor.STRING, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Resp", etensor.STRING, nil, nil},
		{"Err", etensor.FLOAT64, nil, nil},
		{"Err2", etensor.FLOAT64, nil, nil},
		{"TrgAct", etensor.FLOAT64, nil, nil},
		{"UnitErr", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"DecResp", etensor.INT64, nil, nil},
		{"DecErr", etensor.FLOAT64, nil, nil},
		{"DecErr2", etensor.FLOAT64, nil, nil},
		{"FirstErr", etensor.FLOAT64, nil, nil},
		{"FirstErr2", etensor.FLOAT64, nil, nil},
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
	plt.SetColParams("Cat", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

	plt.SetColParams("Err", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Err2", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("TrgAct", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("UnitErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("DecErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("DecErr2", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("FirstErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("FirstErr2", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	trl := ss.TstTrlLog
	if ss.UseMPI {
		empi.GatherTableRows(ss.TstTrlLogAll, ss.TstTrlLog, ss.Comm)
		trl = ss.TstTrlLogAll
	}
	tix := etable.NewIdxView(trl)
	epc := ss.TrainEnv.Epoch.Prv // ?

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("UnitErr", row, agg.Sum(tix, "UnitErr")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(tix, "Err")[0])
	dt.SetCellFloat("PctCor", row, 1-agg.Mean(tix, "Err")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(tix, "CosDiff")[0])
	dt.SetCellFloat("PctErr2", row, agg.Mean(tix, "Err2")[0])
	dt.SetCellFloat("PctDecErr", row, agg.Mean(tix, "DecErr")[0])
	dt.SetCellFloat("PctDecErr2", row, agg.Mean(tix, "DecErr2")[0])
	dt.SetCellFloat("PctFirstErr", row, agg.Mean(tix, "FirstErr")[0])
	dt.SetCellFloat("PctFirstErr2", row, agg.Mean(tix, "FirstErr2")[0])

	spl := split.GroupBy(tix, []string{"Cat"})
	_, err := split.AggTry(spl, "Err", agg.AggMean)
	if err != nil {
		log.Println(err)
	}
	objs := spl.AggsToTable(etable.AddAggName)
	no := objs.Rows

	for i := 0; i < no; i++ {
		cat := objs.Cols[0].StringVal1D(i)
		dt.SetCellFloat(cat, row, objs.Cols[1].FloatVal1D(i))
	}

	ss.TstEpcPlot.GoUpdate()
	if ss.TstEpcFile != nil {
		if ss.TrainEnv.Run.Cur == ss.StartRun && row == 0 {
			dt.WriteCSVHeaders(ss.TstEpcFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TstEpcFile, row, etable.Tab)
	}

	if ss.TstTrlFile != nil && !(!ss.UseMPI || ss.SaveProcLog) { // saved at trial level otherwise
		if ss.TrainEnv.Run.Cur == ss.StartRun && row == 0 {
			// note: can't just use row=0 b/c reset table each run
			trl.WriteCSVHeaders(ss.TstTrlFile, etable.Tab)
		}
		for ri := 0; ri < trl.Rows; ri++ {
			trl.WriteCSVRow(ss.TstTrlFile, ri, etable.Tab)
		}
	}
}

func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"UnitErr", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"PctErr2", etensor.FLOAT64, nil, nil},
		{"PctDecErr", etensor.FLOAT64, nil, nil},
		{"PctDecErr2", etensor.FLOAT64, nil, nil},
		{"ErrTrgAct", etensor.FLOAT64, nil, nil},
		{"CorTrgAct", etensor.FLOAT64, nil, nil},
		{"PctFirstErr", etensor.FLOAT64, nil, nil},
		{"PctFirstErr2", etensor.FLOAT64, nil, nil},
	}
	for _, cat := range ss.TestEnv.Images.Cats {
		sch = append(sch, etable.Column{cat, etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Object Recognition Testing Epoch Plot"
	plt.Params.XAxisCol = "Cat"
	plt.Params.Type = eplot.Bar
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("UnitErr", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, cat := range ss.TestEnv.Images.Cats {
		plt.SetColParams(cat, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	}
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
	plt.SetColParams("UnitErr", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	nv.Params.MaxRecs = 300
	// cam := &(nv.Scene().Camera)
	// cam.Pose.Pos.Set(0.0, 1.733, 2.3)
	// cam.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
	// cam.Pose.Quat.SetFromAxisAngle(mat32.Vec3{-1, 0, 0}, 0.4077744)
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("lvis")
	gi.SetAppAbout(`This simulation explores how a hierarchy of areas in the ventral stream of visual processing (up to inferotemporal (IT) cortex) can produce robust object recognition that is invariant to changes in position, size, etc of retinal input images. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch6/lvis/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewMainWindow("lvis", "Object Recognition", width, height)
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

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TrnCycPlot").(*eplot.Plot2D)
	ss.TrnCycPlot = ss.ConfigTrnCycPlot(plt, ss.TrnCycLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

	tg := tv.AddNewTab(etview.KiT_TensorGrid, "Image").(*etview.TensorGrid)
	tg.SetStretchMax()
	tg.SetTensor(&ss.TrainEnv.Img.Tsr)
	ss.CurImgGrid = tg

	stb := tv.AddNewTab(gi.KiT_Layout, "Spike Rasters").(*gi.Layout)
	stb.Lay = gi.LayoutVert
	stb.SetStretchMax()
	for _, lnm := range ss.SpikeRastNms {
		sr := ss.SpikeRastTsr(lnm)
		tg := ss.SpikeRastGrid(lnm)
		tg.SetName(lnm + "Spikes")
		gi.AddNewLabel(stb, lnm, lnm+":")
		stb.AddChild(tg)
		gi.AddNewSpace(stb, lnm+"_spc")
		ss.ConfigSpikeGrid(tg, sr)
	}

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

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

	tbar.AddAction(gi.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAll()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Conf To Test", Icon: "fast-fwd", Tooltip: "Plots accuracy from current confusion probs to test trial log for each category (diagonal of confusion matrix).", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		giv.CallMethod(ss, "ConfusionTstPlot", vp)
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
			gi.OpenURL("https://github.com/lvis/blob/main/sims/lvis_cu3d100_te16deg/README.md")
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
	var note string
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
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
	if ss.ParamSet != "" {
		mpi.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	if saveEpcLog && (ss.SaveProcLog || mpi.WorldRank() == 0) {
		var err error
		fnm := ss.LogFileName("trn_epc")
		ss.TrnEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnEpcFile = nil
		} else {
			mpi.Printf("Saving training epoch log to: %s\n", fnm)
			defer ss.TrnEpcFile.Close()
		}
		fnm = ss.LogFileName("tst_epc")
		ss.TstEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TstEpcFile = nil
		} else {
			mpi.Printf("Saving testing epoch log to: %s\n", fnm)
			defer ss.TstEpcFile.Close()
		}
	}
	if saveTrnTrlLog && (ss.SaveProcLog || mpi.WorldRank() == 0) {
		var err error
		fnm := ss.LogFileName("trn_trl")
		ss.TrnTrlFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnTrlFile = nil
		} else {
			mpi.Printf("Saving train trial log to: %v\n", fnm)
			defer ss.TrnTrlFile.Close()
		}
	}
	if saveTstTrlLog && (ss.SaveProcLog || mpi.WorldRank() == 0) {
		var err error
		fnm := ss.LogFileName("tst_trl")
		ss.TstTrlFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TstTrlFile = nil
		} else {
			mpi.Printf("Saving testing trial log to: %v\n", fnm)
			defer ss.TstTrlFile.Close()
		}
	}
	if saveRunLog && (ss.SaveProcLog || mpi.WorldRank() == 0) {
		var err error
		fnm := ss.LogFileName("run")
		ss.RunFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.RunFile = nil
		} else {
			mpi.Printf("Saving run log to: %s\n", fnm)
			defer ss.RunFile.Close()
		}
	}
	if ss.SaveWts {
		if mpi.WorldRank() != 0 {
			ss.SaveWts = false
		}
		mpi.Printf("Saving final weights per run\n")
	}
	mpi.Printf("Running %d Runs starting at %d\n", ss.MaxRuns, ss.StartRun)
	ss.TrainEnv.Run.Set(ss.StartRun)
	ss.TrainEnv.Run.Max = ss.StartRun + ss.MaxRuns
	ss.Train()
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
	ss.Net.WtFmDWt()
}
