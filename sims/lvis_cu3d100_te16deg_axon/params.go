// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
)

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "needs some special inhibition and learning params",
				Params: params.Params{
					"Layer.Inhib.Pool.FFEx0":          "0.15", // .15 > .18; Ex .05 -- .2/.1, .2/.2, .3/.5 all blow up
					"Layer.Inhib.Pool.FFEx":           "0.0",  // 0 > .05 for trace -- not good
					"Layer.Inhib.Layer.FFEx0":         "0.15",
					"Layer.Inhib.Layer.FFEx":          "0.0",  // 0 > .05 for trace -- not good
					"Layer.Inhib.Layer.Gi":            "1.1",  // 1.1 def, 1.0 for lower layers is best
					"Layer.Inhib.Pool.Gi":             "1.1",  // "
					"Layer.Act.Dend.GbarExp":          "0.2",  // 0.2 > 0.1 > 0
					"Layer.Act.Dend.GbarR":            "3",    // 2 good for 0.2
					"Layer.Act.Dt.VmDendTau":          "5",    // 5 much better in fsa!
					"Layer.Act.NMDA.MgC":              "1.4",  // mg1, voff0, gbarexp.2, gbarr3 = better
					"Layer.Act.NMDA.Voff":             "5",    // mg1, voff0 = mg1.4, voff5 w best params
					"Layer.Act.AK.Gbar":               "1",    // 1 >= 0 > 2
					"Layer.Act.VGCC.Gbar":             "0.02", // non nmda: 0.15 good, 0.3 blows up, nmda: .02 best
					"Layer.Act.VGCC.Ca":               "20",   // 20 / 10tau similar to spk
					"Layer.Learn.CaLrn.Norm":          "80",   // 60 makes CaLrnMax closer to 1
					"Layer.Learn.CaLrn.SpkVGCC":       "true", // sig better..
					"Layer.Learn.CaLrn.SpkVgccCa":     "35",   // 70 / 5 or 35 / 10 both work
					"Layer.Learn.CaLrn.VgccTau":       "10",   // 10 > 5 ?
					"Layer.Learn.CaLrn.Dt.MTau":       "2",    // 2 > 1 ?
					"Layer.Learn.CaSpk.SpikeG":        "12",   // 12 > 8 -- for larger nets
					"Layer.Learn.CaSpk.SynTau":        "30",   // 30 > 20, 40
					"Layer.Learn.CaSpk.Dt.MTau":       "5",    // 5 > 10?
					"Layer.Learn.LrnNMDA.MgC":         "1.4",  // 1.2 for unified Act params, else 1.4
					"Layer.Learn.LrnNMDA.Voff":        "5",    // 0 for unified Act params, else 5
					"Layer.Learn.LrnNMDA.Tau":         "100",  // 100 def
					"Layer.Learn.TrgAvgAct.On":        "true", // critical!
					"Layer.Learn.TrgAvgAct.SubMean":   "1",    // 1 > 0 is important
					"Layer.Learn.RLrate.On":           "true", // beneficial for trace
					"Layer.Learn.RLrate.SigDeriv":     "true",
					"Layer.Learn.RLrate.MidRange.Min": "0.1", // 0.1, 0.9 best
					"Layer.Learn.RLrate.MidRange.Max": "0.9", // 0.1, 0.9 best
					"Layer.Learn.RLrate.NonMid":       "0.05",
					"Layer.Learn.RLrate.Diff":         "true",
					"Layer.Learn.RLrate.ActDiffThr":   "0.02", // 0.02 def - todo
					"Layer.Learn.RLrate.ActThr":       "0.1",  // 0.1 def
					"Layer.Learn.RLrate.Min":          "0.001",
				}},
			{Sel: ".Input", Desc: "all V1 input layers",
				Params: params.Params{
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Layer.Gi":    "0.9",  // 0.9 >= 1.1 def -- more activity -- clamp.Ge more important
					"Layer.Inhib.Pool.Gi":     "0.9",  // 0.9 >= 1.1 def -- more activity
					"Layer.Inhib.ActAvg.Init": "0.06", // .06 for !SepColor actuals: V1m8: .04, V1m16: .03
					"Layer.Act.Clamp.Ge":      "1.0",  // 1.0 > .6 -- more activity
					"Layer.Act.Decay.Act":     "1",    // these make no diff
					"Layer.Act.Decay.Glong":   "1",
				}},
			{Sel: ".V2", Desc: "pool inhib, sparse activity",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.02",
					"Layer.Inhib.Pool.On":     "true", // needs pool-level
					"Layer.Inhib.Layer.Gi":    "1.0",  // 1.0 for lower layers is best
					"Layer.Inhib.Pool.Gi":     "1.0",  // 1.0 > 1.1 -- is sig worse!
					"Layer.Inhib.Topo.On":     "false",
					"Layer.Inhib.Topo.Width":  "4",
					"Layer.Inhib.Topo.Sigma":  "1.0",
					"Layer.Inhib.Topo.Gi":     "0.002", // 0.002 best -- reduces Top5, keeps NStrong
					"Layer.Inhib.Topo.FF0":    "0.2",   // 0.2 best -- test more
				}},
			// {Sel: ".V2m", Desc: "pool inhib, sparse activity",
			// 	Params: params.Params{
			// 		"Layer.Inhib.ActAvg.Init": "0.02",
			// 	}},
			{Sel: ".V2l", Desc: "pool inhib, sparse activity",
				Params: params.Params{
					"Layer.Inhib.Topo.Width": "2", // smaller
				}},
			// {Sel: "#V2l16", Desc: "this layer is too active, drives V4f16 too strongly",
			// 	Params: params.Params{
			// 		"Layer.Inhib.ActAvg.Init": "0.02", // not clear if needed now..
			// 	}},
			// {Sel: ".V2h", Desc: "pool inhib, sparse activity",
			// 	Params: params.Params{
			// 		"Layer.Inhib.ActAvg.Init": "0.02",
			// 	}},
			{Sel: ".V3h", Desc: "pool inhib, sparse activity -- only for h16",
				Params: params.Params{
					"Layer.Inhib.Pool.On":     "true", // needs pool-level
					"Layer.Inhib.ActAvg.Init": "0.02", // .02 > .04
				}},
			{Sel: ".V4", Desc: "pool inhib, sparse activity",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.04", // .04 >= .03 > .05
					"Layer.Inhib.Pool.On":     "true", // needs pool-level
					"Layer.Inhib.Layer.Gi":    "1.0",  // 1.0 maybe > 1.1
					"Layer.Inhib.Pool.Gi":     "1.1",  // 1.1 > 1.0
					"Layer.Inhib.Topo.On":     "false",
					"Layer.Inhib.Topo.Width":  "4", // was 4
					"Layer.Inhib.Topo.Sigma":  "1.0",
					"Layer.Inhib.Topo.Gi":     "0.002", // 0.002 best -- reduces Top5, keeps NStrong
					"Layer.Inhib.Topo.FF0":    "0.2",   // 0.2 best -- test more
				}},
			{Sel: ".TEO", Desc: "initial activity",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.06",  // .06 > .05 = .04
					"Layer.Inhib.Pool.On":     "true",  // needs pool-level
					"Layer.Inhib.Layer.On":    "false", // no layer!
					"Layer.Inhib.Layer.Gi":    "1.1",   // 1.1 def
					"Layer.Inhib.Pool.Gi":     "1.1",   // 1.1 def
				}},
			{Sel: "#TE", Desc: "initial activity",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.04",  // .03 actual with gi 1.2, was .06
					"Layer.Inhib.Pool.On":     "true",  // needs pool-level
					"Layer.Inhib.Layer.On":    "false", // no layer!
					"Layer.Inhib.Layer.Gi":    "1.1",   // 1.1 def
					"Layer.Inhib.Pool.Gi":     "1.1",   // 1.1 def
				}},
			{Sel: "#Output", Desc: "general output, Localist default -- see RndOutPats, LocalOutPats",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":          "1.2",   // 1.3 adapt > fixed: 1.2, 1.23 too low, 1.25, 1.3 too high
					"Layer.Inhib.ActAvg.Init":       "0.005", // .005 > .008 > .01 -- prevents loss of Ge over time..
					"Layer.Inhib.ActAvg.Targ":       "0.01",  // .01 > 0.011 > 0.012 > 0.009
					"Layer.Inhib.ActAvg.AdaptGi":    "true",  // true: it is essential -- too hard to balance manually
					"Layer.Inhib.ActAvg.LoTol":      "0.1",   // 0.1 > 0.05 > 0.2 > 0.5
					"Layer.Inhib.ActAvg.HiTol":      "0.2",   // 0.1 > 0 def
					"Layer.Inhib.ActAvg.AdaptRate":  "0.02",  // 0.02 >= 0.01 -- 0.005 worse, tol 0.1
					"Layer.Act.Clamp.Ge":            "0.6",   // .6 = .7 > .5 (tiny diff) -- input has 1.0 now
					"Layer.Learn.CaSpk.SpikeG":      "12",    // 12 > 8 probably; 8 = orig, 12 = new trace
					"Layer.Inhib.Pool.FFEx":         "0.0",   // no
					"Layer.Inhib.Layer.FFEx":        "0.0",   //
					"Layer.Learn.RLrate.On":         "true",  // beneficial for trace
					"Layer.Learn.RLrate.NonMid":     "1",     // 1 > lower for output
					"Layer.Learn.RLrate.Diff":       "true",
					"Layer.Learn.RLrate.ActDiffThr": "0.02", // 0.02 def - todo
					"Layer.Learn.RLrate.ActThr":     "0.1",  // 0.1 def
					"Layer.Learn.RLrate.Min":        "0.001",
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
					"Prjn.SWt.Adapt.On":           "true",   // true > false, esp in cosdiff
					"Prjn.SWt.Adapt.Lrate":        "0.0002", // .0002, .001 > .01 > .1 after 250epc in NStrong
					"Prjn.SWt.Adapt.DreamVar":     "0.0",    // 0.02 good overall, no ToOut
					"Prjn.SWt.Adapt.SubMean":      "1",
					"Prjn.Learn.Lrate.Base":       "0.01", // 0.01 > 0.02 later (trace)
					"Prjn.Com.PFail":              "0.0",
					"Prjn.Learn.Trace.SubMean":    "0",  // 0-1 makes no diff, at least early on!
					"Prjn.Learn.KinaseCa.SpikeG":  "12", // 12 matches theta exactly, higher dwtavg but ok
					"Prjn.Learn.KinaseCa.Dt.MTau": "5",  // 5 > 10 test more
					"Prjn.Learn.KinaseCa.Dt.PTau": "40",
					"Prjn.Learn.KinaseCa.Dt.DTau": "40",
					"Prjn.Learn.KinaseCa.UpdtThr": "0.01", // 0.01 > 0.05 -- was LrnThr
					"Prjn.Learn.KinaseCa.MaxISI":  "100",  // 100 >= 50 -- not much diff, no sig speed diff with 50
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
					"Prjn.PrjnScale.Abs":      "2.0",   // 2.0 >= 1.8 > 2.2 > 1.5 > 1.2 trace
				}},
			{Sel: ".FmOut", Desc: "from output -- some things should be different..",
				Params: params.Params{}},
			{Sel: ".Inhib", Desc: "inhibitory projection",
				Params: params.Params{
					"Prjn.Learn.Learn":         "true",   // learned decorrel is good
					"Prjn.Learn.Lrate.Base":    "0.0001", // .0001 > .001 -- slower better!
					"Prjn.Learn.Trace.SubMean": "1",      // 1 is *essential* here!
					"Prjn.SWt.Init.Var":        "0.0",
					"Prjn.SWt.Init.Mean":       "0.1",
					"Prjn.SWt.Init.Sym":        "false",
					"Prjn.SWt.Adapt.On":        "false",
					"Prjn.PrjnScale.Abs":       "0.2", // .2 > .1 for controlling PCA; .3 or.4 with GiSynThr .01
					"Prjn.IncGain":             "1",   // .5 def
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

// Prjns holds all the special projections
type Prjns struct {
	Prjn4x4Skp2              *prjn.PoolTile    `desc:"Standard feedforward topographic projection, recv = 1/2 send size"`
	Prjn4x4Skp2Recip         *prjn.PoolTile    `desc:"Reciprocal"`
	Prjn4x4Skp2Sub2          *prjn.PoolTileSub `desc:"Standard feedforward topographic projection, recv = 1/2 send size"`
	Prjn4x4Skp2Sub2Recip     *prjn.PoolTileSub `desc:"Reciprocal"`
	Prjn4x4Skp2Sub2Send      *prjn.PoolTileSub `desc:"Standard feedforward topographic projection, recv = 1/2 send size"`
	Prjn4x4Skp2Sub2SendRecip *prjn.PoolTileSub `desc:"Standard feedforward topographic projection, recv = 1/2 send size"`
	Prjn2x2Skp1              *prjn.PoolTile    `desc:"same-size prjn"`
	Prjn2x2Skp1Recip         *prjn.PoolTile    `desc:"same-size prjn reciprocal"`
	Prjn2x2Skp1Sub2          *prjn.PoolTileSub `desc:"same-size prjn"`
	Prjn2x2Skp1Sub2Recip     *prjn.PoolTileSub `desc:"same-size prjn reciprocal"`
	Prjn2x2Skp1Sub2Send      *prjn.PoolTileSub `desc:"same-size prjn"`
	Prjn2x2Skp1Sub2SendRecip *prjn.PoolTileSub `desc:"same-size prjn reciprocal"`
	Prjn2x2Skp2              *prjn.PoolTileSub `desc:"lateral inhib projection"`
	Prjn4x4Skp0              *prjn.PoolTile    `desc:"for V4 <-> TEO"`
	Prjn4x4Skp0Recip         *prjn.PoolTile    `desc:"for V4 <-> TEO"`
	Prjn4x4Skp0Sub2          *prjn.PoolTileSub `desc:"for V4 <-> TEO"`
	Prjn4x4Skp0Sub2Recip     *prjn.PoolTileSub `desc:"for V4 <-> TEO"`
	Prjn1x1Skp0              *prjn.PoolTile    `desc:"for TE <-> TEO"`
	Prjn1x1Skp0Recip         *prjn.PoolTile    `desc:"for TE <-> TEO"`
	Prjn6x6Skp2Lat           *prjn.PoolTileSub `desc:"lateral inhibitory connectivity for subpools"`
}

func (pj *Prjns) New() {
	pj.Prjn4x4Skp2 = prjn.NewPoolTile()
	pj.Prjn4x4Skp2.Size.Set(4, 4)
	pj.Prjn4x4Skp2.Skip.Set(2, 2)
	pj.Prjn4x4Skp2.Start.Set(-1, -1)
	pj.Prjn4x4Skp2.TopoRange.Min = 0.8
	pj.Prjn4x4Skp2Recip = prjn.NewPoolTileRecip(pj.Prjn4x4Skp2)

	pj.Prjn4x4Skp2Sub2 = prjn.NewPoolTileSub()
	pj.Prjn4x4Skp2Sub2.Size.Set(4, 4)
	pj.Prjn4x4Skp2Sub2.Skip.Set(2, 2)
	pj.Prjn4x4Skp2Sub2.Start.Set(-1, -1)
	pj.Prjn4x4Skp2Sub2.Subs.Set(2, 2)
	pj.Prjn4x4Skp2Sub2.TopoRange.Min = 0.8
	pj.Prjn4x4Skp2Sub2Recip = prjn.NewPoolTileSubRecip(pj.Prjn4x4Skp2Sub2)

	pj.Prjn4x4Skp2Sub2Send = prjn.NewPoolTileSub()
	*pj.Prjn4x4Skp2Sub2Send = *pj.Prjn4x4Skp2Sub2
	pj.Prjn4x4Skp2Sub2Send.SendSubs = true
	pj.Prjn4x4Skp2Sub2SendRecip = prjn.NewPoolTileSubRecip(pj.Prjn4x4Skp2Sub2Send)

	pj.Prjn2x2Skp1 = prjn.NewPoolTile()
	pj.Prjn2x2Skp1.Size.Set(2, 2)
	pj.Prjn2x2Skp1.Skip.Set(1, 1)
	pj.Prjn2x2Skp1.Start.Set(0, 0)
	pj.Prjn2x2Skp1.TopoRange.Min = 0.8
	pj.Prjn2x2Skp1Recip = prjn.NewPoolTileRecip(pj.Prjn2x2Skp1)

	pj.Prjn2x2Skp1Sub2 = prjn.NewPoolTileSub()
	pj.Prjn2x2Skp1Sub2.Size.Set(2, 2)
	pj.Prjn2x2Skp1Sub2.Skip.Set(1, 1)
	pj.Prjn2x2Skp1Sub2.Start.Set(0, 0)
	pj.Prjn2x2Skp1Sub2.Subs.Set(2, 2)
	pj.Prjn2x2Skp1Sub2.TopoRange.Min = 0.8

	pj.Prjn2x2Skp1Sub2Recip = prjn.NewPoolTileSubRecip(pj.Prjn2x2Skp1Sub2)

	pj.Prjn2x2Skp1Sub2Send = prjn.NewPoolTileSub()
	pj.Prjn2x2Skp1Sub2Send.Size.Set(2, 2)
	pj.Prjn2x2Skp1Sub2Send.Skip.Set(1, 1)
	pj.Prjn2x2Skp1Sub2Send.Start.Set(0, 0)
	pj.Prjn2x2Skp1Sub2Send.Subs.Set(2, 2)
	pj.Prjn2x2Skp1Sub2Send.SendSubs = true
	pj.Prjn2x2Skp1Sub2Send.TopoRange.Min = 0.8

	pj.Prjn2x2Skp1Sub2SendRecip = prjn.NewPoolTileSub()
	*pj.Prjn2x2Skp1Sub2SendRecip = *pj.Prjn2x2Skp1Sub2Send
	pj.Prjn2x2Skp1Sub2SendRecip.Recip = true

	pj.Prjn2x2Skp2 = prjn.NewPoolTileSub()
	pj.Prjn2x2Skp2.Size.Set(2, 2)
	pj.Prjn2x2Skp2.Skip.Set(2, 2)
	pj.Prjn2x2Skp2.Start.Set(0, 0)
	pj.Prjn2x2Skp2.Subs.Set(2, 2)

	pj.Prjn4x4Skp0 = prjn.NewPoolTile()
	pj.Prjn4x4Skp0.Size.Set(4, 4)
	pj.Prjn4x4Skp0.Skip.Set(0, 0)
	pj.Prjn4x4Skp0.Start.Set(0, 0)
	pj.Prjn4x4Skp0.GaussFull.Sigma = 1.5
	pj.Prjn4x4Skp0.GaussInPool.Sigma = 1.5
	pj.Prjn4x4Skp0.TopoRange.Min = 0.8
	pj.Prjn4x4Skp0Recip = prjn.NewPoolTileRecip(pj.Prjn4x4Skp0)

	pj.Prjn4x4Skp0Sub2 = prjn.NewPoolTileSub()
	pj.Prjn4x4Skp0Sub2.Size.Set(4, 4)
	pj.Prjn4x4Skp0Sub2.Skip.Set(0, 0)
	pj.Prjn4x4Skp0Sub2.Start.Set(0, 0)
	pj.Prjn4x4Skp0Sub2.Subs.Set(2, 2)
	pj.Prjn4x4Skp0Sub2.SendSubs = true
	pj.Prjn4x4Skp0Sub2.GaussFull.Sigma = 1.5
	pj.Prjn4x4Skp0Sub2.GaussInPool.Sigma = 1.5
	pj.Prjn4x4Skp0Sub2.TopoRange.Min = 0.8
	pj.Prjn4x4Skp0Sub2Recip = prjn.NewPoolTileSubRecip(pj.Prjn4x4Skp0Sub2)

	pj.Prjn1x1Skp0 = prjn.NewPoolTile()
	pj.Prjn1x1Skp0.Size.Set(1, 1)
	pj.Prjn1x1Skp0.Skip.Set(0, 0)
	pj.Prjn1x1Skp0.Start.Set(0, 0)
	pj.Prjn1x1Skp0.GaussFull.Sigma = 1.5
	pj.Prjn1x1Skp0.GaussInPool.Sigma = 1.5
	pj.Prjn1x1Skp0.TopoRange.Min = 0.8
	pj.Prjn1x1Skp0Recip = prjn.NewPoolTileRecip(pj.Prjn1x1Skp0)

	pj.Prjn6x6Skp2Lat = prjn.NewPoolTileSub()
	pj.Prjn6x6Skp2Lat.Size.Set(6, 6)
	pj.Prjn6x6Skp2Lat.Skip.Set(2, 2)
	pj.Prjn6x6Skp2Lat.Start.Set(-2, -2)
	pj.Prjn6x6Skp2Lat.Subs.Set(2, 2)
	pj.Prjn6x6Skp2Lat.TopoRange.Min = 0.8
}
