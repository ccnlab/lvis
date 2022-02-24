// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/params"

// ParamSetsAll has all the parameters explored up through 1/2022
var ParamSetsAll = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "needs some special inhibition and learning params",
				Params: params.Params{
					"Layer.Inhib.Inhib.AvgTau":           "30",   // 30 > 20 >> 1 definitively
					"Layer.Inhib.Inhib.GiSynThr":         "0.0",  // 0.01 shows effects
					"Layer.Inhib.Layer.Gi":               "1.1",  // 1.1 > 1.0 > 1.2 -- all layers
					"Layer.Inhib.Pool.Gi":                "1.1",  // 1.1 > 1.0 -- universal for all layers
					"Layer.Inhib.Pool.FFEx0":             "0.15", // .15 > .18; Ex .05 -- .2/.1, .2/.2, .3/.5 all blow up
					"Layer.Inhib.Pool.FFEx":              "0.05", // .05 best so far
					"Layer.Inhib.Layer.FFEx0":            "0.15",
					"Layer.Inhib.Layer.FFEx":             "0.05", // .05 best so far
					"Layer.Inhib.Layer.Bg":               "0.0",  // .2 worse
					"Layer.Inhib.Pool.Bg":                "0.0",  // "
					"Layer.Act.Dend.GbarExp":             "0.2",  // 0.2 > 0.1 > 0
					"Layer.Act.Dend.GbarR":               "3",    // 2 good for 0.2
					"Layer.Act.Dt.VmDendTau":             "2.81", // 5 vs. 2.81? test..
					"Layer.Act.Dt.IntTau":                "40",   // 40 > 20
					"Layer.Act.Gbar.L":                   "0.2",  // 0.2 orig > 0.1 new def
					"Layer.Act.Decay.Act":                "0.2",  // 0.2 > 0 > 0.5 w/ glong.7 459
					"Layer.Act.Decay.Glong":              "0.6",  // 0.6 > 0.7 > 0.8
					"Layer.Act.KNa.Fast.Max":             "0.1",  // fm both .2 worse
					"Layer.Act.KNa.Med.Max":              "0.2",  // 0.2 > 0.1 def
					"Layer.Act.KNa.Slow.Max":             "0.2",  // 0.2 > higher
					"Layer.Act.Noise.On":                 "false",
					"Layer.Act.Noise.Ge":                 "0.005", // 0.002 has sig effects..
					"Layer.Act.Noise.Gi":                 "0.0",
					"Layer.Act.GTarg.GeMax":              "1.2",  // 1 > .8 -- rescaling not very useful.
					"Layer.Act.Dt.LongAvgTau":            "20",   // 50 > 20 in terms of stability, but weird effect late
					"Layer.Learn.ActAvg.MinLrn":          "0.02", // sig improves "top5" hogging in pca strength
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
					"Layer.Inhib.Layer.Gi":    "0.9",  // 0.9 >= 1.1 def -- more activity -- clamp.Ge more important
					"Layer.Inhib.Pool.Gi":     "0.9",  // 0.9 >= 1.1 def -- more activity
					"Layer.Inhib.ActAvg.Init": "0.06", // .06 for !SepColor actuals: V1m8: .04, V1m16: .03
					"Layer.Act.Clamp.Ge":      "1.0",  // 1.0 > .6 -- more activity
					"Layer.Act.Decay.Act":     "1",    // these make no diff
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
					"Layer.Act.Clamp.Ge": "0.6", // .6 = .7 > .5 (tiny diff) -- input has 1.0 now
					// "Layer.Act.Spike.Tr":       "3",     // 2 >= 3 > 1 > 0
					// "Layer.Act.GABAB.Gbar":   "0.005", // .005 > .01 > .02 > .05 > .1 > .2
					// "Layer.Act.NMDA.Gbar":    "0.03",  // was .02
					"Layer.Learn.RLrate.On":  "true", // todo: try false
					"Layer.Inhib.Pool.FFEx":  "0.0",  // no
					"Layer.Inhib.Layer.FFEx": "0.0",  //
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
					"Prjn.Learn.Lrate.Base":     "0.02",   // 0.02 std in initial NeurSpk models
					"Prjn.Learn.XCal.SubMean":   "1",      // testing..
					"Prjn.Learn.XCal.DWtThr":    "0.0001", // 0.0001 > 0.001
					"Prjn.Com.PFail":            "0.0",
					"Prjn.Learn.Kinase.On":      "true",
					"Prjn.Learn.Kinase.SAvgThr": "0.02", // 0.02 = 0.01 > 0.05
					"Prjn.Learn.Kinase.MTau":    "40",
					"Prjn.Learn.Kinase.PTau":    "10",
					"Prjn.Learn.Kinase.DTau":    "40",
					"Prjn.Learn.Kinase.DScale":  "0.95", // 0.93 > 0.94 > 1 > .9
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
