package main

import "github.com/emer/emergent/params"

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
					"Layer.Act.Dend.GbarExp":        "0.2",  // 0.2 > 0.5 > 0.1 > 0
					"Layer.Act.Dend.GbarR":          "3",    // 3 > 6 > 2 good for 0.2 -- too low rel to ExpGbar causes fast ini learning, but then unravels
					"Layer.Act.Dt.GeTau":            "5",    // 5 = 4 (bit slower) > 6 > 7 @176
					"Layer.Act.Dt.LongAvgTau":       "20",   // 20 > 50 > 100
					"Layer.Act.Dt.VmDendTau":        "5",    // 5 much better in fsa!
					"Layer.Act.NMDA.MgC":            "1.4",  // mg1.2 alt
					"Layer.Act.NMDA.Voff":           "5",    // 0 alt
					"Layer.Act.VGCC.Gbar":           "0.02", // non nmda: 0.15 good, 0.3 blows up
					"Layer.Act.AK.Gbar":             "1",    // 1 == .1 trace-v8
					"Layer.Learn.NeurCa.SpkVGCC":    "true", // sig better..
					"Layer.Learn.NeurCa.MTauCaLrn":  "false",
					"Layer.Learn.NeurCa.SpkVGCCa":   "30", // 180 = equivalent of 1200 from v7; ~30 matches in !mtau
					"Layer.Learn.NeurCa.SpikeG":     "12", // 12 > 8 def major
					"Layer.Learn.NeurCa.CaMax":      "80", // 38 = 250 from v7; 65 matches in !mtau, but 80 works better
					"Layer.Learn.NeurCa.SynTau":     "30", // 30 best on lvis
					"Layer.Learn.NeurCa.MTau":       "5",  // 40, 10 same as 10, 40 for Neur
					"Layer.Learn.NeurCa.PTau":       "40",
					"Layer.Learn.NeurCa.DTau":       "40",
					"Layer.Learn.NeurCa.Decay":      "false",
					"Layer.Learn.NeurCa.DecayCaLrn": "true",
					"Layer.Learn.LrnNMDA.MgC":       "1.4", // 1.2 for unified Act params, else 1.4
					"Layer.Learn.LrnNMDA.Voff":      "5",   // 0 for unified Act params, else 5
					"Layer.Learn.LrnNMDA.Tau":       "100", // 100 else 50
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
					"Layer.Inhib.ActAvg.Init": "0.04", // .04 def -- .03 more accurate
				}},
			{Sel: "#IT", Desc: "initial activity",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1.0",  // 1.1 > 1.0, 1.2
					"Layer.Inhib.ActAvg.Init": "0.04", // .05 > .04 with adapt
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
					"Prjn.Learn.Lrate.Base":       "0.2",   // 0.1 nominal
					"Prjn.SWt.Adapt.Lrate":        "0.005", // 0.005 > others maybe?  0.02 > 0.05 > .1
					"Prjn.SWt.Init.SPct":          "1",     // 1 >= lower
					"Prjn.Com.PFail":              "0.0",
					"Prjn.Learn.KinaseCa.SpikeG":  "12", // 12 def / ra25
					"Prjn.Learn.KinaseCa.MTau":    "5",  // 5 > 10 test more
					"Prjn.Learn.KinaseCa.PTau":    "40",
					"Prjn.Learn.KinaseCa.DTau":    "40",
					"Prjn.Learn.KinaseCa.UpdtThr": "0.01", // 0.01 > 0.02 max tolerable
					"Prjn.Learn.KinaseCa.Decay":   "true",
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
