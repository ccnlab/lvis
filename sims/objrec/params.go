package main

import "github.com/emer/emergent/params"

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "needs some special inhibition and learning params",
				Params: params.Params{
					"Layer.Inhib.Layer.FS0":         "0.1", // .1 -- highly sensitive .08, .12 both sig worse
					"Layer.Inhib.Pool.FS0":          "0.1", // .1
					"Layer.Inhib.Layer.FSTau":       "6",   // 6 best
					"Layer.Inhib.Layer.FB":          "1",   // def = 1 for most layers
					"Layer.Inhib.Pool.FB":           "4",
					"Layer.Inhib.Layer.SSfTau":      "20",   // 20 > 30  > 15
					"Layer.Inhib.Layer.SSiTau":      "50",   // 50 > 40 -- try 40, 60 @ gi= 1.1?
					"Layer.Act.Dt.IntTau":           "40",   // 30 == 40 no diff
					"Layer.Act.Decay.Act":           "0.0",  // 0 > .2 -- highly sensitive
					"Layer.Act.Decay.Glong":         "0.6",  // 0.6 def > 0.5, .7 -- highly sensitive
					"Layer.Act.Decay.AHP":           "0.0",  // 0 def
					"Layer.Act.Dend.SSGi":           "3",    // 3.0o0 best
					"Layer.Act.Dend.GbarExp":        "0.2",  // 0.2 > 0.5 > 0.1 > 0
					"Layer.Act.Dend.GbarR":          "3",    // 3 > 6 > 2 good for 0.2 -- too low rel to ExpGbar causes fast ini learning, but then unravels
					"Layer.Act.Dt.GeTau":            "5",    // 5 = 4 (bit slower) > 6 > 7 @176
					"Layer.Act.Dt.LongAvgTau":       "20",   // 20 > 50 > 100
					"Layer.Act.Dt.VmDendTau":        "5",    // 5 much better in fsa!
					"Layer.Act.NMDA.MgC":            "1.4",  // 1.4, 5 > 1.2, 0
					"Layer.Act.NMDA.Voff":           "5",    // see above
					"Layer.Act.NMDA.Gbar":           "0.15", // .15 > .2 (blows up -- needs more inhib)
					"Layer.Act.GABAB.Gbar":          "0.2",  // .2
					"Layer.Act.AK.Gbar":             "0.1",  // 1 == .1 trace-v8
					"Layer.Act.VGCC.Gbar":           "0.02", // non nmda: 0.15 good, 0.3 blows up
					"Layer.Act.VGCC.Ca":             "25",   // 25 / 10tau best performance
					"Layer.Act.Mahp.Gbar":           "0.02", // .02 > .05 -- 0.01 best in lvis
					"Layer.Learn.CaLrn.Norm":        "80",   // 80 default
					"Layer.Learn.CaLrn.SpkVGCC":     "true", // sig better?
					"Layer.Learn.CaLrn.SpkVgccCa":   "35",   // 35 > 40, 45
					"Layer.Learn.CaLrn.VgccTau":     "10",   // 10 > 5 ?
					"Layer.Learn.CaLrn.Dt.MTau":     "2",    // 2 > 1 ?
					"Layer.Learn.CaSpk.SpikeG":      "12",   // 12 > 8 > 15 (too high) -- 12 makes everything work!
					"Layer.Learn.CaSpk.SynTau":      "30",   // 30 > 20, 40
					"Layer.Learn.CaSpk.Dt.MTau":     "5",    // 5 == 10 -- no big diff
					"Layer.Learn.LrnNMDA.MgC":       "1.4",  // 1.4, 5 > 1.2, 0
					"Layer.Learn.LrnNMDA.Voff":      "5",    // see above
					"Layer.Learn.LrnNMDA.Tau":       "100",  // 100 def
					"Layer.Learn.TrgAvgAct.On":      "true", // critical!
					"Layer.Learn.TrgAvgAct.SubMean": "1.0",  // 1 > 0 -- key!
					"Layer.Learn.RLrate.On":         "true", // beneficial for trace
					"Layer.Learn.RLrate.SigmoidMin": "0.05",
					"Layer.Learn.RLrate.Diff":       "true", // always key
					"Layer.Learn.RLrate.DiffThr":    "0.02", // 0.02 def - todo
					"Layer.Learn.RLrate.SpkThr":     "0.1",  // 0.1 def
					"Layer.Learn.RLrate.Min":        "0.001",
				}},
			{Sel: "#V1", Desc: "pool inhib (not used), initial activity",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.08", // 0.08 == 0.9 just noisier
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Layer.Gi":    "0.9", // 0.9 def
					"Layer.Inhib.Pool.Gi":     "0.9", // 0.9 def
					"Layer.Inhib.Layer.FB":    "1",
					"Layer.Inhib.Pool.FB":     "1",
					"Layer.Inhib.Layer.SS":    "30",
					"Layer.Inhib.Pool.SS":     "30",
					"Layer.Act.Clamp.Ge":      "1.5", // 1.5 for fsffffb
					"Layer.Act.Decay.Act":     "1",   // 1 = slightly beneficial
					"Layer.Act.Decay.Glong":   "1",
				}},
			{Sel: "#V4", Desc: "pool inhib, sparse activity",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.03", // 0.03 > .04
					"Layer.Inhib.Layer.FB":    "1",    // 1.1 FB1 >> 4!
					"Layer.Inhib.Pool.FB":     "4",    // 0.9 FB4 > others
					"Layer.Inhib.Layer.SS":    "30",   // 30 best
					"Layer.Inhib.Pool.SS":     "30",   // 0 works here..
					"Layer.Inhib.Layer.Gi":    "1.0",  // 1.1 > 1.0 -- def 1.1, 1.0 > 1.0, 1.1!
					"Layer.Inhib.Pool.Gi":     "0.9",  // 0.9 FB4 > 1.0 FB1
					"Layer.Inhib.Pool.On":     "true", // needs pool-level
				}},
			{Sel: "#IT", Desc: "initial activity",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.04", // 0.05 -- 0.05 actual
					"Layer.Inhib.Layer.Gi":    "1.0",  // 1.1 FB1 > 1.0 FB4
					"Layer.Inhib.Layer.FB":    "4",    // 1.1 FB1 > 1.0 FB4
				}},
			{Sel: "#Output", Desc: "high inhib for one-hot output",
				Params: params.Params{
					// "Layer.Act.Decay.Act":     "0.0",  // 0.2 with glong .6 best in lvis, slows learning here
					// "Layer.Act.Decay.Glong":   "0.6",  // 0.6 def
					"Layer.Inhib.ActAvg.Init": "0.05", // 0.05 nominal
					"Layer.Inhib.Layer.Gi":    "1.2",  // 1.2 FB1 > 1.1 FB4
					"Layer.Inhib.Layer.FB":    "1",    //
					"Layer.Inhib.Layer.SS":    "30",   // 0 ok here -- not better in the end
					// "Layer.Inhib.ActAvg.Targ":       "0.04",  //
					// "Layer.Inhib.ActAvg.AdaptGi":    "false", //
					// "Layer.Inhib.ActAvg.LoTol":      "0.1",   //
					// "Layer.Inhib.ActAvg.HiTol":      "0.02",  //
					// "Layer.Inhib.ActAvg.AdaptRate":  "0.01",  //
					"Layer.Act.Clamp.Ge":            "0.8",  // 0.8 > 1.0 > 0.6 1.6.4
					"Layer.Learn.CaSpk.SpikeG":      "12",   // 12 > 8 -- not a big diff
					"Layer.Learn.RLrate.On":         "true", // beneficial for trace
					"Layer.Learn.RLrate.SigmoidMin": "0.05", // 0.05 > 1 -- key!
					"Layer.Learn.RLrate.Diff":       "true",
					"Layer.Learn.RLrate.DiffThr":    "0.02", // 0.02 def - todo
					"Layer.Learn.RLrate.SpkThr":     "0.1",  // 0.1 def
					"Layer.Learn.RLrate.Min":        "0.001",
				}},
			{Sel: "Prjn", Desc: "yes extra learning factors",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":       "0.2",    // 0.4 for NeuronCa; 0.2 best, 0.1 nominal
					"Prjn.Learn.Trace.NeuronCa":   "false",  // false = sig better (SynCa)
					"Prjn.Learn.Trace.SubMean":    "1",      // 1 prevents sig amount of late deterioration
					"Prjn.SWt.Adapt.Lrate":        "0.0001", // 0.005 == .1 == .01
					"Prjn.SWt.Init.SPct":          "1",      // 1 >= lower (trace-v11)
					"Prjn.SWt.Adapt.SubMean":      "1",
					"Prjn.Com.PFail":              "0.0",
					"Prjn.Learn.KinaseCa.SpikeG":  "12", // 12 def / ra25
					"Prjn.Learn.KinaseCa.Dt.MTau": "5",  // 5 > 10 test more
					"Prjn.Learn.KinaseCa.Dt.PTau": "40",
					"Prjn.Learn.KinaseCa.Dt.DTau": "40",
					"Prjn.Learn.KinaseCa.UpdtThr": "0.01", // 0.01 > 0.02 max tolerable
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates -- smaller as network gets bigger",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2",  // .2 >= .3 > .15 > .1 > .05 @176
					"Prjn.Learn.Learn":   "true", // keep random weights to enable exploration
					// "Prjn.Learn.Lrate.Base":      "0.04", // lrate = 0 allows syn scaling still
				}},
			{Sel: ".Forward", Desc: "special forward-only params: com prob",
				Params: params.Params{}},
			{Sel: "#ITToOutput", Desc: "",
				Params: params.Params{
					"Prjn.Com.PFail": "0.0",
					// "Prjn.Learn.Lrate.Base": "0.1", // no effect
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
