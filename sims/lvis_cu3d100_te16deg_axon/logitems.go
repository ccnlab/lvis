// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/elog"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
	"github.com/emer/etable/split"
)

func (ss *Sim) ConfigLogItems() {
	ss.Logs.AddItem(&elog.Item{
		Name: "Run",
		Type: etensor.INT64,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.AllTimes): func(ctx *elog.Context) {
				ctx.SetStatInt("Run")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Params",
		Type: etensor.STRING,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.AllTimes): func(ctx *elog.Context) {
				ctx.SetString(ss.RunName())
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Epoch",
		Type: etensor.INT64,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			elog.Scopes([]elog.EvalModes{elog.AllModes}, []elog.Times{elog.Epoch, elog.Trial}): func(ctx *elog.Context) {
				ctx.SetStatInt("Epoch")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Trial",
		Type: etensor.INT64,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatInt("Trial")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Idx",
		Type: etensor.INT64,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetInt(ctx.Row)
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Cat",
		Type: etensor.STRING,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatString("TrlCat")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "TrialName",
		Type: etensor.STRING,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatString("TrialName")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Cycle",
		Type: etensor.INT64,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Cycle): func(ctx *elog.Context) {
				ctx.SetStatInt("Cycle")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Resp",
		Type: etensor.STRING,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatString("TrlResp")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Err",
		Type: etensor.FLOAT64,
		Plot: elog.DTrue,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlErr")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Err2",
		Type: etensor.FLOAT64,
		Plot: elog.DTrue,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlErr2")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "TrgAct",
		Type: etensor.FLOAT64,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlTrgAct")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "UnitErr",
		Type: etensor.FLOAT64,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlUnitErr")
			}, elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean)
			}, elog.Scope(elog.AllModes, elog.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, elog.Epoch, 5)
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "PctErr",
		Type:   etensor.FLOAT64,
		Plot:   elog.DFalse,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Write: elog.WriteMap{
			elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
				pcterr := ctx.SetAggItem(ctx.Mode, elog.Trial, "Err", agg.AggMean)
				epc := ctx.Stats.Int("Epoch")
				if ss.Stats.Int("FirstZero") < 0 && pcterr == 0 {
					ss.Stats.SetInt("FirstZero", epc)
				}
				if pcterr == 0 {
					nzero := ss.Stats.Int("NZero")
					ss.Stats.SetInt("NZero", nzero+1)
				} else {
					ss.Stats.SetInt("NZero", 0)
				}
			}, elog.Scope(elog.Test, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAggItem(ctx.Mode, elog.Trial, "Err", agg.AggMean)
			}, elog.Scope(elog.AllModes, elog.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, elog.Epoch, 5) // cached
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "PctCor",
		Type:   etensor.FLOAT64,
		Plot:   elog.DTrue,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetFloat64(1 - ctx.ItemFloatScope(ctx.Scope, "PctErr"))
			}, elog.Scope(elog.AllModes, elog.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, elog.Epoch, 5) // cached
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "PctErr2",
		Type: etensor.FLOAT64,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAggItem(ctx.Mode, elog.Trial, "Err2", agg.AggMean)
			}, elog.Scope(elog.AllModes, elog.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, elog.Epoch, 5)
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "CosDiff",
		Type:   etensor.FLOAT64,
		Plot:   elog.DTrue,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetFloat64(ss.Stats.Float("TrlCosDiff"))
			}, elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean)
			}, elog.Scope(elog.Train, elog.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(elog.Train, elog.Epoch, 5) // cached
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	// DecErr decoding
	ss.Logs.AddItem(&elog.Item{
		Name: "DecResp",
		Type: etensor.STRING,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatInt("TrlDecRespIdx")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "DecErr",
		Type: etensor.FLOAT64,
		Plot: elog.DTrue,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlDecErr")
			}, elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean)
			}, elog.Scope(elog.AllModes, elog.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, elog.Epoch, 5)
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "DecErr2",
		Type: etensor.FLOAT64,
		Plot: elog.DTrue,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlDecErr2")
			}, elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean)
			}, elog.Scope(elog.AllModes, elog.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, elog.Epoch, 5)
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	// FirstErr
	ss.Logs.AddItem(&elog.Item{
		Name: "FirstErr",
		Type: etensor.FLOAT64,
		Plot: elog.DTrue,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlFirstErr")
			}, elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean)
			}, elog.Scope(elog.AllModes, elog.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, elog.Epoch, 5)
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "FirstErr2",
		Type: etensor.FLOAT64,
		Plot: elog.DTrue,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlFirstErr2")
			}, elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean)
			}, elog.Scope(elog.AllModes, elog.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, elog.Epoch, 5)
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})

	ss.Logs.AddItem(&elog.Item{
		Name: "ErrTrgAct",
		Type: etensor.FLOAT64,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetStatFloat("EpcErrTrgAct")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "CorTrgAct",
		Type: etensor.FLOAT64,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetStatFloat("EpcCorTrgAct")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "PerTrlMSec",
		Type: etensor.FLOAT64,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
				nm := ctx.Item.Name
				tmr := ctx.Stats.StopTimer(nm)
				trls := ctx.Logs.Table(ctx.Mode, elog.Trial)
				tmr.N = trls.Rows
				pertrl := tmr.AvgMSecs()
				if ctx.Row == 0 {
					pertrl = 0 // first one is always inaccruate
				}
				ctx.Stats.SetFloat(nm, pertrl)
				ctx.SetFloat64(pertrl)
				tmr.ResetStart()
			}, elog.Scope(elog.AllModes, elog.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, elog.Epoch, 5)
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:  "FirstZero",
		Type:  etensor.FLOAT64,
		Plot:  elog.DTrue,
		Range: minmax.F64{Min: -1},
		Write: elog.WriteMap{
			elog.Scope(elog.Train, elog.Run): func(ctx *elog.Context) {
				ctx.SetStatInt("FirstZero")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:      "CatErr",
		Type:      etensor.FLOAT64,
		CellShape: []int{20},
		DimNames:  []string{"Cat"},
		Plot:      elog.DTrue,
		Range:     minmax.F64{Min: 0},
		// TensorIdx: -1, // plot all values
		Write: elog.WriteMap{
			elog.Scope(elog.Test, elog.Epoch): func(ctx *elog.Context) {
				ix, _ := ctx.Logs.NamedIdxView(elog.Test, elog.Trial, ctx.Item.Name)
				spl := split.GroupBy(ix, []string{"Cat"})
				split.AggTry(spl, "Err", agg.AggMean)
				cats := spl.AggsToTable(etable.ColNameOnly)
				ss.Logs.MiscTables[ctx.Item.Name] = cats
				ctx.SetTensor(cats.Cols[1])
			}}})

	// Copy over Testing items
	stats := []string{"UnitErr", "PctErr", "PctCor", "PctErr2", "CosDiff", "DecErr", "DecErr2", "FirstErr", "FirstErr2"}
	for _, st := range stats {
		stnm := st
		tstnm := "Tst" + st
		ss.Logs.AddItem(&elog.Item{
			Name: tstnm,
			Type: etensor.FLOAT64,
			Plot: elog.DFalse,
			Write: elog.WriteMap{
				elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ctx.SetFloat64(ctx.ItemFloat(elog.Test, elog.Epoch, stnm))
				}}})
	}

	// Standard stats for Ge and AvgAct tuning -- for all hidden, output layers
	layers := ss.Net.LayersByClass("Hidden", "Target")
	for _, lnm := range layers {
		clnm := lnm
		cly := ss.Net.LayerByName(clnm)
		uvals := ss.Stats.F32Tensor(clnm)
		cly.UnitValsRepTensor(uvals, "Act")               // for sizing
		if len(uvals.Shape.Shp) != len(cly.Shape().Shp) { // reshape
			uvals.SetShape(ss.CenterPoolShape(cly, 2), nil, cly.Shape().DimNames())
		}
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_ActAvg",
			Type:   etensor.FLOAT64,
			Plot:   elog.DFalse,
			FixMax: elog.DFalse,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.AllModes, elog.Cycle): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].Inhib.Act.Avg)
				}, elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].Inhib.Act.Avg)
				}, elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.ActAvg.ActMAvg)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_ActMax",
			Type:   etensor.FLOAT64,
			Plot:   elog.DFalse,
			FixMax: elog.DFalse,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.AllModes, elog.Cycle): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].Inhib.Act.Max)
				}, elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].Inhib.Act.Max)
				}, elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + "_MaxGeM",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].GeM.Max)
				}, elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.ActAvg.AvgMaxGeM)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + "_MaxGiM",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].GiM.Max)
				}, elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.ActAvg.AvgMaxGiM)
				}}})
		if clnm == "Output" {
			ss.Logs.AddItem(&elog.Item{
				Name:   clnm + "_GiMult",
				Type:   etensor.FLOAT64,
				Plot:   elog.DFalse,
				FixMax: elog.DFalse,
				Range:  minmax.F64{Max: 1},
				Write: elog.WriteMap{
					elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
						ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
						ctx.SetFloat32(ly.ActAvg.GiMult)
					}}})
		}
		// ss.Logs.AddItem(&elog.Item{
		// 	Name:  clnm + "_AvgDifAvg",
		// 	Type:  etensor.FLOAT64,
		// 	Plot:  elog.DFalse,
		// 	Range: minmax.F64{Max: 1},
		// 	Write: elog.WriteMap{
		// 		elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
		// 			ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
		// 			ctx.SetFloat32(ly.Pools[0].AvgDif.Avg)
		// 		}}})
		// ss.Logs.AddItem(&elog.Item{
		// 	Name:  clnm + "_AvgDifMax",
		// 	Type:  etensor.FLOAT64,
		// 	Plot:  elog.DFalse,
		// 	Range: minmax.F64{Max: 1},
		// 	Write: elog.WriteMap{
		// 		elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
		// 			ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
		// 			ctx.SetFloat32(ly.Pools[0].AvgDif.Max)
		// 		}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + "_DWtRaw_Max",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.Train, elog.Trial): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.DWtRaw.Max)
				}, elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean)
				}, elog.Scope(elog.Train, elog.Run): func(ctx *elog.Context) {
					ix := ctx.LastNRows(ctx.Mode, elog.Epoch, 5)
					ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + "_CosDiff",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.Train, elog.Trial): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(1 - ly.CosDiff.Cos)
				}, elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + "_CorCosDiff",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ctx.SetFloat64(ss.Logs.MiscTables["TrainErrStats"].CellFloat(lnm+"_CosDiff:Mean", 0))
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + "_ErrCosDiff",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ctx.SetFloat64(ss.Logs.MiscTables["TrainErrStats"].CellFloat(lnm+"_CosDiff:Mean", 1))
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + "_FirstCyc",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.Train, elog.Trial): func(ctx *elog.Context) {
					fcyc := ss.FirstActStat(ctx.Logs.Table(ctx.Mode, elog.Cycle), clnm)
					ctx.SetInt(fcyc)
				}, elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + "_FF_AvgMaxG",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.Train, elog.Trial): func(ctx *elog.Context) {
					ffpj := cly.RecvPrjn(0).(*axon.Prjn)
					ctx.SetFloat32(ffpj.GScale.AvgMax)
				}, elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + "_FB_AvgMaxG",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.Train, elog.Trial): func(ctx *elog.Context) {
					if cly.NRecvPrjns() > 1 {
						fbpj := cly.RecvPrjn(1).(*axon.Prjn)
						ctx.SetFloat32(fbpj.GScale.AvgMax)
					}
				}, elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean)
				}}})
		if clnm == "Output" {
			ss.Logs.AddItem(&elog.Item{
				Name:  clnm + "_FF_Scale",
				Type:  etensor.FLOAT64,
				Plot:  elog.DFalse,
				Range: minmax.F64{Max: 1},
				Write: elog.WriteMap{
					elog.Scope(elog.Train, elog.Trial): func(ctx *elog.Context) {
						ffpj := cly.RecvPrjn(0).(*axon.Prjn)
						ctx.SetFloat32(ffpj.GScale.Scale)
					}, elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
						ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean)
					}}})
			ss.Logs.AddItem(&elog.Item{
				Name:  clnm + "_FB_Scale",
				Type:  etensor.FLOAT64,
				Plot:  elog.DFalse,
				Range: minmax.F64{Max: 1},
				Write: elog.WriteMap{
					elog.Scope(elog.Train, elog.Trial): func(ctx *elog.Context) {
						if cly.NRecvPrjns() > 1 {
							fbpj := cly.RecvPrjn(1).(*axon.Prjn)
							ctx.SetFloat32(fbpj.GScale.Scale)
						}
					}, elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
						ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean)
					}}})
		}
		// PCA Analyze
		ss.Logs.AddItem(&elog.Item{
			Name:      clnm + "_ActM",
			Type:      etensor.FLOAT64,
			CellShape: uvals.Shape.Shp,
			FixMax:    elog.DTrue,
			Range:     minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.Analyze, elog.Trial): func(ctx *elog.Context) {
					ctx.SetLayerRepTensor(clnm, "ActM")
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name: clnm + "_PCA_NStrong",
			Type: etensor.FLOAT64,
			Plot: elog.DFalse,
			Write: elog.WriteMap{
				elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ctx.SetStatFloat(ctx.Item.Name)
				}, elog.Scope(elog.AllModes, elog.Run): func(ctx *elog.Context) {
					ix := ctx.LastNRows(ctx.Mode, elog.Epoch, 5)
					ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name: clnm + "_PCA_Top5",
			Type: etensor.FLOAT64,
			Plot: elog.DFalse,
			Write: elog.WriteMap{
				elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ctx.SetStatFloat(ctx.Item.Name)
				}, elog.Scope(elog.AllModes, elog.Run): func(ctx *elog.Context) {
					ix := ctx.LastNRows(ctx.Mode, elog.Epoch, 5)
					ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name: clnm + "_PCA_Next5",
			Type: etensor.FLOAT64,
			Plot: elog.DFalse,
			Write: elog.WriteMap{
				elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ctx.SetStatFloat(ctx.Item.Name)
				}, elog.Scope(elog.AllModes, elog.Run): func(ctx *elog.Context) {
					ix := ctx.LastNRows(ctx.Mode, elog.Epoch, 5)
					ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name: clnm + "_PCA_Rest",
			Type: etensor.FLOAT64,
			Plot: elog.DFalse,
			Write: elog.WriteMap{
				elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ctx.SetStatFloat(ctx.Item.Name)
				}, elog.Scope(elog.AllModes, elog.Run): func(ctx *elog.Context) {
					ix := ctx.LastNRows(ctx.Mode, elog.Epoch, 5)
					ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
				}}})
	}
	layers = ss.Net.LayersByClass("Target")
	for _, lnm := range layers {
		clnm := lnm
		cly := ss.Net.LayerByName(clnm)
		uvals := ss.Stats.F32Tensor(clnm)
		cly.UnitValsRepTensor(uvals, "Act") // for sizing
		ss.Logs.AddItem(&elog.Item{
			Name:      clnm + "_Act",
			Type:      etensor.FLOAT32,
			CellShape: uvals.Shape.Shp,
			FixMax:    elog.DTrue,
			Range:     minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.AllModes, elog.Cycle): func(ctx *elog.Context) {
					ctx.SetLayerRepTensor(clnm, "Act")
				}}})
	}

	// input layer average activity -- important for tuning
	layers = ss.Net.LayersByClass("Input")
	for _, lnm := range layers {
		clnm := lnm
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_ActAvg",
			Type:   etensor.FLOAT64,
			Plot:   elog.DFalse,
			FixMax: elog.DTrue,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.ActAvg.ActMAvg)
				}}})
	}
}
