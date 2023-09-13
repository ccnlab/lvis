// Copyright (c) 2019, The Emergent Authors. All rights reserved.
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
	"fmt"
	"log"
	"os"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/decoder"
	"github.com/emer/emergent/econfig"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/timer"
	"github.com/emer/empi/empi"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview"
	"github.com/emer/etable/minmax"
	"github.com/emer/etable/split"
	"github.com/emer/etable/tsragg"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/mat32"
)

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	if sim.Config.GUI {
		gimain.Main(sim.RunGUI)
	} else {
		sim.RunNoGUI()
	}
}

// see params.go for params, config.go for Config

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// simulation configuration parameters -- set by .toml config file and / or args
	Config Config `desc:"simulation configuration parameters -- set by .toml config file and / or args"`

	// [view: no-inline] the network -- click to view / edit parameters for layers, prjns, etc
	Net *axon.Network `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`

	// [view: inline] all parameter management
	Params emer.NetParams `view:"inline" desc:"all parameter management"`

	// [view: no-inline] contains looper control loops for running sim
	Loops *looper.Manager `view:"no-inline" desc:"contains looper control loops for running sim"`

	// contains computed statistic values
	Stats estats.Stats `desc:"contains computed statistic values"`

	// Contains all the logs and information about the logs.'
	Logs elog.Logs `desc:"Contains all the logs and information about the logs.'"`

	// [view: no-inline] Environments
	Envs env.Envs `view:"no-inline" desc:"Environments"`

	// axon timing parameters and state
	Context axon.Context `desc:"axon timing parameters and state"`

	// [view: inline] netview update parameters
	ViewUpdt netview.ViewUpdt `view:"inline" desc:"netview update parameters"`

	// decoder for better output
	Decoder decoder.SoftMax `desc:"decoder for better output"`

	// special projections -- see config.go
	Prjns Prjns `desc:"special projections -- see config.go "`

	// [view: -] manages all the gui elements
	GUI egui.GUI `view:"-" desc:"manages all the gui elements"`

	// [view: -] a list of random seeds to use for each run
	RndSeeds erand.Seeds `view:"-" desc:"a list of random seeds to use for each run"`

	// [view: -] mpi communicator
	Comm *mpi.Comm `view:"-" desc:"mpi communicator"`

	// [view: -] buffer of all dwt weight changes -- for mpi sharing
	AllDWts []float32 `view:"-" desc:"buffer of all dwt weight changes -- for mpi sharing"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Config.Defaults()
	ss.Prjns.Defaults()
	econfig.Config(&ss.Config, "config.toml")
	if ss.Config.Run.MPI {
		ss.MPIInit()
	}
	if mpi.WorldRank() != 0 {
		ss.Config.Log.SaveWts = false
		ss.Config.Log.NetData = false
	}
	if ss.Config.Bench {
		if ss.Config.Run.MPI {
			ss.Config.Run.NTrials = 512
		} else {
			ss.Config.Run.NTrials = 64
		}
		ss.Config.Run.NEpochs = 1
	}
	ss.Net = &axon.Network{}
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	ss.Stats.Init()
	ss.RndSeeds.Init(100) // max 100 runs
	ss.InitRndSeed(0)
	ss.Context.Defaults()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Params.Params, &ss.Config, ss.Config.Params.Good)
		os.Exit(0)
	}
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn, tst *ImagesEnv
	if len(ss.Envs) == 0 {
		trn = &ImagesEnv{}
		tst = &ImagesEnv{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*ImagesEnv)
		tst = ss.Envs.ByMode(etime.Test).(*ImagesEnv)
	}

	plus := false // plus images are a bit worse overall -- stranger objects etc.

	var path string
	if plus {
		path = "images/CU3D_100_plus_renders"
		trn.ImageFile = "cu3d100plus"
	} else {
		path = "images/CU3D_100_renders_lr20_u30_nb"
		trn.ImageFile = "cu3d100old"
	}

	trn.Nm = etime.Train.String()
	trn.Dsc = "training params and state"
	trn.Defaults()
	trn.RndSeed = 73
	trn.NOutPer = ss.Config.Env.NOutPer
	trn.High16 = false // not useful -- may need more tuning?
	trn.ColorDoG = true
	trn.Images.NTestPerCat = 2
	trn.Images.SplitByItm = true
	tst.OutRandom = ss.Config.Env.RndOutPats
	trn.OutSize.Set(10, 10)
	trn.Images.SetPath(path, []string{".png"}, "_")
	trn.OpenConfig()
	if ss.Config.Env.Env != nil {
		params.ApplyMap(trn, ss.Config.Env.Env, ss.Config.Debug)
	}
	// trn.Images.OpenPath(path, []string{".png"}, "_")
	// trn.SaveConfig()

	trn.Validate()
	trn.Trial.Max = ss.Config.Run.NTrials

	tst.Nm = etime.Test.String()
	tst.Dsc = "testing params and state"
	tst.ImageFile = trn.ImageFile
	tst.Defaults()
	tst.RndSeed = 73
	trn.NOutPer = ss.Config.Env.NOutPer
	tst.High16 = trn.High16
	tst.ColorDoG = trn.ColorDoG
	tst.Images.NTestPerCat = 2
	tst.Images.SplitByItm = true
	tst.OutRandom = ss.Config.Env.RndOutPats
	tst.OutSize.Set(10, 10)
	tst.Test = true
	tst.Images.SetPath(path, []string{".png"}, "_")
	tst.OpenConfig()
	// tst.Images.OpenPath(path, []string{".png"}, "_")
	// tst.SaveConfig()
	tst.Trial.Max = ss.Config.Run.NTrials
	if ss.Config.Env.Env != nil {
		params.ApplyMap(tst, ss.Config.Env.Env, ss.Config.Debug)
	}
	tst.Validate()

	/*
		// Delete to 60
			last20 := []string{"submarine", "synthesizer", "tablelamp", "tank", "telephone", "television", "toaster", "toilet", "trafficcone", "trafficlight", "trex", "trombone", "tropicaltree", "trumpet", "turntable", "umbrella", "wallclock", "warningsign", "wrench", "yacht"}
			next20 := []string{"pedestalsink", "person", "piano", "plant", "plate", "pliers", "propellor", "remote", "rolltopdesk", "sailboat", "scissors", "screwdriver", "sectionalcouch", "simpledesk", "skateboard", "skull", "slrcamera", "speaker", "spotlightlamp", "stapler"}
			last40 := append(last20, next20...)
			trn.Images.DeleteCats(last40)
			tst.Images.DeleteCats(last40)
	*/

	/*
		objs20 := []string{"banana", "layercake", "trafficcone", "sailboat", "trex", "person", "guitar", "tablelamp", "doorknob", "handgun", "donut", "chair", "slrcamera", "elephant", "piano", "fish", "car", "heavycannon", "stapler", "motorcycle"}

		objsnxt20 := []string{"submarine", "synthesizer", "tank", "telephone", "television", "toaster", "toilet", "trafficlight", "tropicaltree", "trumpet", "turntable", "umbrella", "wallclock", "warningsign", "wrench", "yacht", "pedestalsink", "pliers", "sectionalcouch", "skull"}

		objs40 := append(objs20, objsnxt20...)

		trn.Images.SelectCats(objs40)
		tst.Images.SelectCats(objs40)
	*/

	// remove most confusable items
	confuse := []string{"blade", "flashlight", "pckeyboard", "scissors", "screwdriver", "submarine"}
	trn.Images.DeleteCats(confuse)
	tst.Images.DeleteCats(confuse)

	if ss.Config.Run.MPI {
		if ss.Config.Debug {
			mpi.Printf("Did Env MPIAlloc\n")
		}
		trn.MPIAlloc()
		tst.MPIAlloc()
	}

	trn.Init(0)
	tst.Init(0)

	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	net.InitName(net, "Lvis")
	net.SetMaxData(ctx, ss.Config.Run.NData)
	net.SetRndSeed(ss.RndSeeds[0]) // init new separate random seed, using run = 0

	trn := ss.Envs.ByMode(etime.Train).(*ImagesEnv)

	v1nrows := 5
	if trn.V1m16.SepColor {
		v1nrows += 4
	}
	hi16 := trn.High16
	cdog := trn.ColorDoG

	v2mNp := 8
	v2lNp := 4
	v2Nu := 8
	v4Np := 4
	v4Nu := 10
	if ss.Config.Params.SubPools {
		v2mNp *= 2
		v2lNp *= 2
		v2Nu = 6
		v4Np = 8
		v4Nu = 7
	}

	v1m16 := net.AddLayer4D("V1m16", 16, 16, v1nrows, 4, axon.InputLayer)
	v1l16 := net.AddLayer4D("V1l16", 8, 8, v1nrows, 4, axon.InputLayer)
	v1m8 := net.AddLayer4D("V1m8", 16, 16, v1nrows, 4, axon.InputLayer)
	v1l8 := net.AddLayer4D("V1l8", 8, 8, v1nrows, 4, axon.InputLayer)
	v1m16.SetClass("V1m")
	v1l16.SetClass("V1l")
	v1m8.SetClass("V1m")
	v1l8.SetClass("V1l")

	v1m16.SetRepIdxsShape(ss.CenterPoolIdxs(v1m16, 2), emer.CenterPoolShape(v1m16, 2))
	v1l16.SetRepIdxsShape(ss.CenterPoolIdxs(v1l16, 2), emer.CenterPoolShape(v1l16, 2))
	v1m8.SetRepIdxsShape(ss.CenterPoolIdxs(v1m8, 2), emer.CenterPoolShape(v1m8, 2))
	v1l8.SetRepIdxsShape(ss.CenterPoolIdxs(v1l8, 2), emer.CenterPoolShape(v1l8, 2))

	// not useful so far..
	// clst := net.AddLayer2D("Claustrum", 5, 5, axon.SuperLayer)

	var v1cm16, v1cl16, v1cm8, v1cl8 *axon.Layer
	if cdog {
		v1cm16 = net.AddLayer4D("V1Cm16", 16, 16, 2, 2, axon.InputLayer)
		v1cl16 = net.AddLayer4D("V1Cl16", 8, 8, 2, 2, axon.InputLayer)
		v1cm8 = net.AddLayer4D("V1Cm8", 16, 16, 2, 2, axon.InputLayer)
		v1cl8 = net.AddLayer4D("V1Cl8", 8, 8, 2, 2, axon.InputLayer)
		v1cm16.SetClass("V1Cm")
		v1cl16.SetClass("V1Cl")
		v1cm8.SetClass("V1Cm")
		v1cl8.SetClass("V1Cl")

		v1cm16.SetRepIdxsShape(ss.CenterPoolIdxs(v1cm16, 2), emer.CenterPoolShape(v1cm16, 2))
		v1cl16.SetRepIdxsShape(ss.CenterPoolIdxs(v1cl16, 2), emer.CenterPoolShape(v1cl16, 2))
		v1cm8.SetRepIdxsShape(ss.CenterPoolIdxs(v1cm8, 2), emer.CenterPoolShape(v1cm8, 2))
		v1cl8.SetRepIdxsShape(ss.CenterPoolIdxs(v1cl8, 2), emer.CenterPoolShape(v1cl8, 2))
	}

	v2m16 := net.AddLayer4D("V2m16", v2mNp, v2mNp, v2Nu, v2Nu, axon.SuperLayer)
	v2l16 := net.AddLayer4D("V2l16", v2lNp, v2lNp, v2Nu, v2Nu, axon.SuperLayer)
	v2m8 := net.AddLayer4D("V2m8", v2mNp, v2mNp, v2Nu, v2Nu, axon.SuperLayer)
	v2l8 := net.AddLayer4D("V2l8", v2lNp, v2lNp, v2Nu, v2Nu, axon.SuperLayer)
	v2m16.SetClass("V2m V2")
	v2l16.SetClass("V2l V2")
	v2m8.SetClass("V2m V2")
	v2l8.SetClass("V2l V2")

	v2m16.SetRepIdxsShape(ss.CenterPoolIdxs(v2m16, 2), emer.CenterPoolShape(v2m16, 2))
	v2l16.SetRepIdxsShape(ss.CenterPoolIdxs(v2l16, 2), emer.CenterPoolShape(v2l16, 2))
	v2m8.SetRepIdxsShape(ss.CenterPoolIdxs(v2m8, 2), emer.CenterPoolShape(v2m8, 2))
	v2l8.SetRepIdxsShape(ss.CenterPoolIdxs(v2l8, 2), emer.CenterPoolShape(v2l8, 2))

	var v1h16, v2h16, v3h16 *axon.Layer
	if hi16 {
		v1h16 = net.AddLayer4D("V1h16", 32, 32, 5, 4, axon.InputLayer)
		v2h16 = net.AddLayer4D("V2h16", 32, 32, v2Nu, v2Nu, axon.SuperLayer)
		v3h16 = net.AddLayer4D("V3h16", 16, 16, v2Nu, v2Nu, axon.SuperLayer)
		v1h16.SetClass("V1h")
		v2h16.SetClass("V2h V2")
		v3h16.SetClass("V3h")

		v1h16.SetRepIdxsShape(ss.CenterPoolIdxs(v1h16, 2), emer.CenterPoolShape(v1h16, 2))
		v2h16.SetRepIdxsShape(ss.CenterPoolIdxs(v2h16, 2), emer.CenterPoolShape(v2h16, 2))
		v3h16.SetRepIdxsShape(ss.CenterPoolIdxs(v3h16, 2), emer.CenterPoolShape(v3h16, 2))
	}

	v4f16 := net.AddLayer4D("V4f16", v4Np, v4Np, v4Nu, v4Nu, axon.SuperLayer)
	v4f8 := net.AddLayer4D("V4f8", v4Np, v4Np, v4Nu, v4Nu, axon.SuperLayer)
	v4f16.SetClass("V4")
	v4f8.SetClass("V4")

	v4f16.SetRepIdxsShape(ss.CenterPoolIdxs(v4f16, 2), emer.CenterPoolShape(v4f16, 2))
	v4f8.SetRepIdxsShape(ss.CenterPoolIdxs(v4f8, 2), emer.CenterPoolShape(v4f8, 2))

	teo16 := net.AddLayer4D("TEOf16", 2, 2, 15, 15, axon.SuperLayer)
	teo8 := net.AddLayer4D("TEOf8", 2, 2, 15, 15, axon.SuperLayer)
	teo16.SetClass("TEO")
	teo8.SetClass("TEO")

	te := net.AddLayer4D("TE", 2, 2, 15, 15, axon.SuperLayer)

	var out *axon.Layer
	if ss.Config.Env.RndOutPats {
		out = net.AddLayer2D("Output", trn.OutSize.Y, trn.OutSize.X, axon.TargetLayer)
	} else {
		// out = net.AddLayer4D("Output", trn.OutSize.Y, trn.OutSize.X, trn.NOutPer, 1, axon.TargetLayer)
		// 2D layer:
		out = net.AddLayer2D("Output", trn.OutSize.Y, trn.OutSize.X*trn.NOutPer, axon.TargetLayer)
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

	if ss.Config.Params.SubPools {
		p4x4s2 = pj.Prjn4x4Skp2Sub2
		p2x2s1 = pj.Prjn2x2Skp1Sub2
		p4x4s2send = pj.Prjn4x4Skp2Sub2Send
		p2x2s1send = pj.Prjn2x2Skp1Sub2Send
		p4x4s2recip = pj.Prjn4x4Skp2Sub2SendRecip
		p2x2s1recip = pj.Prjn2x2Skp1Sub2SendRecip
		v4toteo = pj.Prjn4x4Skp0Sub2
		teotov4 = pj.Prjn4x4Skp0Sub2Recip
	}

	net.ConnectLayers(v1m16, v2m16, p4x4s2, axon.ForwardPrjn).SetClass("V1V2")
	net.ConnectLayers(v1l16, v2m16, p2x2s1, axon.ForwardPrjn).SetClass("V1V2fmSm V1V2")

	net.ConnectLayers(v1l16, v2l16, p4x4s2, axon.ForwardPrjn).SetClass("V1V2")

	net.ConnectLayers(v1m8, v2m8, p4x4s2, axon.ForwardPrjn).SetClass("V1V2")
	net.ConnectLayers(v1l8, v2m8, p2x2s1, axon.ForwardPrjn).SetClass("V1V2fmSm V1V2")

	net.ConnectLayers(v1l8, v2l8, p4x4s2, axon.ForwardPrjn).SetClass("V1V2")

	if cdog {
		net.ConnectLayers(v1cm16, v2m16, p4x4s2, axon.ForwardPrjn).SetClass("V1V2")
		net.ConnectLayers(v1cl16, v2m16, p2x2s1, axon.ForwardPrjn).SetClass("V1V2fmSm V1V2")

		net.ConnectLayers(v1cl16, v2l16, p4x4s2, axon.ForwardPrjn).SetClass("V1V2")

		net.ConnectLayers(v1cm8, v2m8, p4x4s2, axon.ForwardPrjn).SetClass("V1V2")
		net.ConnectLayers(v1cl8, v2m8, p2x2s1, axon.ForwardPrjn).SetClass("V1V2fmSm V1V2")

		net.ConnectLayers(v1cl8, v2l8, p4x4s2, axon.ForwardPrjn).SetClass("V1V2")
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
		net.ConnectLayers(v1h16, v2h16, p4x4s2, axon.ForwardPrjn).SetClass("V1V2")
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
	net.ConnectLayers(v4f8, teo16, v4toteo, axon.ForwardPrjn).SetClass("V4TEOoth")

	v4teo, teov4 = net.BidirConnectLayers(v4f8, teo8, v4toteo)
	v4teo.SetClass("V4TEO")
	teov4.SetClass("TEOV4").SetPattern(teotov4)
	net.ConnectLayers(v4f16, teo8, v4toteo, axon.ForwardPrjn).SetClass("V4TEOoth")

	teote, teteo := net.BidirConnectLayers(teo16, te, full)
	teote.SetClass("TEOTE")
	teteo.SetClass("TETEO")
	teote, teteo = net.BidirConnectLayers(teo8, te, full)
	teote.SetClass("TEOTE")
	teteo.SetClass("TETEO")

	// TEO -> out ends up saturating quite a bit with consistently high weights,
	// but removing those projections is not good -- still makes use of them.
	// perhaps in a transitional way that sets up better TE reps.

	// outteo := net.ConnectLayers(out, teo16, full, emer.Back)
	teoout, outteo := net.BidirConnectLayers(teo16, out, full)
	teoout.SetClass("TEOOut ToOut")
	outteo.SetClass("OutTEO FmOut")

	// outteo = net.ConnectLayers(out, teo8, full, emer.Back)
	teoout, outteo = net.BidirConnectLayers(teo8, out, full)
	teoout.SetClass("TEOOut ToOut")
	outteo.SetClass("OutTEO FmOut")

	teout, _ := net.BidirConnectLayers(te, out, full)
	teout.SetClass("ToOut FmOut")

	/*
		// trace: not useful
		// v59 459 -- only useful later -- TEO maybe not doing as well later?
		v4out, outv4 := net.BidirConnectLayers(v4f16, out, full)
		v4out.SetClass("V4Out ToOut")
		outv4.SetClass("OutV4 FmOut")

		v4out, outv4 = net.BidirConnectLayers(v4f8, out, full)
		v4out.SetClass("V4Out ToOut")
		outv4.SetClass("OutV4 FmOut")
	*/

	/*
		var v2inhib, v4inhib prjn.Pattern
		v2inhib = pool1to1
		v4inhib = pool1to1
		if ss.SubPools {
			v2inhib = pj.Prjn2x2Skp2 // pj.Prjn6x6Skp2Lat
			v4inhib = pj.Prjn2x2Skp2
		}

			// this extra inhibition drives decorrelation, produces significant learning benefits
			net.LateralConnectLayerPrjn(v2m16, v2inhib, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
			net.LateralConnectLayerPrjn(v2l16, v2inhib, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
			net.LateralConnectLayerPrjn(v2m8, v2inhib, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
			net.LateralConnectLayerPrjn(v2l8, v2inhib, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
			net.LateralConnectLayerPrjn(v4f16, v4inhib, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
			net.LateralConnectLayerPrjn(v4f8, v4inhib, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
			net.LateralConnectLayerPrjn(teo16, pool1to1, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
			net.LateralConnectLayerPrjn(teo8, pool1to1, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
			net.LateralConnectLayerPrjn(te, pool1to1, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)

			if hi16 {
				net.LateralConnectLayerPrjn(v2h16, v2inhib, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
				net.LateralConnectLayerPrjn(v3h16, v2inhib, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
			}
	*/

	///////////////////////
	// 	Shortcuts:

	// clst not useful
	// net.ConnectLayers(v1l16, clst, full, axon.ForwardPrjn)

	// V1 shortcuts best for syncing all layers -- like the pulvinar basically
	net.ConnectLayers(v1l16, v4f16, rndcut, axon.ForwardPrjn).SetClass("V1SC")
	net.ConnectLayers(v1l8, v4f8, rndcut, axon.ForwardPrjn).SetClass("V1SC")
	net.ConnectLayers(v1l16, teo16, rndcut, axon.ForwardPrjn).SetClass("V1SC")
	net.ConnectLayers(v1l16, teo16, rndcut, axon.ForwardPrjn).SetClass("V1SC")
	net.ConnectLayers(v1l8, teo8, rndcut, axon.ForwardPrjn).SetClass("V1SC")
	net.ConnectLayers(v1l8, teo8, rndcut, axon.ForwardPrjn).SetClass("V1SC")
	net.ConnectLayers(v1l16, te, rndcut, axon.ForwardPrjn).SetClass("V1SC")
	net.ConnectLayers(v1l8, te, rndcut, axon.ForwardPrjn).SetClass("V1SC")

	if hi16 {
		net.ConnectLayers(v1l16, v3h16, rndcut, axon.ForwardPrjn).SetClass("V1SC")
	}

	//////////////////////
	// 	Positioning

	space := float32(4)
	v1m8.PlaceRightOf(v1m16, space)

	v1l16.PlaceBehind(v1m16, space)
	v1l8.PlaceBehind(v1m8, space)
	// clst.PlaceBehind(v1l8, XAlign: relpos.Left, Space: 4, Scale: 2})

	if cdog {
		v1cm16.PlaceRightOf(v1m8, space)
		v1cm8.PlaceRightOf(v1cm16, space)
		v1cl16.PlaceBehind(v1cm16, space)
		v1cl8.PlaceBehind(v1cm8, space)
	}

	if hi16 {
		v1h16.PlaceRightOf(v1m8, space)
		v2h16.PlaceRightOf(v2m8, space)
		v3h16.PlaceBehind(v4f16, space)
	}

	v2m16.PlaceAbove(v1m16)

	v2m8.PlaceRightOf(v2m16, space)

	v2l16.PlaceBehind(v2m16, space)
	v2l8.PlaceBehind(v2m8, space)

	v4f16.PlaceAbove(v2m16)
	teo16.PlaceRightOf(v4f16, space)

	v4f8.PlaceRightOf(teo16, space)
	teo8.PlaceRightOf(v4f8, space)

	te.PlaceBehind(teo8, 15)

	out.PlaceBehind(te, 15)

	net.Build(ctx)
	net.Defaults()
	net.SetNThreads(ss.Config.Run.NThreads)
	ss.ApplyParams()
	net.InitWts(ctx)

	mpi.Println(net.SizeReport(false))

	// adding each additional layer type improves decoding..
	layers := []emer.Layer{v4f16, v4f8, teo16, teo8, out}
	// layers := []emer.Layer{teo16, teo8, out}
	// layers := []emer.Layer{teo16, teo8}
	ss.Decoder.InitLayer(len(trn.Images.Cats), layers)
	ss.Decoder.Lrate = 0.05 // 0.05 > 0.1 > 0.2 for larger number of objs!
	if ss.Config.Run.MPI {
		ss.Decoder.Comm = ss.Comm
	}
}

func (ss *Sim) ApplyParams() {
	ss.Params.SetAll() // first hard-coded defaults
	if ss.Config.Params.Network != nil {
		ss.Params.SetNetworkMap(ss.Net, ss.Config.Params.Network)
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	if ss.Config.GUI {
		ss.Stats.SetString("RunName", ss.Params.RunName(0)) // in case user interactively changes tag
	}
	ss.Loops.ResetCounters()
	ss.InitRndSeed(0)
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.Net.GPU.SyncParamsToGPU()
	ss.NewRun()
	ss.ViewUpdt.Update()
	ss.ViewUpdt.RecordSyns()
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed(run int) {
	ss.RndSeeds.Set(run)
	ss.RndSeeds.Set(run, &ss.Net.Rand)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	ss.Context.SlowInterval = int32(4 * 100) // decompensate..

	totND := ss.Config.Run.NData * mpi.WorldSize() // both sources of data parallel
	totTrls := int(mat32.IntMultipleGE(float32(ss.Config.Run.NTrials), float32(totND)))
	trls := totTrls / mpi.WorldSize()

	man.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.Run.NRuns).
		AddTime(etime.Epoch, ss.Config.Run.NEpochs).
		AddTimeIncr(etime.Trial, trls, ss.Config.Run.NData).
		AddTime(etime.Cycle, 200)

	man.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTimeIncr(etime.Trial, trls, ss.Config.Run.NData).
		AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdt) // std algo code

	man.GetLoop(etime.Train, etime.Trial).OnEnd.Replace("UpdateWeights", func() {
		ss.Net.DWt(&ss.Context)
		if ss.ViewUpdt.IsViewingSynapse() {
			ss.Net.GPU.SyncSynapsesFmGPU()
			ss.Net.GPU.SyncSynCaFmGPU() // note: only time we call this
			ss.ViewUpdt.RecordSyns()    // note: critical to update weights here so DWt is visible
		}
		ss.MPIWtFmDWt()
	})
	if ss.Config.Debug {
		man.GetLoop(etime.Train, etime.Epoch).OnStart.Add("ValidateMPIReplicaConsistency",
			ss.AssertMPIReplicaConsistency)
	}

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
	}

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	// Add Testing
	trainEpoch := man.GetLoop(etime.Train, etime.Epoch)
	trainEpoch.OnStart.Add("TestAtInterval", func() {
		if (ss.Config.Run.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.Run.TestInterval == 0) {
			// Note the +1 so that it doesn't occur at the 0th timestep.
			ss.TestAll()
		}
	})

	trainEpoch.OnEnd.Add("RandCheck", func() {
		if ss.Config.Run.MPI {
			empi.RandCheck(ss.Comm) // prints error message
		}
	})

	/////////////////////////////////////////////
	// Logging

	man.GetLoop(etime.Test, etime.Epoch).OnEnd.Add("LogTestErrors", func() {
		axon.LogTestErrors(&ss.Logs)
	})
	man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("PCAStats", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.Config.Run.PCAInterval > 0) && (trnEpc%ss.Config.Run.PCAInterval == 0) {
			if ss.Config.Run.MPI {
				ss.Logs.MPIGatherTableRows(etime.Analyze, etime.Trial, ss.Comm)
			}
			axon.PCAStats(ss.Net, &ss.Logs, &ss.Stats)
			ss.Logs.ResetLog(etime.Analyze, etime.Trial)
		}
	})

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)

	man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("LogAnalyze", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.Config.Run.PCAInterval > 0) && (trnEpc%ss.Config.Run.PCAInterval == 0) {
			ss.Log(etime.Analyze, etime.Trial)
		}
	})

	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("RunStats", func() {
		ss.Logs.RunStats("PctCor", "FirstZero", "LastZero")
	})

	// Save weights to file at end, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() { ss.SaveWeights() })

	// lrate schedule
	// man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("LrateSched", func() {
	// 	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	// 	switch trnEpc {
	// 	case 10:
	// 		mpi.Printf("NData increase back to specified: %d at: %d\n", ss.Config.Run.NData, trnEpc)
	// 		ss.Context.NetIdxs.NData = uint32(ss.Config.Run.NData)
	// 	}
	// })

	man.GetLoop(etime.Train, etime.Epoch).AddNewEvent("SaveWeights", 10, func() { ss.SaveWeights() })
	man.GetLoop(etime.Train, etime.Epoch).AddNewEvent("SaveWeights", 100, func() { ss.SaveWeights() })
	man.GetLoop(etime.Train, etime.Epoch).AddNewEvent("SaveWeights", 500, func() { ss.SaveWeights() })
	man.GetLoop(etime.Train, etime.Epoch).AddNewEvent("SaveWeights", 1000, func() { ss.SaveWeights() })
	man.GetLoop(etime.Train, etime.Epoch).AddNewEvent("SaveWeights", 1500, func() { ss.SaveWeights() })

	////////////////////////////////////////////
	// GUI
	if !ss.Config.GUI {
		// man.GetLoop(etime.Test, etime.Trial).Main.Add("NetDataRecord", func() {
		// 	ss.GUI.NetDataRecord(ss.ViewUpdt.Text)
		// })
	} else {
		// this is actually fairly expensive
		man.GetLoop(etime.Test, etime.Trial).OnEnd.Add("ActRFs", func() {
			for di := 0; di < int(ss.Context.NetIdxs.NData); di++ {
				ss.Stats.UpdateActRFs(ss.Net, "ActM", 0.01, di)
			}
		})

		man.GetLoop(etime.Train, etime.Trial).OnStart.Add("UpdtImage", func() {
			ss.GUI.Grid("Image").UpdateSig()
		})
		man.GetLoop(etime.Test, etime.Trial).OnStart.Add("UpdtImage", func() {
			ss.GUI.Grid("Image").UpdateSig()
		})

		axon.LooperUpdtNetView(man, &ss.ViewUpdt, ss.Net, ss.NetViewCounters)
		axon.LooperUpdtPlots(man, &ss.GUI)
	}

	if ss.Config.Debug {
		mpi.Println(man.DocString())
	}
	ss.Loops = man
}

// SaveWeights saves weights with filename recording run, epoch
func (ss *Sim) SaveWeights() {
	ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
	axon.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWts, ctrString, ss.Stats.String("RunName"))
}

// CenterPoolIdxs returns the unit indexes for 2x2 center pools
// if sub-pools are present, then only first such subpool is used.
func (ss *Sim) CenterPoolIdxs(ly emer.Layer, n int) []int {
	npy := ly.Shape().Dim(0)
	npx := ly.Shape().Dim(1)
	npxact := npx
	nu := ly.Shape().Dim(2) * ly.Shape().Dim(3)
	nsp := 1
	if ss.Config.Params.SubPools {
		npy /= 2
		npx /= 2
		nsp = 2
	}
	cpy := (npy - n) / 2
	cpx := (npx - n) / 2
	nt := n * n * nu
	idxs := make([]int, nt)

	ix := 0
	for py := 0; py < 2; py++ {
		y := (py + cpy) * nsp
		for px := 0; px < 2; px++ {
			x := (px + cpx) * nsp
			si := (y*npxact + x) * nu
			for ni := 0; ni < nu; ni++ {
				idxs[ix+ni] = si + ni
			}
			ix += nu
		}
	}
	return idxs
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net
	ev := ss.Envs.ByMode(ctx.Mode).(*ImagesEnv)
	net.InitExt(ctx)
	lays := net.LayersByType(axon.InputLayer, axon.TargetLayer)
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		ev.Step()
		ss.Stats.SetStringDi("TrialName", int(di), ev.String()) // for logging
		ss.Stats.SetIntDi("TrlCatIdx", int(di), ev.CurCatIdx)
		ss.Stats.SetStringDi("TrlCat", int(di), ev.CurCat)
		for _, lnm := range lays {
			ly := ss.Net.AxonLayerByName(lnm)
			pats := ev.State(ly.Nm)
			if pats != nil {
				ly.ApplyExt(ctx, di, pats)
			}
		}
	}
	net.ApplyExts(ctx)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRndSeed(ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur)
	ss.Envs.ByMode(etime.Train).Init(0)
	ss.Envs.ByMode(etime.Test).Init(0)
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.Net.InitWts(ctx)
	ss.InitStats()
	ss.StatCounters(0)
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Stats.ActRFs.Reset()
	ss.Loops.ResetAndRun(etime.Test)
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
	ss.Stats.ActRFsAvgNorm()
	ss.GUI.ViewActRFs(&ss.Stats.ActRFs)

}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	// ss.Logs.ResetLog(etime.Test, etime.Epoch) // only show last row
	ss.GUI.StopNow = false
	ss.TestAll()
	ss.GUI.Stopped()
}

// ConfusionTstPlot plots the current confusion probability values.
// if cat is empty then it is the diagonal accuracy across all cats
// otherwise it is the confusion row for given category.
// data goes in the TrlErr = Err column.
func (ss *Sim) ConfusionTstPlot(cat string) {
	ev := ss.Envs[etime.Test.String()].(*ImagesEnv)
	ss.Logs.ResetLog(etime.Test, etime.Trial)
	nc := ss.Stats.Confusion.N.Len()
	ti := -1
	if cat != "" {
		ti = ev.Images.CatMap[cat]
	}
	for i := 0; i < nc; i++ {
		ev.Trial.Cur = i
		ev.CurCat = ev.Images.Cats[i]
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

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("UnitErr", 0.0)
	ss.Stats.SetFloat("CorSim", 0.0)
	ss.Logs.InitErrStats() // inits TrlErr, FirstZero, LastZero, NZero
	ss.Stats.SetFloat("TrlErr2", 0.0)
	ss.Stats.SetString("TrlCat", "0")
	ss.Stats.SetInt("TrlCatIdx", 0)
	ss.Stats.SetInt("TrlRespIdx", 0)
	ss.Stats.SetInt("TrlDecRespIdx", 0)
	ss.Stats.SetFloat("TrlDecErr", 0.0)
	ss.Stats.SetFloat("TrlDecErr2", 0.0)
	ev := ss.Envs[etime.Train.String()].(*ImagesEnv)
	ss.Stats.Confusion.InitFromLabels(ev.Images.Cats, 12)
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdt.Text
func (ss *Sim) StatCounters(di int) {
	ctx := &ss.Context
	mode := ctx.Mode
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	ss.Stats.SetInt("Cycle", int(ss.Context.Cycle))
	trl := ss.Stats.Int("Trial")
	ss.Stats.SetInt("Trial", trl+di)
	ss.Stats.SetInt("Di", di)
	ss.Stats.SetString("TrialName", ss.Stats.StringDi("TrialName", di))
	ss.Stats.SetString("TrlResp", ss.Stats.StringDi("TrlResp", di))
}

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.ViewUpdt.View == nil {
		return
	}
	di := ss.ViewUpdt.View.Di
	if tm == etime.Trial {
		ss.TrialStats(di) // get trial stats for current di
	}
	ss.StatCounters(di)
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Di", "TrlCat", "TrlResp", "TrialName", "Cycle", "UnitErr", "TrlErr", "CorSim"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats(di int) {
	ctx := &ss.Context
	out := ss.Net.AxonLayerByName("Output")

	ss.Stats.SetFloat32("CorSim", out.Vals[di].CorSim.Cor)
	ss.Stats.SetFloat("UnitErr", out.PctUnitErr(ctx)[di])

	ovt := ss.Stats.SetLayerTensor(ss.Net, "Output", "ActM", di)
	ev := ss.Envs.ByMode(ctx.Mode).(*ImagesEnv)

	ncats := len(ev.Images.Cats)

	curCatIdx := ss.Stats.IntDi("TrlCatIdx", di)
	curCat := ss.Stats.StringDi("TrlCat", di)
	ss.Stats.SetInt("TrlCatIdx", curCatIdx)
	ss.Stats.SetString("TrlCat", curCat)

	rsp, trlErr, trlErr2 := ev.OutErr(ovt, curCatIdx)
	ss.Stats.SetIntDi("TrlRespIdx", di, rsp) // save for stat counter
	ss.Stats.SetFloatDi("TrlErr", di, trlErr)
	ss.Stats.SetFloatDi("TrlErr2", di, trlErr2)
	ss.Stats.SetInt("TrlRespIdx", rsp) // used in logging current trial
	ss.Stats.SetFloat("TrlErr", trlErr)
	ss.Stats.SetFloat("TrlErr2", trlErr2)
	if rsp >= 0 && rsp < ncats {
		ss.Stats.SetStringDi("TrlResp", di, ev.Images.Cats[rsp])
		ss.Stats.SetString("TrlResp", ev.Images.Cats[rsp])
	} else {
		ss.Stats.SetStringDi("TrlResp", di, "none")
		ss.Stats.SetString("TrlResp", "none")
	}

	trnEpc := ss.Loops.GetLoop(etime.Train, etime.Epoch).Counter.Cur
	if trnEpc > ss.Config.Run.ConfusionEpc {
		ss.Stats.Confusion.Incr(curCatIdx, rsp)
	}

	ss.Stats.SetFloat("TrlTrgAct", float64(out.Pool(0, uint32(di)).AvgMax.Act.Plus.Avg/0.01))
	decIdx := ss.Decoder.Decode("ActM", di)
	ss.Stats.SetInt("TrlDecRespIdx", decIdx)
	if ctx.Mode == etime.Train {
		if ss.Config.Run.MPI {
			ss.Decoder.TrainMPI(curCatIdx)
		} else {
			ss.Decoder.Train(curCatIdx)
		}
	}
	decErr := float64(0)
	if decIdx != curCatIdx {
		decErr = 1
	}
	ss.Stats.SetFloat("TrlDecErr", decErr)
	decErr2 := decErr
	if ss.Decoder.Sorted[1] == curCatIdx {
		decErr2 = 0
	}
	ss.Stats.SetFloat("TrlDecErr2", decErr2)
	ss.Stats.SetFloat32("TrlOutRT", out.Vals[di].RT)
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.Trial, "Di")
	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrlCat", "TrialName", "TrlResp")

	ss.Logs.AddStatAggItem("CorSim", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("UnitErr", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()

	// Copy over Testing items
	ss.Logs.AddCopyFromFloatItems(etime.Train, []etime.Times{etime.Epoch, etime.Run}, etime.Test, etime.Epoch, "Tst", "CorSim", "UnitErr", "PctCor", "PctErr", "PctErr2", "DecErr", "DecErr2")

	ss.ConfigActRFs()

	axon.LogAddDiagnosticItems(&ss.Logs, ss.Net.LayersByType(axon.SuperLayer, axon.TargetLayer), etime.Train, etime.Epoch, etime.Trial)
	axon.LogAddPCAItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "TargetLayer")

	// this was useful during development of trace learning:
	// axon.LogAddCaLrnDiagnosticItems(&ss.Logs, ss.Net, etime.Epoch, etime.Trial)

	ss.Logs.PlotItems("CorSim", "PctErr", "PctErr2", "DecErr", "DecErr2")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
	ss.Logs.SetMeta(etime.Test, etime.Epoch, "Type", "Bar")
}

// ConfigLogItems specifies extra logging items
func (ss *Sim) ConfigLogItems() {
	ss.Logs.AddItem(&elog.Item{
		Name: "Err2",
		Type: etensor.FLOAT64,
		Plot: true,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlErr2")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "PctErr2",
		Type: etensor.FLOAT64,
		Plot: false,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
				ctx.SetAggItem(ctx.Mode, etime.Trial, "Err2", agg.AggMean)
			}, etime.Scope(etime.AllModes, etime.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, etime.Epoch, 5)
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	// DecErr decoding
	ss.Logs.AddItem(&elog.Item{
		Name: "DecResp",
		Type: etensor.STRING,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
				ctx.SetStatInt("TrlDecRespIdx")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "DecErr",
		Type: etensor.FLOAT64,
		Plot: elog.DTrue,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlDecErr")
			}, etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
			}, etime.Scope(etime.AllModes, etime.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, etime.Epoch, 5)
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "DecErr2",
		Type: etensor.FLOAT64,
		Plot: elog.DTrue,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlDecErr2")
			}, etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
			}, etime.Scope(etime.AllModes, etime.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, etime.Epoch, 5)
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})

	ss.Logs.AddItem(&elog.Item{
		Name:      "CatErr",
		Type:      etensor.FLOAT64,
		CellShape: []int{100},
		DimNames:  []string{"Cat"},
		Plot:      true,
		Range:     minmax.F64{Min: 0},
		TensorIdx: -1, // plot all values
		Write: elog.WriteMap{
			etime.Scope(etime.Test, etime.Epoch): func(ctx *elog.Context) {
				ix := ctx.Logs.IdxView(etime.Test, etime.Trial)
				spl := split.GroupBy(ix, []string{"TrlCat"})
				split.AggTry(spl, "Err", agg.AggMean)
				cats := spl.AggsToTable(etable.ColNameOnly)
				ss.Logs.MiscTables[ctx.Item.Name] = cats
				ctx.SetTensor(cats.Cols[1])
			}}})

	layers := ss.Net.LayersByType(axon.SuperLayer, axon.TargetLayer)
	for _, lnm := range layers {
		clnm := lnm
		ly := ss.Net.AxonLayerByName(clnm)
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + "_RT",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Trial): func(ctx *elog.Context) {
					fcyc := ly.LayerVals(uint32(ctx.Di)).RT
					ctx.SetFloat32(fcyc)
				}, etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + "_AvgCaDiff",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Trial): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "CaDiff")
					avg := tsragg.Mean(tsr)
					ctx.SetFloat64(avg)
				}, etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_Gnmda",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Trial): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "Gnmda")
					avg := tsragg.Mean(tsr)
					ctx.SetFloat64(avg)
				}, etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_GgabaB",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Trial): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "GgabaB")
					avg := tsragg.Mean(tsr)
					ctx.SetFloat64(avg)
				}, etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_SSGi",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Trial): func(ctx *elog.Context) {
					ctx.SetFloat32(ly.Pools[0].Inhib.SSGi)
				}, etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
				}}})

	}
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	ctx := &ss.Context
	if mode != etime.Analyze {
		ctx.Mode = mode // Also set specifically in a Loop callback.
	}

	if ss.Config.Run.MPI && time == etime.Epoch { // gather data from trial level
		ss.Logs.MPIGatherTableRows(mode, etime.Trial, ss.Comm)
	}

	dt := ss.Logs.Table(mode, time)
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		return
	case time == etime.Trial:
		for di := 0; di < int(ctx.NetIdxs.NData); di++ {
			ss.TrialStats(di)
			ss.StatCounters(di)
			ss.Logs.LogRowDi(mode, time, row, di)
		}
		return // don't do reg below
		// case time == etime.Epoch:
		// 	mpi.AllPrintf("Epoch trial dt rows: %d\n", ss.Logs.Table(mode, etime.Trial).Rows)
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc

	if time == etime.Epoch {
		trnEpc := ss.Loops.GetLoop(etime.Train, etime.Epoch).Counter.Cur
		if trnEpc > ss.Config.Run.ConfusionEpc && trnEpc%ss.Config.Run.ConfusionEpc == 0 {
			ss.Stats.Confusion.Probs()
			fnm := elog.LogFileName("trn_conf", ss.Net.Name(), ss.Stats.String("RunName"))
			ss.Stats.Confusion.SaveCSV(gi.FileName(fnm))
		}
	}
}

// ConfigActRFs
func (ss *Sim) ConfigActRFs() {
	ss.Stats.SetF32Tensor("Image", &ss.Envs[etime.Test.String()].(*ImagesEnv).Img.Tsr) // image used for actrfs, must be there first
	ss.Stats.InitActRFs(ss.Net, []string{"V4f16:Image", "V4f16:Output", "TEOf16:Image", "TEOf16:Output", "TEOf8:Image", "TEOf8:Output"}, "ActM")
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
	ss.ViewUpdt.Config(nv, etime.Trial, etime.Trial)
	ss.GUI.ViewUpdt = &ss.ViewUpdt
	ss.ConfigNetView(nv)

	ss.GUI.AddPlots(title, &ss.Logs)

	tg := ss.GUI.TabView.AddNewTab(etview.KiT_TensorGrid, "Image").(*etview.TensorGrid)
	tg.SetStretchMax()
	ss.GUI.SetGrid("Image", tg)
	tg.SetTensor(&ss.Envs[etime.Train.String()].(*ImagesEnv).Img.Tsr)

	ss.GUI.AddActRFGridTabs(&ss.Stats.ActRFs)

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddLooperCtrl(ss.Loops, []etime.Modes{etime.Train, etime.Test})

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
			ss.RndSeeds.NewSeeds()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "README",
		Icon:    "file-markdown",
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			gi.OpenURL("https://github.com/lvis/blob/main/sims/lvis_cu3d100_te16deg_axon/README.md")
		},
	})
	ss.GUI.FinalizeGUI(false)
	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUwithGUI(&ss.Context)
		gi.SetQuitCleanFunc(func() {
			ss.Net.GPU.Destroy()
		})
	}
	return ss.GUI.Win
}

func (ss *Sim) RunGUI() {
	ss.Init()
	win := ss.ConfigGui()
	win.StartEventLoop()
}

// These props register methods so they can be called through gui with arg prompts
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
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

func (ss *Sim) RunNoGUI() {
	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}
	if ss.Config.Log.SaveWts {
		mpi.Printf("Saving final weights per run\n")
	}
	runName := ss.Params.RunName(ss.Config.Run.Run)
	ss.Stats.SetString("RunName", runName) // used for naming logs, stats, etc
	netName := ss.Net.Name()

	if mpi.WorldRank() == 0 {
		elog.SetLogFile(&ss.Logs, ss.Config.Log.Epoch, etime.Train, etime.Epoch, "epc", netName, runName)
		elog.SetLogFile(&ss.Logs, ss.Config.Log.Run, etime.Train, etime.Run, "run", netName, runName)
		elog.SetLogFile(&ss.Logs, ss.Config.Log.TestEpoch, etime.Test, etime.Epoch, "tst_epc", netName, runName)
	}
	// Special cases for mpi per-node saving of trial data
	if ss.Config.Log.Trial {
		fnm := elog.LogFileName(fmt.Sprintf("trl_%d", mpi.WorldRank()), netName, runName)
		ss.Logs.SetLogFile(etime.Train, etime.Trial, fnm)
	}
	if ss.Config.Log.TestTrial {
		fnm := elog.LogFileName(fmt.Sprintf("tst_trl_%d", mpi.WorldRank()), netName, runName)
		ss.Logs.SetLogFile(etime.Test, etime.Trial, fnm)
	}

	netdata := ss.Config.Log.NetData
	if netdata {
		mpi.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	ss.Init()

	mpi.Printf("Running %d Runs starting at %d\n", ss.Config.Run.NRuns, ss.Config.Run.Run)
	ss.Loops.GetLoop(etime.Train, etime.Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.NRuns)

	if ss.Config.Run.GPU {
		if ss.Config.Run.MPI && ss.Config.Run.GPUSameNodeMPI {
			os.Setenv("VK_DEVICE_SELECT", fmt.Sprintf("%d", mpi.WorldRank()))
		}
		// expt with diff memory config:
		// ss.Context.SynapseCaVars.SetSynapseOuter(int(ss.Context.NetIdxs.MaxData))
		// ss.Net.Ctx.SynapseCaVars.SetSynapseOuter(int(ss.Context.NetIdxs.MaxData))
		ss.Net.ConfigGPUnoGUI(&ss.Context)
	}
	mpi.Printf("Set NThreads to: %d\n", ss.Net.NThreads)

	tmr := timer.Time{}
	tmr.Start()

	ss.Loops.Run(etime.Train)

	tmr.Stop()
	if ss.Config.Bench {
		tm := tmr.TotalSecs()
		ptmsec := (tm / float64(ss.Config.Run.NTrials)) * 1000
		// note: getting some variability across nodes here -- keeping this as all print
		mpi.AllPrintf("Total Time: %6.3g   Bench Per Trl Msec: %g\n", tm, ptmsec)
	} else {
		mpi.Printf("Total Time: %6.3g\n", tmr.TotalSecs())
	}
	ss.Net.TimerReport()

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}

	ss.Net.GPU.Destroy() // safe even if no GPU
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
		ss.Config.Run.MPI = false
	} else {
		mpi.Printf("MPI running on %d procs\n", mpi.WorldSize())
	}
}

// MPIFinalize finalizes MPI
func (ss *Sim) MPIFinalize() {
	if ss.Config.Run.MPI {
		mpi.Finalize()
	}
}

// MPIWtFmDWt updates weights from weight changes, using MPI to integrate
// DWt changes across parallel nodes, each of which are learning on different
// sequences of inputs.
func (ss *Sim) MPIWtFmDWt() {
	ctx := &ss.Context
	if ss.Config.Run.MPI {
		ss.Net.CollectDWts(ctx, &ss.AllDWts)
		ss.Comm.AllReduceF32(mpi.OpSum, ss.AllDWts, nil) // in place
		ss.Net.SetDWts(ctx, ss.AllDWts, mpi.WorldSize())
	}
	ss.Net.WtFmDWt(ctx)
}

func (ss *Sim) AssertMPIReplicaConsistency() {
	if ss.Comm.Size() == 1 {
		return
	}
	hash := ss.Net.WtsHash()
	orig := []uint8(hash)
	var MPIHashes []byte
	if ss.Comm.Rank() == 0 {
		MPIHashes = make([]byte, ss.Comm.Size()*len(orig))
	}
	err := ss.Comm.GatherU8(0, MPIHashes, orig)
	if err != nil {
		panic(err)
	}
	if ss.Comm.Rank() == 0 {
		for i := 0; i < ss.Comm.Size(); i++ {
			if string(MPIHashes[i*len(orig):(i+1)*len(orig)]) != hash {
				panic("Hashes do not match! The models on different nodes have diverged.")
			}
		}
	}
}
