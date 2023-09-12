// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/prjn"

// EnvConfig has config params for environment
// note: only adding fields for key Env params that matter for both Network and Env
// other params are set via the Env map data mechanism.
type EnvConfig struct {

	// env parameters -- can set any field/subfield on Env struct, using standard TOML formatting
	Env map[string]any `desc:"env parameters -- can set any field/subfield on Env struct, using standard TOML formatting"`

	// other option: "images/CU3D_100_plus_renders", ImageFile = "cu3d100plus"
	// works somewhat worse

	// [def: images/CU3D_100_renders_lr20_u30_nb] path for the images
	Path string `def:"images/CU3D_100_renders_lr20_u30_nb" desc:"path for the images"`

	// [def: cu3d100old] file with list of images
	ImageFile string `def:"cu3d100old" desc:"file with list of images"`

	// [def: 5] number of units per localist output unit
	NOutPer int `def:"5" desc:"number of units per localist output unit"`

	// [def: false] if true, use random output patterns -- else localist
	RndOutPats bool `def:"false" desc:"if true, use random output patterns -- else localist"`
}

// ParamConfig has config parameters related to sim params
type ParamConfig struct {

	// network parameters
	Network map[string]any `desc:"network parameters"`

	// Extra Param Sheet name(s) to use (space separated if multiple) -- must be valid name as listed in compiled-in params or loaded params
	Sheet string `desc:"Extra Param Sheet name(s) to use (space separated if multiple) -- must be valid name as listed in compiled-in params or loaded params"`

	// extra tag to add to file names and logs saved from this run
	Tag string `desc:"extra tag to add to file names and logs saved from this run"`

	// user note -- describe the run params etc -- like a git commit message for the run
	Note string `desc:"user note -- describe the run params etc -- like a git commit message for the run"`

	// Name of the JSON file to input saved parameters from.
	File string `nest:"+" desc:"Name of the JSON file to input saved parameters from."`

	// [def: true] if true, organize layers and connectivity with 2x2 sub-pools within each topological pool
	SubPools bool `def:"true" desc:"if true, organize layers and connectivity with 2x2 sub-pools within each topological pool"`

	// Save a snapshot of all current param and config settings in a directory named params_<datestamp> (or _good if Good is true), then quit -- useful for comparing to later changes and seeing multiple views of current params
	SaveAll bool `nest:"+" desc:"Save a snapshot of all current param and config settings in a directory named params_<datestamp> (or _good if Good is true), then quit -- useful for comparing to later changes and seeing multiple views of current params"`

	// for SaveAll, save to params_good for a known good params state.  This can be done prior to making a new release after all tests are passing -- add results to git to provide a full diff record of all params over time.
	Good bool `nest:"+" desc:"for SaveAll, save to params_good for a known good params state.  This can be done prior to making a new release after all tests are passing -- add results to git to provide a full diff record of all params over time."`
}

// RunConfig has config parameters related to running the sim
type RunConfig struct {

	// use MPI message passing interface for data parallel computation between nodes running identical copies of the same sim, sharing DWt changes
	MPI bool `desc:"use MPI message passing interface for data parallel computation between nodes running identical copies of the same sim, sharing DWt changes"`

	// [def: true] use the GPU for computation -- generally faster even for small models if NData ~16
	GPU bool `def:"true" desc:"use the GPU for computation -- generally faster even for small models if NData ~16"`

	// [def: true] if true and both MPI and GPU are being used, this selects a different GPU for each MPI proc rank, assuming a multi-GPU node -- set to false if running MPI across multiple GPU nodes
	GPUSameNodeMPI bool `def:"true" desc:"if true and both MPI and GPU are being used, this selects a different GPU for each MPI proc rank, assuming a multi-GPU node -- set to false if running MPI across multiple GPU nodes"`

	// [def: 16] [min: 1] number of data-parallel items to process in parallel per trial -- works (and is significantly faster) for both CPU and GPU.  Results in an effective mini-batch of learning.
	NData int `def:"16" min:"1" desc:"number of data-parallel items to process in parallel per trial -- works (and is significantly faster) for both CPU and GPU.  Results in an effective mini-batch of learning."`

	// [def: 0] number of parallel threads for CPU computation -- 0 = use default
	NThreads int `def:"0" desc:"number of parallel threads for CPU computation -- 0 = use default"`

	// [def: 0] starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1
	Run int `def:"0" desc:"starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1"`

	// [def: 1] [min: 1] total number of runs to do when running Train
	NRuns int `def:"1" min:"1" desc:"total number of runs to do when running Train"`

	// [def: 500] total number of epochs per run -- mostly asymptotes at 1,000 with small continued improvements out to 2,000.  500 is fine for most purposes
	NEpochs int `def:"500" desc:"total number of epochs per run -- mostly asymptotes at 1,000 with small continued improvements out to 2,000.  500 is fine for most purposes"`

	// [def: 512] total number of trials per epoch.  Should be an even multiple of NData.
	NTrials int `def:"512" desc:"total number of trials per epoch.  Should be an even multiple of NData."`

	// [def: 10] how frequently (in epochs) to compute PCA on hidden representations to measure variance?
	PCAInterval int `def:"10" desc:"how frequently (in epochs) to compute PCA on hidden representations to measure variance?"`

	// [def: 500] epoch to start recording confusion matrix
	ConfusionEpc int `def:"500" desc:"epoch to start recording confusion matrix"`

	// [def: 20] how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing
	TestInterval int `def:"20" desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
}

// LogConfig has config parameters related to logging data
type LogConfig struct {

	// if true, save final weights after each run
	SaveWts bool `desc:"if true, save final weights after each run"`

	// [def: true] if true, save train epoch log to file, as .epc.tsv typically
	Epoch bool `def:"true" nest:"+" desc:"if true, save train epoch log to file, as .epc.tsv typically"`

	// [def: false] if true, save run log to file, as .run.tsv typically
	Run bool `def:"false" nest:"+" desc:"if true, save run log to file, as .run.tsv typically"`

	// [def: false] if true, save train trial log to file, as .trl.tsv typically. May be large.
	Trial bool `def:"false" nest:"+" desc:"if true, save train trial log to file, as .trl.tsv typically. May be large."`

	// [def: false] if true, save testing epoch log to file, as .tst_epc.tsv typically.  In general it is better to copy testing items over to the training epoch log and record there.
	TestEpoch bool `def:"false" nest:"+" desc:"if true, save testing epoch log to file, as .tst_epc.tsv typically.  In general it is better to copy testing items over to the training epoch log and record there."`

	// [def: false] if true, save testing trial log to file, as .tst_trl.tsv typically. May be large.
	TestTrial bool `def:"false" nest:"+" desc:"if true, save testing trial log to file, as .tst_trl.tsv typically. May be large."`

	// if true, save network activation etc data from testing trials, for later viewing in netview
	NetData bool `desc:"if true, save network activation etc data from testing trials, for later viewing in netview"`
}

// Config is a standard Sim config -- use as a starting point.
type Config struct {

	// specify include files here, and after configuration, it contains list of include files added
	Includes []string `desc:"specify include files here, and after configuration, it contains list of include files added"`

	// [def: true] open the GUI -- does not automatically run -- if false, then runs automatically and quits
	GUI bool `def:"true" desc:"open the GUI -- does not automatically run -- if false, then runs automatically and quits"`

	// log debugging information
	Debug bool `desc:"log debugging information"`

	// run a standard benchmarking configuration: runs 64 trials (512 for MPI which can run more data parallel) for 1 epoch and reports timing
	Bench bool `desc:"run a standard benchmarking configuration: runs 64 trials (512 for MPI which can run more data parallel) for 1 epoch and reports timing "`

	// [view: add-fields] environment configuration options
	Env EnvConfig `view:"add-fields" desc:"environment configuration options"`

	// [view: add-fields] parameter related configuration options
	Params ParamConfig `view:"add-fields" desc:"parameter related configuration options"`

	// [view: add-fields] sim running related configuration options
	Run RunConfig `view:"add-fields" desc:"sim running related configuration options"`

	// [view: add-fields] data logging related configuration options
	Log LogConfig `view:"add-fields" desc:"data logging related configuration options"`
}

func (cfg *Config) IncludesPtr() *[]string { return &cfg.Includes }

func (cfg *Config) Defaults() {
}

//////////////////////////////////////////////////////////////////////////////
//   Prjns

// note: clutters args to put all these in config

// Prjns holds all the special projections
type Prjns struct {

	// Standard feedforward topographic projection, recv = 1/2 send size
	Prjn4x4Skp2 *prjn.PoolTile `nest:"+" desc:"Standard feedforward topographic projection, recv = 1/2 send size"`

	// Reciprocal
	Prjn4x4Skp2Recip *prjn.PoolTile `nest:"+" desc:"Reciprocal"`

	// Standard feedforward topographic projection, recv = 1/2 send size
	Prjn4x4Skp2Sub2 *prjn.PoolTileSub `nest:"+" desc:"Standard feedforward topographic projection, recv = 1/2 send size"`

	// Reciprocal
	Prjn4x4Skp2Sub2Recip *prjn.PoolTileSub `nest:"+" desc:"Reciprocal"`

	// Standard feedforward topographic projection, recv = 1/2 send size
	Prjn4x4Skp2Sub2Send *prjn.PoolTileSub `nest:"+" desc:"Standard feedforward topographic projection, recv = 1/2 send size"`

	// Standard feedforward topographic projection, recv = 1/2 send size
	Prjn4x4Skp2Sub2SendRecip *prjn.PoolTileSub `nest:"+" desc:"Standard feedforward topographic projection, recv = 1/2 send size"`

	// same-size prjn
	Prjn2x2Skp1 *prjn.PoolTile `nest:"+" desc:"same-size prjn"`

	// same-size prjn reciprocal
	Prjn2x2Skp1Recip *prjn.PoolTile `nest:"+" desc:"same-size prjn reciprocal"`

	// same-size prjn
	Prjn2x2Skp1Sub2 *prjn.PoolTileSub `nest:"+" desc:"same-size prjn"`

	// same-size prjn reciprocal
	Prjn2x2Skp1Sub2Recip *prjn.PoolTileSub `nest:"+" desc:"same-size prjn reciprocal"`

	// same-size prjn
	Prjn2x2Skp1Sub2Send *prjn.PoolTileSub `nest:"+" desc:"same-size prjn"`

	// same-size prjn reciprocal
	Prjn2x2Skp1Sub2SendRecip *prjn.PoolTileSub `nest:"+" desc:"same-size prjn reciprocal"`

	// lateral inhib projection
	Prjn2x2Skp2 *prjn.PoolTileSub `nest:"+" desc:"lateral inhib projection"`

	// for V4 <-> TEO
	Prjn4x4Skp0 *prjn.PoolTile `nest:"+" desc:"for V4 <-> TEO"`

	// for V4 <-> TEO
	Prjn4x4Skp0Recip *prjn.PoolTile `nest:"+" desc:"for V4 <-> TEO"`

	// for V4 <-> TEO
	Prjn4x4Skp0Sub2 *prjn.PoolTileSub `nest:"+" desc:"for V4 <-> TEO"`

	// for V4 <-> TEO
	Prjn4x4Skp0Sub2Recip *prjn.PoolTileSub `nest:"+" desc:"for V4 <-> TEO"`

	// for TE <-> TEO
	Prjn1x1Skp0 *prjn.PoolTile `nest:"+" desc:"for TE <-> TEO"`

	// for TE <-> TEO
	Prjn1x1Skp0Recip *prjn.PoolTile `nest:"+" desc:"for TE <-> TEO"`

	// lateral inhibitory connectivity for subpools
	Prjn6x6Skp2Lat *prjn.PoolTileSub `nest:"+" desc:"lateral inhibitory connectivity for subpools"`
}

func (pj *Prjns) Defaults() {
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
