// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"fmt"
	"image"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/empi/empi"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
	"github.com/goki/gi/gi"
	"github.com/goki/ki/ints"
	"github.com/goki/mat32"
	"golang.org/x/image/draw"
	"golang.org/x/image/math/f64"
)

// ImagesEnv provides the rendered results of the Obj3D + Saccade generator.
type ImagesEnv struct {
	Nm         string          `desc:"name of this environment"`
	Dsc        string          `desc:"description of this environment"`
	Test       bool            `desc:"present test items, else train"`
	Images     Images          `desc:"images list"`
	TransMax   mat32.Vec2      `desc:"def 0.3 maximum amount of translation as proportion of half-width size in each direction -- 1 = something in center is now at right edge"`
	TransSigma float32         `def:"0.15" desc:"if > 0, generate translations using gaussian normal distribution with this standard deviation, and then clip to TransMax range -- this facilitates learning on the central region while still giving exposure to wider area.  Tyically turn off for last 100 epochs to measure true uniform distribution performance."`
	ScaleRange minmax.F32      `desc:"def 0.5 - 1.1 range of scale"`
	RotateMax  float32         `def:"8" desc:"def 8 maximum degrees of rotation in plane -- image is rotated plus or minus in this range"`
	V1m16      Vis             `desc:"v1 16deg medium resolution filtering of image -- V1AllTsr has result"`
	V1h16      Vis             `desc:"v1 16deg higher resolution filtering of image -- V1AllTsr has result"`
	V1m8       Vis             `desc:"v1 8deg medium resolution filtering of image -- V1AllTsr has result"`
	V1h8       Vis             `desc:"v1 8deg higher resolution filtering of image -- V1AllTsr has result"`
	Output     etensor.Float32 `desc:"output category"`
	StRow      int             `desc:"starting row, e.g., for mpi allocation across processors"`
	EdRow      int             `desc:"ending row -- if 0 it is ignored"`
	Order      []int           `desc:"order of images to present"`
	Run        env.Ctr         `view:"inline" desc:"current run of model as provided during Init"`
	Epoch      env.Ctr         `view:"inline" desc:"arbitrary aggregation of trials, for stats etc"`
	Trial      env.Ctr         `view:"inline" desc:"each object trajectory is one trial"`
	Row        env.Ctr         `view:"inline" desc:"row of item list  -- this is actual counter driving everything"`
	CurCat     string          `desc:"current category"`
	CurCatIdx  int             `desc:"index of current category"`
	CurImg     string          `desc:"current image"`
	CurTrans   mat32.Vec2      `desc:"current translation"`
	CurScale   float32         `desc:"current scaling"`
	CurRot     float32         `desc:"current rotation"`

	Image image.Image `view:"-" desc:"rendered image as loaded"`
}

func (ev *ImagesEnv) Name() string { return ev.Nm }
func (ev *ImagesEnv) Desc() string { return ev.Dsc }

func (ev *ImagesEnv) Validate() error {
	return nil
}

func (ev *ImagesEnv) Defaults() {
	ev.TransSigma = 0.15
	ev.TransMax.Set(0.3, 0.3)
	ev.ScaleRange.Set(0.4, 1.0)
	ev.RotateMax = 8
	ev.V1m16.Defaults(24, 8)
	ev.V1h16.Defaults(12, 4)
	ev.V1m8.Defaults(12, 4)
	ev.V1m8.V1sGeom.Border = image.Point{38, 38}
	ev.V1h8.Defaults(6, 2)
	ev.V1h8.V1sGeom.Border = image.Point{38, 38}
}

// ImageList returns the list of images -- train or test
func (ev *ImagesEnv) ImageList() []string {
	if ev.Test {
		return ev.Images.FlatTest
	}
	return ev.Images.FlatTrain
}

// MPIAlloc allocate objects based on mpi processor number
func (ev *ImagesEnv) MPIAlloc() {
	nim := len(ev.ImageList())
	ev.StRow, ev.EdRow, _ = empi.AllocN(nim)
	mpi.PrintAllProcs = true
	mpi.Printf("allocated images: n: %d st: %d ed: %d\n", nim, ev.StRow, ev.EdRow)
	mpi.PrintAllProcs = false
}

func (ev *ImagesEnv) Init(run int) {
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Row.Scale = env.Tick
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.Row.Cur = -1 // init state -- key so that first Step() = 0
	nitm := len(ev.ImageList())
	if ev.EdRow > 0 {
		ev.EdRow = ints.MinInt(ev.EdRow, nitm)
		nr := ev.EdRow - ev.StRow
		ev.Order = make([]int, nr)
		for i := 0; i < nr; i++ {
			ev.Order[i] = ev.StRow + i
		}
		erand.PermuteInts(ev.Order)
		ev.Row.Max = nr
	} else {
		ev.Row.Max = nitm
		ev.Order = rand.Perm(ev.Row.Max)
	}
	ev.Output.SetShape([]int{len(ev.Images.Cats)}, nil, nil)
}

// SaveListJSON saves flat string list to a JSON-formatted file.
func SaveListJSON(list []string, filename string) error {
	b, err := json.MarshalIndent(list, "", "  ")
	if err != nil {
		log.Println(err) // unlikely
		return err
	}
	err = ioutil.WriteFile(string(filename), b, 0644)
	if err != nil {
		log.Println(err)
	}
	return err
}

// OpenListJSON opens flat string list from a JSON-formatted file.
func OpenListJSON(list *[]string, filename string) error {
	b, err := ioutil.ReadFile(string(filename))
	if err != nil {
		log.Println(err)
		return err
	}
	return json.Unmarshal(b, list)
}

// SaveList2JSON saves double-string list to a JSON-formatted file.
func SaveList2JSON(list [][]string, filename string) error {
	b, err := json.MarshalIndent(list, "", "  ")
	if err != nil {
		log.Println(err) // unlikely
		return err
	}
	err = ioutil.WriteFile(string(filename), b, 0644)
	if err != nil {
		log.Println(err)
	}
	return err
}

// OpenList2JSON opens double-string list from a JSON-formatted file.
func OpenList2JSON(list *[][]string, filename string) error {
	b, err := ioutil.ReadFile(string(filename))
	if err != nil {
		log.Println(err)
		return err
	}
	return json.Unmarshal(b, list)
}

// OpenConfig opens saved configuration for current images
func (ev *ImagesEnv) OpenConfig() bool {
	cfnm := fmt.Sprintf("%s_cats.json", ev.Nm)
	tsfnm := fmt.Sprintf("%s_ntest%d_tst.json", ev.Nm, ev.Images.NTestPerCat)
	trfnm := fmt.Sprintf("%s_ntest%d_trn.json", ev.Nm, ev.Images.NTestPerCat)
	_, err := os.Stat(tsfnm)
	if !os.IsNotExist(err) {
		OpenListJSON(&ev.Images.Cats, cfnm)
		OpenList2JSON(&ev.Images.ImagesTest, tsfnm)
		OpenList2JSON(&ev.Images.ImagesTrain, trfnm)
		ev.Images.Flats()
		return true
	}
	return false
}

// SaveConfig saves configuration for current images
func (ev *ImagesEnv) SaveConfig() {
	cfnm := fmt.Sprintf("%s_cats.json", ev.Nm)
	tsfnm := fmt.Sprintf("%s_ntest%d_tst.json", ev.Nm, ev.Images.NTestPerCat)
	trfnm := fmt.Sprintf("%s_ntest%d_trn.json", ev.Nm, ev.Images.NTestPerCat)
	SaveListJSON(ev.Images.Cats, cfnm)
	SaveList2JSON(ev.Images.ImagesTest, tsfnm)
	SaveList2JSON(ev.Images.ImagesTrain, trfnm)
}

// CurImage returns current image based on row and
func (ev *ImagesEnv) CurImage() string {
	il := ev.ImageList()
	sz := len(ev.Order)
	if ev.Row.Cur >= sz {
		ev.Row.Max = sz
		ev.Row.Cur = 0
		erand.PermuteInts(ev.Order)
	}
	r := ev.Row.Cur
	if r < 0 {
		r = 0
	}
	i := ev.Order[r]
	ev.CurImg = il[i]
	ev.CurCat = ev.Images.Cat(ev.CurImg)
	ev.CurCatIdx = ev.Images.CatMap[ev.CurCat]
	return ev.CurImg
}

// OpenImage opens current image
func (ev *ImagesEnv) OpenImage() error {
	img := ev.CurImage()
	fnm := filepath.Join(ev.Images.Path, img)
	var err error
	ev.Image, err = gi.OpenImage(fnm)
	if err != nil {
		log.Println(err)
	}
	return err
}

// RandTransforms generates random transforms
func (ev *ImagesEnv) RandTransforms() {
	if ev.TransSigma > 0 {
		ev.CurTrans.X = float32(erand.Gauss(float64(ev.TransSigma), -1))
		ev.CurTrans.X = mat32.Clamp(ev.CurTrans.X, -ev.TransMax.X, ev.TransMax.X)
		ev.CurTrans.Y = float32(erand.Gauss(float64(ev.TransSigma), -1))
		ev.CurTrans.Y = mat32.Clamp(ev.CurTrans.Y, -ev.TransMax.Y, ev.TransMax.Y)
	} else {
		ev.CurTrans.X = (rand.Float32()*2 - 1) * ev.TransMax.X
		ev.CurTrans.Y = (rand.Float32()*2 - 1) * ev.TransMax.Y
	}
	ev.CurScale = ev.ScaleRange.Min + ev.ScaleRange.Range()*rand.Float32()
	ev.CurRot = (rand.Float32()*2 - 1) * ev.RotateMax
}

// TransformImage transforms the image according to current translation and scaling
func (ev *ImagesEnv) TransformImage() {
	s := mat32.NewVec2FmPoint(ev.Image.Bounds().Size())
	transformer := draw.BiLinear
	tx := 0.5 * ev.CurTrans.X * s.X
	ty := 0.5 * ev.CurTrans.Y * s.Y
	m := mat32.Translate2D(s.X*.5+tx, s.Y*.5+ty).Scale(ev.CurScale, ev.CurScale).Rotate(mat32.DegToRad(ev.CurRot)).Translate(-s.X*.5, -s.Y*.5)
	s2d := f64.Aff3{float64(m.XX), float64(m.XY), float64(m.X0), float64(m.YX), float64(m.YY), float64(m.Y0)}

	// use first color in upper left as fill color
	clr := ev.Image.At(0, 0)
	dst := image.NewRGBA(ev.Image.Bounds())
	src := image.NewUniform(clr)
	draw.Draw(dst, dst.Bounds(), src, image.ZP, draw.Src)

	transformer.Transform(dst, s2d, ev.Image, ev.Image.Bounds(), draw.Over, nil) // Over superimposes over bg
	ev.Image = dst
}

// FilterImage opens and filters current image
func (ev *ImagesEnv) FilterImage() error {
	err := ev.OpenImage()
	if err != nil {
		fmt.Println(err)
		return err
	}
	ev.TransformImage()
	tsz := ev.V1m16.ImgSize
	isz := ev.Image.Bounds().Size()
	if isz != tsz {
		ev.Image = transform.Resize(ev.Image, tsz.X, tsz.Y, transform.Linear)
	}
	ev.V1m16.Filter(ev.Image)
	ev.V1h16.Filter(ev.Image)
	ev.V1m8.Filter(ev.Image)
	ev.V1h8.Filter(ev.Image)
	return nil
}

// SetOutput sets output by category
func (ev *ImagesEnv) SetOutput() {
	ev.Output.SetZeros()
	ev.Output.Set1D(ev.CurCatIdx, 1)
}

func (ev *ImagesEnv) String() string {
	return fmt.Sprintf("%s:%s_%d", ev.CurCat, ev.CurImg, ev.Trial.Cur)
}

func (ev *ImagesEnv) Step() bool {
	ev.Epoch.Same() // good idea to just reset all non-inner-most counters at start
	ev.Row.Incr()   // auto-rotates
	if ev.Trial.Incr() {
		ev.Epoch.Incr()
	}
	ev.RandTransforms()
	ev.FilterImage()
	ev.SetOutput()
	return true
}

func (ev *ImagesEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return -1, -1, false
}

func (ev *ImagesEnv) State(element string) etensor.Tensor {
	switch element {
	case "V1m16":
		return &ev.V1m16.V1AllTsr
	case "V1h16":
		return &ev.V1h16.V1AllTsr
	case "V1m8":
		return &ev.V1m8.V1AllTsr
	case "V1h8":
		return &ev.V1h8.V1AllTsr
	case "Output":
		return &ev.Output
	}
	return nil
}

func (ev *ImagesEnv) Action(element string, input etensor.Tensor) {
	// nop
}

// Compile-time check that implements Env interface
var _ env.Env = (*ImagesEnv)(nil)
