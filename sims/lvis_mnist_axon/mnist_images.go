// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/gzip"
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"os"
	"path/filepath"
)

// DImages implements management of the MNIST digit dataset using original gzipped files.
// See: http://yann.lecun.com/exdb/mnist/ for format and other info
// The name of an image is batch_index with index in %05d format, batch 0 = train, 1 = test
// We load all into memory at the start.
type DImages struct {
	Path        string         `desc:"path to image files -- this should point to a directory that has the standard MNIST gz binary files"`
	ImgSize     int            `def:"28" desc:"size of image (assumed square)"`
	Cats        []string       `desc:"list of image categories"`
	CatMap      map[string]int `desc:"map of categories to indexes in Cats list"`
	ImagesAll   [][]string     `desc:"full list of images, organized by category and then filename (batch_index)"`
	ImagesTrain [][]string     `desc:"list of training images, organized by category and then filename"`
	ImagesTest  [][]string     `desc:"list of testing images, organized by category and then filename"`
	FlatAll     []string       `desc:"flat list of all images, as cat/filename.ext -- Flats() makes from above"`
	FlatTrain   []string       `desc:"flat list of all training images, as cat/filename.ext -- Flats() makes from above"`
	FlatTest    []string       `desc:"flat list of all testing images, as cat/filename.ext -- Flats() makes from above"`
	ImgBins     [][]byte       `view:"-" desc:"binary data for the images, skipping headers, read directly from the files -- first index is train / test index (0/1)"`
}

// SetPath sets path where binary files live
func (im *DImages) SetPath(path string) {
	im.Path = path
}

// OpenPath opens binary files at given path
func (im *DImages) OpenPath(path string) error {
	im.SetPath(path)
	files := []string{"train-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"}
	im.ImgBins = make([][]byte, len(files))
	for i, fl := range files {
		fn := filepath.Join(im.Path, fl)
		file, err := os.Open(fn)
		if err != nil {
			fmt.Printf("MNIST image file failed to open: %s\n", err)
			return err
		}
		gzr, err := gzip.NewReader(file)
		if err != nil {
			fmt.Printf("MNIST image gzip failed to open: %s\n", err)
			file.Close()
			return err
		}
		b, _ := ioutil.ReadAll(gzr) // todo: 1.16 should be io
		im.ImgBins[i] = b[16:]      // data starts at 16
		gzr.Close()
		file.Close()
	}
	im.InitCats()
	im.ReadNames()
	return nil
}

func (im *DImages) MakeCatMap() {
	nc := len(im.Cats)
	im.CatMap = make(map[string]int, nc)
	for ci, c := range im.Cats {
		im.CatMap[c] = ci
	}
}

// InitCats initializes categories
func (im *DImages) InitCats() {
	im.Cats = make([]string, 10)
	for i := 0; i < 10; i++ {
		im.Cats[i] = fmt.Sprintf("%d", i)
	}
	im.MakeCatMap()
}

// ReadNames iterates over ImgBins and extracts all files per cat
// Cats must already be made
func (im *DImages) ReadNames() error {
	ncats := len(im.Cats)
	im.ImagesAll = make([][]string, ncats)
	im.ImagesTrain = make([][]string, ncats)
	im.ImagesTest = make([][]string, ncats)

	if im.ImgSize == 0 {
		im.ImgSize = 28
	}

	files := []string{"train-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz"}

	imgsz := im.ImgSize * im.ImgSize

	for bi, d := range im.ImgBins {
		fl := files[bi]
		fn := filepath.Join(im.Path, fl)
		file, err := os.Open(fn)
		if err != nil {
			fmt.Printf("MNIST label file failed to open: %s\n", err)
			return err
		}
		gzr, err := gzip.NewReader(file)
		if err != nil {
			fmt.Printf("MNIST label gzip failed to open: %s\n", err)
			file.Close()
			return err
		}
		b, _ := ioutil.ReadAll(gzr) // todo: 1.16 should be io
		b = b[8:]                   // data starts at 8
		gzr.Close()
		file.Close()

		nimg := len(d) / imgsz
		for ii := 0; ii < nimg; ii++ {
			ci := b[ii]
			// cat := im.Cats[ci]
			fnm := fmt.Sprintf("%d_%05d", bi, ii)
			im.ImagesAll[ci] = append(im.ImagesAll[ci], fnm)
			if bi == 1 {
				im.ImagesTest[ci] = append(im.ImagesTest[ci], fnm)
			} else {
				im.ImagesTrain[ci] = append(im.ImagesTrain[ci], fnm)
			}
		}
	}
	im.Flats()
	return nil
}

// Image sets image bytes from given image name (batch_index) -- initializes image if not yet
func (im *DImages) Image(img *image.RGBA, fn string) (*image.RGBA, error) {
	var bi, ii int
	itnm := im.Item(fn)
	si, err := fmt.Sscanf(itnm, "%d_%05d", &bi, &ii)
	if si < 2 || err != nil {
		fmt.Printf("DImages Image name %s parsing error: %s\n", fn, err)
		return img, err
	}

	imgsz := im.ImgSize * im.ImgSize
	off := ii * imgsz
	if img == nil {
		img = image.NewRGBA(image.Rect(0, 0, im.ImgSize, im.ImgSize))
	} else {
		sz := img.Bounds().Size()
		if sz.X != im.ImgSize || sz.Y != im.ImgSize {
			img = image.NewRGBA(image.Rect(0, 0, im.ImgSize, im.ImgSize))
		}
	}
	for y := 0; y < im.ImgSize; y++ {
		for x := 0; x < im.ImgSize; x++ {
			pi := y*im.ImgSize + x
			v := 255 - im.ImgBins[bi][off+pi] // 0 = white, 255 = black
			clr := color.RGBA{v, v, v, 255}
			img.SetRGBA(x, y, clr)
		}
	}
	return img, nil
}

func (im *DImages) Cat(f string) string {
	dir, _ := filepath.Split(f)
	if len(dir) > 1 && dir[len(dir)-1] == '/' {
		dir = dir[:len(dir)-1]
	}
	return dir
}

func (im *DImages) Item(f string) string {
	_, itm := filepath.Split(f)
	return itm
}

// SelectCats filters the list of images to those within given list of categories.
func (im *DImages) SelectCats(cats []string) {
	nc := len(im.Cats)
	for ci := nc - 1; ci >= 0; ci-- {
		cat := im.Cats[ci]

		sel := false
		for _, cs := range cats {
			if cat == cs {
				sel = true
				break
			}
		}
		if !sel {
			im.Cats = append(im.Cats[:ci], im.Cats[ci+1:]...)
			im.ImagesAll = append(im.ImagesAll[:ci], im.ImagesAll[ci+1:]...)
			im.ImagesTrain = append(im.ImagesTrain[:ci], im.ImagesTrain[ci+1:]...)
			im.ImagesTest = append(im.ImagesTest[:ci], im.ImagesTest[ci+1:]...)
		}
	}
	im.MakeCatMap()
	im.Flats()
}

// DeleteCats filters the list of images to exclude those within given list of categories.
func (im *DImages) DeleteCats(cats []string) {
	nc := len(im.Cats)
	for ci := nc - 1; ci >= 0; ci-- {
		cat := im.Cats[ci]

		del := false
		for _, cs := range cats {
			if cat == cs {
				del = true
				break
			}
		}
		if del {
			im.Cats = append(im.Cats[:ci], im.Cats[ci+1:]...)
			im.ImagesAll = append(im.ImagesAll[:ci], im.ImagesAll[ci+1:]...)
			im.ImagesTrain = append(im.ImagesTrain[:ci], im.ImagesTrain[ci+1:]...)
			im.ImagesTest = append(im.ImagesTest[:ci], im.ImagesTest[ci+1:]...)
		}
	}
	im.MakeCatMap()
	im.Flats()
}

// Flats generates flat lists from categorized lists, in form categ/fname.obj
func (im *DImages) Flats() {
	im.FlatAll = im.FlatImpl(im.ImagesAll)
	im.FlatTrain = im.FlatImpl(im.ImagesTrain)
	im.FlatTest = im.FlatImpl(im.ImagesTest)
}

// FlatImpl generates flat lists from categorized lists, in form categ/fname.obj
func (im *DImages) FlatImpl(images [][]string) []string {
	var flat []string
	for ci, fls := range images {
		cat := im.Cats[ci]
		for _, fn := range fls {
			fn = cat + "/" + fn
			flat = append(flat, fn)
		}
	}
	return flat
}

// UnFlat translates FlatTrain, FlatTest into full nested lists -- Cats must
// also have already been loaded.  Call after loading FlatTrain, FlatTest
func (im *DImages) UnFlat() {
	nc := len(im.Cats)
	im.ImagesAll = make([][]string, nc)
	im.ImagesTrain = make([][]string, nc)
	im.ImagesTest = make([][]string, nc)

	im.MakeCatMap()

	for _, fn := range im.FlatTrain {
		cat := im.Cat(fn)
		ci := im.CatMap[cat]
		im.ImagesTrain[ci] = append(im.ImagesTrain[ci], fn)
		im.ImagesAll[ci] = append(im.ImagesAll[ci], fn)
	}
	for _, fn := range im.FlatTest {
		cat := im.Cat(fn)
		ci := im.CatMap[cat]
		im.ImagesTest[ci] = append(im.ImagesTest[ci], fn)
		im.ImagesAll[ci] = append(im.ImagesAll[ci], fn)
	}
	im.FlatAll = im.FlatImpl(im.ImagesAll)
}

// ToTrainAll compiles TrainAll from ImagesTrain, ImagesTest
func (im *DImages) ToTrainAll() {
	nc := len(im.Cats)
	im.ImagesAll = make([][]string, nc)

	im.MakeCatMap()

	for ci, fl := range im.ImagesTrain {
		im.ImagesAll[ci] = append(im.ImagesAll[ci], fl...)
	}
	for ci, fl := range im.ImagesTest {
		im.ImagesAll[ci] = append(im.ImagesAll[ci], fl...)
	}
}
