// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"os"
	"path/filepath"
)

// CImages implements management of the CIFAR-10 and -100 image datasets.
// See: https://www.cs.toronto.edu/~kriz/cifar.html for format and other info
// https://code.google.com/archive/p/cuda-convnet/wikis/Methodology.wiki has methodology
// e.g., use batches 1-4 for training, testing on batch 5, then do final test on batch-6.
// The name of an image is: batch_index where batch is 1-6 and index is 0-9,999 for index
// of image within the batch.  We use the binary format, loading all into memory at the start.
type CImages struct {
	Path        string         `desc:"path to image files -- this should point to a directory that has the standard CIFAR binary files"`
	ImgSize     int            `def:"32" desc:"size of image (assumed square)"`
	TestBatch   int            `desc:"batch number to use for testing -- must be set when reading -- 0 starting index"`
	Cats        []string       `desc:"list of image categories"`
	CatMap      map[string]int `desc:"map of categories to indexes in Cats list"`
	ImagesAll   [][]string     `desc:"full list of images, organized by category and then filename (batch_index)"`
	ImagesTrain [][]string     `desc:"list of training images, organized by category and then filename"`
	ImagesTest  [][]string     `desc:"list of testing images, organized by category and then filename"`
	FlatAll     []string       `desc:"flat list of all images, as cat/filename.ext -- Flats() makes from above"`
	FlatTrain   []string       `desc:"flat list of all training images, as cat/filename.ext -- Flats() makes from above"`
	FlatTest    []string       `desc:"flat list of all testing images, as cat/filename.ext -- Flats() makes from above"`
	ImgBins     [][]byte       `view:"-" desc:"binary data for the images, read directly from the files -- first index is batch number"`
}

// SetPath sets path where binary files live
func (im *CImages) SetPath(path string) {
	im.Path = path
}

// OpenPath opens binary files at given path
func (im *CImages) OpenPath(path string) error {
	im.SetPath(path)
	files := []string{"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin"}
	im.ImgBins = make([][]byte, len(files))
	for i, fl := range files {
		fn := filepath.Join(im.Path, fl)
		file, err := os.Open(fn)
		if err != nil {
			fmt.Printf("CIFAR image file failed to open: %s\n", err)
		} else {
			im.ImgBins[i], _ = ioutil.ReadAll(file) // todo: 1.16 should be io
			file.Close()
		}
	}
	im.OpenCats("batches.meta.txt")
	im.ReadNames()
	return nil
}

func (im *CImages) MakeCatMap() {
	nc := len(im.Cats)
	im.CatMap = make(map[string]int, nc)
	for ci, c := range im.Cats {
		im.CatMap[c] = ci
	}
}

// OpenCats reads categories from a file
func (im *CImages) OpenCats(fl string) error {
	fn := filepath.Join(im.Path, fl)
	file, err := os.Open(fn)
	if err != nil {
		fmt.Printf("CIFAR category meta file failed to open: %s\n", err)
		return err
	}
	defer file.Close()
	im.Cats = make([]string, 0)
	scan := bufio.NewScanner(file)
	for scan.Scan() {
		ct := scan.Text()
		if ct != "" {
			im.Cats = append(im.Cats, ct)
		}
	}
	im.MakeCatMap()
	return scan.Err()
}

// ReadNames iterates over ImgBins and extracts all files per cat
// Cats must already be made
func (im *CImages) ReadNames() error {
	ncats := len(im.Cats)
	im.ImagesAll = make([][]string, ncats)
	im.ImagesTrain = make([][]string, ncats)
	im.ImagesTest = make([][]string, ncats)

	if im.ImgSize == 0 {
		im.ImgSize = 32
	}

	imgcsz := im.ImgSize * im.ImgSize
	imgsz := imgcsz * 3 // rgb
	fimgsz := imgsz + 1 // full image size with cat label

	for bi, d := range im.ImgBins {
		nimg := len(d) / fimgsz
		for ii := 0; ii < nimg; ii++ {
			off := ii * fimgsz
			ci := d[off]
			// cat := im.Cats[ci]
			fnm := fmt.Sprintf("%d_%05d", bi, ii)
			im.ImagesAll[ci] = append(im.ImagesAll[ci], fnm)
			if bi == im.TestBatch {
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
func (im *CImages) Image(img *image.RGBA, fn string) (*image.RGBA, error) {
	var bi, ii int
	itnm := im.Item(fn)
	si, err := fmt.Sscanf(itnm, "%d_%05d", &bi, &ii)
	if si < 2 || err != nil {
		fmt.Printf("CImages Image name %s parsing error: %s\n", fn, err)
		return img, err
	}

	imgcsz := im.ImgSize * im.ImgSize
	imgsz := imgcsz * 3  // rgb
	fimgsz := imgsz + 1  // full image size with cat label
	off := ii*fimgsz + 1 // skip cat
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
			r := im.ImgBins[bi][off+pi]
			g := im.ImgBins[bi][off+imgcsz+pi]
			b := im.ImgBins[bi][off+2*imgcsz+pi]
			clr := color.RGBA{r, g, b, 255}
			img.SetRGBA(x, y, clr)
		}
	}
	return img, nil
}

func (im *CImages) Cat(f string) string {
	dir, _ := filepath.Split(f)
	if len(dir) > 1 && dir[len(dir)-1] == '/' {
		dir = dir[:len(dir)-1]
	}
	return dir
}

func (im *CImages) Item(f string) string {
	_, itm := filepath.Split(f)
	return itm
}

// SelectCats filters the list of images to those within given list of categories.
func (im *CImages) SelectCats(cats []string) {
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
func (im *CImages) DeleteCats(cats []string) {
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
func (im *CImages) Flats() {
	im.FlatAll = im.FlatImpl(im.ImagesAll)
	im.FlatTrain = im.FlatImpl(im.ImagesTrain)
	im.FlatTest = im.FlatImpl(im.ImagesTest)
}

// FlatImpl generates flat lists from categorized lists, in form categ/fname.obj
func (im *CImages) FlatImpl(images [][]string) []string {
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
func (im *CImages) UnFlat() {
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
func (im *CImages) ToTrainAll() {
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
