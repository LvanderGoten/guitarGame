package main

import (
	"fmt"
	"github.com/fogleman/gg"
	"math"
	"os"
)

const (
	NumStrings int = 6
	NumFrets int = 25
	MinOctave int = 2
	MaxOctave int = 6
	CylinderRadius float64 = 1.0
	CylinderHeight float64 = 5.0
	CylinderNumRotationAngles int = 32
	CylinderNumHeightDivisions int = 100
	ScreenWidth int = 512
	ScreenHeight int = 512
	DistanceToCameraPlane float64 = 20.0
)

func getOpenStringNotes() [6]string {
	return [6]string{"E", "B", "G", "D", "A", "E"}
}

func getOpenStringOctaves() [6]int {
	return [6]int{4, 3, 3, 3, 2, 2}
}

func getCanonicalNotes() [12]string {
	return [12]string{
		"C", "C#", "D", "D#",
		"E", "F", "F#", "G",
		"G#", "A", "A#", "B"}
}

func getIntrinsicMatrix() [3][4]float64 {
	return [3][4]float64{
		{0, 1, 2, 3} ,
		{4, 5, 6, 7} ,
		{8, 9, 10, 11}
	}
}

type Vertex struct {
	id int
	coord Vector3d
}

type Face struct {
	v1 *Vertex
	v2 *Vertex
	v3 *Vertex

	n Vector3d
}

type Vector3d struct {
	x float64
	y float64
	z float64
}

type Cylinder struct {
	vertices []Vertex
	faces []Face
}

type Camera struct {
	position Vector3d
	lookAt Vector3d
}

func plus(v1 Vector3d, v2 Vector3d) Vector3d {
	return Vector3d{v1.x + v2.x, v1.y + v2.y, v1.z + v2.z}
}

func minus(v1 Vector3d, v2 Vector3d) Vector3d {
	return Vector3d{v1.x - v2.x, v1.y - v2.y, v1.z - v2.z}
}

func dotProduct(v1 Vector3d, v2 Vector3d) float64 {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
}

func scaleByScalar(v Vector3d, s float64) Vector3d {
	return Vector3d{v.x * s, v.y * s, v.z * s}
}

func crossProduct(v1 Vector3d, v2 Vector3d) Vector3d {
	return Vector3d{
		v1.y * v2.z - v1.z * v2.y,
		v1.z * v2.x - v1.x * v2.z,
		v1.x * v2.y - v1.y * v2.x,
	}
}

func euclideanNorm(v Vector3d) float64 {
	return math.Sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
}

func matrixVectorProduct(A [][]float64, b[]float64) []float64 {
	if len(A[0]) != len(b) {
		panic("Matrix-vector product is ill-defined!")
	}
	m := len(A)
	n := len(A[0])

	result := make([]float64, m)

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			result[i] += A[i][j] * b[j]
		}
	}

	return result
}

func matrixMatrixProduct(A [][]float64, B [][]float64) [][]float64 {
	if len(A[0]) != len(B) {
		panic("Matrix-matrix product is ill-defined!")
	}
	m := len(A)
	n := len(B)
	p := len(B[0])

	result := make([][]float64, m)

	for i := 0; i < m; i++ {
		result[i] = make([]float64, p)
		for j := 0; j < p ; j++ {
			for k := 0; k < n; k++ {
				result[i][j] += A[i][k] * B[k][j]
			}
		}
	}
}

func computeNormal(v1 *Vertex, v2 *Vertex, v3 *Vertex) Vector3d {
	p1 := minus(v2.coord, v1.coord)
	p2 := minus(v3.coord, v1.coord)
	d := crossProduct(p1, p2)
	dNorm := euclideanNorm(d)
	return scaleByScalar(d, 1.0/dNorm)
}

func computeMeanCoordinate(cylinder *Cylinder) Vector3d {
	meanCoord := Vector3d{0.0, 0.0, 0.0}
	for i, vertex := range cylinder.vertices {
		i := float64(i)
		meanCoord = plus(scaleByScalar(meanCoord, i/(i + 1)), scaleByScalar(vertex.coord, 1/(i + 1)))
	}
	return meanCoord
}

func getCylinder(x0 float64, y0 float64) *Cylinder {
	var cylinder *Cylinder
	cylinder = new(Cylinder)

	zInc := CylinderHeight / float64(CylinderNumHeightDivisions - 1)
	phiInc := (2.0 * math.Pi) / float64(CylinderNumRotationAngles - 1)

	cylinder.vertices = make([]Vertex, CylinderNumHeightDivisions * CylinderNumRotationAngles + 2)

	// Vertices
	for i := 0; i < CylinderNumHeightDivisions; i++ {
		z := float64(i) * zInc

		for j := 0; j < CylinderNumRotationAngles; j++ {
			phi := float64(j) * phiInc

			x := x0 + CylinderRadius * math.Cos(phi)
			y := y0 + CylinderRadius * math.Sin(phi)
			z := z

			cylinder.vertices[i * CylinderNumRotationAngles + j] = Vertex{i * CylinderNumRotationAngles + j, Vector3d{x, y, z}}
		}
	}

	// Faces (triangulation)
	cylinder.faces = make([]Face, 0)
	for i := 0; i < CylinderNumHeightDivisions - 1; i++ {
		for j := 0; j < CylinderNumRotationAngles; j++ {

			v1i := i * CylinderNumRotationAngles + j
			v2i := v1i - v1i % CylinderNumRotationAngles + (v1i + 1) % CylinderNumRotationAngles
			v3i := v1i + CylinderNumRotationAngles
			v4i := v3i - v3i % CylinderNumRotationAngles + (v3i + 1) % CylinderNumRotationAngles

			v1 := &cylinder.vertices[v1i]
			v2 := &cylinder.vertices[v2i]
			v3 := &cylinder.vertices[v3i]
			v4 := &cylinder.vertices[v4i]

			nAlpha := computeNormal(v1, v2, v3)
			nBeta := computeNormal(v2, v3, v4)
			faceAlpha := Face{v1, v2, v3, nAlpha}
			faceBeta := Face{v2, v3, v4, nBeta}
			cylinder.faces = append(cylinder.faces, faceAlpha, faceBeta)
		}
	}

	// Faces bottom (triangulation)
	vcbi := (CylinderNumHeightDivisions - 1) * CylinderNumRotationAngles + CylinderNumRotationAngles
	vcb := Vertex{ id: vcbi, coord: Vector3d{x0, y0, 0}}
	cylinder.vertices[vcbi] = vcb
	for j := 0; j < CylinderNumRotationAngles; j++ {
		v1i := j
		v2i := (v1i + 1) % CylinderNumRotationAngles

		v1 := &cylinder.vertices[v1i]
		v2 := &cylinder.vertices[v2i]
		v3 := &vcb

		n := computeNormal(v1, v2, v3)
		face := Face{v1, v2, v3, n}
		cylinder.faces = append(cylinder.faces, face)
	}

	// Faces top (triangulation)
	vcti := (CylinderNumHeightDivisions - 1) * CylinderNumRotationAngles + CylinderNumRotationAngles + 1
	vct := Vertex{ id: vcti, coord: Vector3d{x0, y0, CylinderHeight}}
	cylinder.vertices[vcti] = vct
	for j := 0; j < CylinderNumRotationAngles; j++ {
		v1i := (CylinderNumHeightDivisions - 1) * CylinderNumRotationAngles + j
		v2i := v1i - v1i % CylinderNumRotationAngles + (v1i + 1) % CylinderNumRotationAngles

		v1 := &cylinder.vertices[v1i]
		v2 := &cylinder.vertices[v2i]
		v3 := &vct

		n := computeNormal(v1, v2, v3)
		face := Face{v1, v2, v3, n}
		cylinder.faces = append(cylinder.faces, face)
	}

	return cylinder
}

func raytraceLevelToImage(camera *Camera, cylinder *Cylinder, fileName string) {
	// var image [ScreenWidth][ScreenHeight]float64

	dc := gg.NewContext(ScreenWidth, ScreenWidth)
	dc.SetRGB(1, 1, 1)
	dc.Clear()
	dc.SetRGB(0, 0, 0)
	dc.DrawPoint(100, 100, 1)
	dc.Fill()
	dc.SavePNG(fileName)
}

func writeToObjFile(cylinder *Cylinder, fileName string) {
	f, err := os.Create(fileName)
	if err != nil {
		fmt.Println(err)
		return
	}

	for _, vertex := range cylinder.vertices {
		_, err := f.WriteString(fmt.Sprintf("v %.3f %.3f %.3f\n", vertex.coord.x, vertex.coord.y, vertex.coord.z))
		if err != nil {
			return
		}
	}

	for _, face := range cylinder.faces {
		_, err := f.WriteString(fmt.Sprintf("f %d %d %d\n", face.v1.id + 1, face.v2.id + 1, face.v3.id + 1))
		if err != nil {
			return
		}
	}

}

func main() {
	fmt.Println("Hello World")
	cylinder := getCylinder(0.0, 0.0)
	writeToObjFile(cylinder, "test.obj")
	cameraPosition := Vector3d{0, 0, 0}
	lookAt := computeMeanCoordinate(cylinder)
	camera := Camera{cameraPosition, lookAt}
	raytraceLevelToImage(&camera, cylinder, "test.png")
}
