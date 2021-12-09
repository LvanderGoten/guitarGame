package main

import (
	"math"
)

type Vertex struct {
	id     int
	coord  [3]float64
	normal [3]float64
}

type Face struct {
	v1 *Vertex
	v2 *Vertex
	v3 *Vertex

	n [3]float64
}

type Cylinder struct {
	vertices []Vertex
	faces    []Face
}


func getCylinder(x0 float64, y0 float64) *Cylinder {
	var cylinder *Cylinder
	cylinder = new(Cylinder)

	zInc := CylinderHeight / float64(CylinderNumHeightDivisions-1)
	phiInc := (2.0 * math.Pi) / float64(CylinderNumRotationAngles)

	cylinder.vertices = make([]Vertex, CylinderNumHeightDivisions*CylinderNumRotationAngles + 2)

	// Vertices
	for i := 0; i < CylinderNumHeightDivisions; i++ {
		z := float64(i) * zInc

		for j := 0; j < CylinderNumRotationAngles; j++ {
			phi := float64(j) * phiInc

			x := x0 + CylinderRadius*math.Cos(phi)
			y := y0 + CylinderRadius*math.Sin(phi)

			nx := math.Cos(phi)
			ny := math.Sin(phi)
			nz := 0.0

			cylinder.vertices[i*CylinderNumRotationAngles+j] = Vertex{i*CylinderNumRotationAngles + j, [3]float64{x, y, z}, [3]float64{nx, ny, nz}}
		}
	}

	// Faces (triangulation)
	cylinder.faces = make([]Face, 0)
	for i := 0; i < CylinderNumHeightDivisions-1; i++ {
		for j := 0; j < CylinderNumRotationAngles; j++ {

			v1i := i*CylinderNumRotationAngles + j
			v2i := v1i - v1i%CylinderNumRotationAngles + (v1i+1)%CylinderNumRotationAngles
			v3i := v1i + CylinderNumRotationAngles
			v4i := v3i - v3i%CylinderNumRotationAngles + (v3i+1)%CylinderNumRotationAngles

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
	vcbi := CylinderNumHeightDivisions * CylinderNumRotationAngles
	vcb := Vertex{id: vcbi, coord: [3]float64{x0, y0, 0}}
	cylinder.vertices[vcbi] = vcb
	for j := 0; j < CylinderNumRotationAngles; j++ {
		v1i := j
		v2i := (v1i + 1) % CylinderNumRotationAngles

		v1 := &cylinder.vertices[v1i]
		v2 := &cylinder.vertices[v2i]
		v3 := &cylinder.vertices[vcbi]

		n := computeNormal(v1, v2, v3)
		face := Face{v1, v2, v3, n}
		cylinder.faces = append(cylinder.faces, face)
	}

	// Faces top (triangulation)
	vcti := CylinderNumHeightDivisions * CylinderNumRotationAngles + 1
	vct := Vertex{id: vcti, coord: [3]float64{x0, y0, CylinderHeight}}
	cylinder.vertices[vcti] = vct
	for j := 0; j < CylinderNumRotationAngles; j++ {
		v1i := (CylinderNumHeightDivisions-1)*CylinderNumRotationAngles + j
		v2i := v1i - v1i%CylinderNumRotationAngles + (v1i+1)%CylinderNumRotationAngles

		v1 := &cylinder.vertices[v1i]
		v2 := &cylinder.vertices[v2i]
		v3 := &cylinder.vertices[vcti]

		n := computeNormal(v1, v2, v3)
		face := Face{v1, v2, v3, n}
		cylinder.faces = append(cylinder.faces, face)
	}

	return cylinder
}
