package main

import (
	"fmt"
	"math"
)

// No generics :(
func matrixMatrixProduct3x3(A [3][3]float64, B [3][3]float64) [3][3]float64 {
	var result [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 3; k++ {
				result[i][j] += A[i][k] * B[k][j]
			}
		}
	}
	return result
}

func matrixMatrixProduct3x4(A [3][3]float64, B [3][4]float64) [3][4]float64 {
	var result [3][4]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			for k := 0; k < 3; k++ {
				result[i][j] += A[i][k] * B[k][j]
			}
		}
	}
	return result
}

func matrixVectorProduct3x3(A [3][3]float64, b [3]float64) [3]float64 {
	var result [3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			result[i] += A[i][j] * b[j]
		}
	}
	return result
}

func matrixMatrixProduct4x3(A [][4]float64, B [4][3]float64) [][3]float64 {
	m := len(A)
	result := make([][3]float64, m)

	for i := 0; i < m; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 4; k++ {
				result[i][j] += A[i][k] * B[k][j]
			}
		}
	}

	return result
}

func plus(a [3]float64, b [3]float64) [3]float64 {
	return [3]float64{a[0] + b[0], a[1] + b[1], a[2] + b[2]}
}

func minus(a [3]float64, b [3]float64) [3]float64 {
	return [3]float64{a[0] - b[0], a[1] - b[1], a[2] - b[2]}
}

func dotProduct(a [3]float64, b [3]float64) float64 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

func scaleByScalar(v [3]float64, s float64) [3]float64 {
	return [3]float64{v[0] * s, v[1] * s, v[2] * s}
}

func crossProduct(a [3]float64, b [3]float64) [3]float64 {
	return [3]float64{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}

func euclideanNorm(v [3]float64) float64 {
	return math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
}

func printMatrix3x4(A [3][4]float64) {
	for _, row := range A {
		fmt.Print("[")
		for _, elem := range row {
			fmt.Printf("%.3f,", elem)
		}
		fmt.Println("]")
	}
}

func transposeMatrix3x3(A [3][3]float64) [3][3]float64 {
	var result [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			result[i][j] = A[j][i]
		}
	}
	return result
}

func subset3x4Matrix(A [3][4]float64) [3][3]float64 {
	var result [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			result[i][j] = A[i][j]
		}
	}
	return result
}

func subsetVector3x4(A [3][4]float64, j int) [3]float64 {
	var result [3]float64
	for i := 0; i < 3; i++ {
		result[i] = A[i][j]
	}
	return result
}

func invertUpperTriangularMatrix3x3(A [3][3]float64) [3][3]float64 {
	return [3][3]float64{
		{1 / A[0][0], -A[0][1] / (A[0][0] * A[1][1]), (A[0][1]*A[1][2] - A[0][2]*A[1][1]) / (A[0][0] * A[1][1] * A[2][2])},
		{0, 1 / A[1][1], -A[1][2] / (A[1][1] * A[2][2])},
		{0, 0, 1 / A[2][2]},
	}
}

func transposeMatrix3x4(A [3][4]float64) [4][3]float64 {
	var result [4][3]float64
	for i := 0; i < 4; i++ {
		for j := 0; j < 3; j++ {
			result[i][j] = A[j][i]
		}
	}
	return result
}

func computeNormal(v1 *Vertex, v2 *Vertex, v3 *Vertex) [3]float64 {
	p1 := minus(v2.coord, v1.coord)
	p2 := minus(v3.coord, v1.coord)
	d := crossProduct(p1, p2)
	dNorm := euclideanNorm(d)
	return scaleByScalar(d, 1.0/dNorm)
}

func computeMeanCoordinate(cylinder *Cylinder) [3]float64 {
	meanCoord := [3]float64{0.0, 0.0, 0.0}
	for i, vertex := range cylinder.vertices {
		i := float64(i)
		meanCoord = plus(scaleByScalar(meanCoord, i/(i+1)), scaleByScalar(vertex.coord, 1/(i+1)))
	}
	return meanCoord
}
