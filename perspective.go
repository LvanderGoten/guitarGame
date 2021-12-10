package main

import "math"

type Camera struct {
	Position         [3]float64
	Alpha            float64
	Beta             float64
	Gamma            float64
	ScreenWidth      int
	ScreenHeight     int
	ProjectionMatrix [3][4]float64
	M                [3][3]float64
	Minv             [3][3]float64
	P4               [3]float64
	IntrinsicMatrix  [3][3]float64
	ExtrinsicMatrix  [3][4]float64
}

func NewCamera(position [3]float64, alpha float64, beta float64, gamma float64) *Camera {
	camera := new(Camera)
	camera.Position = position
	camera.Alpha = alpha
	camera.Beta = beta
	camera.Gamma = gamma
	camera.ScreenWidth = ScreenWidth
	camera.ScreenHeight = ScreenHeight
	camera.ProjectionMatrix = getCameraMatrix(alpha, beta, gamma, position)
	camera.M = subset3x4Matrix(camera.ProjectionMatrix)
	camera.Minv = invertUpperTriangularMatrix3x3(camera.M)
	camera.P4 = subsetVector3x4(camera.ProjectionMatrix, 3)
	camera.IntrinsicMatrix = getIntrinsicMatrix()
	camera.ExtrinsicMatrix = getExtrinsicMatrix(alpha, beta, gamma, position)

	return camera
}

func getIntrinsicMatrix() [3][3]float64 {
	return [3][3]float64{
		{DistanceToCameraPlane * PixelSize, 0.0, float64(ScreenWidth) / 2.0},
		{0.0, DistanceToCameraPlane * PixelSize, float64(ScreenHeight) / 2.0},
		{0.0, 0.0, 1.0},
	}
}

func getYawMatrix(alpha float64) [3][3]float64 {
	return [3][3]float64{
		{math.Cos(alpha), -math.Sin(alpha), 0.0},
		{math.Sin(alpha), math.Cos(alpha), 0.0},
		{0.0, 0.0, 1.0},
	}
}

func getPitchMatrix(beta float64) [3][3]float64 {
	return [3][3]float64{
		{math.Cos(beta), 0.0, math.Sin(beta)},
		{0.0, 1.0, 0.0},
		{-math.Sin(beta), 0.0, math.Cos(beta)},
	}
}

func getRollMatrix(gamma float64) [3][3]float64 {
	return [3][3]float64{
		{1.0, 0.0, 0.0},
		{0.0, math.Cos(gamma), -math.Sin(gamma)},
		{0.0, math.Sin(gamma), math.Cos(gamma)},
	}
}

func getCameraRotationMatrix(alpha float64, beta float64, gamma float64) [3][3]float64 {
	yawMatrix := getYawMatrix(alpha)
	pitchMatrix := getPitchMatrix(beta)
	rollMatrix := getRollMatrix(gamma)

	return matrixMatrixProduct3x3(matrixMatrixProduct3x3(yawMatrix, pitchMatrix), rollMatrix)
}

func getExtrinsicMatrix(alpha float64, beta float64, gamma float64, cameraPosition [3]float64) [3][4]float64 {
	R := getCameraRotationMatrix(alpha, beta, gamma)
	t := scaleByScalar(matrixVectorProduct3x3(R, cameraPosition), -1.0)

	return [3][4]float64{
		{R[0][0], R[0][1], R[0][2], t[0]},
		{R[1][0], R[1][1], R[1][2], t[1]},
		{R[2][0], R[2][1], R[2][2], t[2]},
	}
}

func getCameraMatrix(alpha float64, beta float64, gamma float64, cameraPosition [3]float64) [3][4]float64 {
	intrinsicMatrix := getIntrinsicMatrix()
	extrinsicMatrix := getExtrinsicMatrix(alpha, beta, gamma, cameraPosition)

	return matrixMatrixProduct3x4(intrinsicMatrix, extrinsicMatrix)

}
