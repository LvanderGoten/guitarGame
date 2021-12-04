package main

import (
	"fmt"
	"github.com/fogleman/gg"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
)

const (
	NumStrings                 int     = 6
	NumFrets                   int     = 25
	MinOctave                  int     = 2
	MaxOctave                  int     = 6
	CylinderRadius             float64 = 1.0
	CylinderHeight             float64 = 5.0
	CylinderNumRotationAngles  int     = 32
	CylinderNumHeightDivisions int     = 100
	ScreenWidth                int     = 512
	ScreenHeight               int     = 512
	DistanceToCameraPlane      float64 = 20.0
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

func getLightingDirection() [3]float64 {
	return [3]float64{1.0 / math.Sqrt(2.0), 1.0 / math.Sqrt(2.0), 0.0}
}

func getIntrinsicMatrix() [3][3]float64 {
	return [3][3]float64{
		{DistanceToCameraPlane, 0.0, float64(ScreenWidth) / 2.0},
		{0.0, DistanceToCameraPlane, float64(ScreenHeight) / 2.0},
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

func getCameraRotationMatrix(alpha float64, beta float64, gamma float64) [3][3]float64 {
	yawMatrix := getYawMatrix(alpha)
	pitchMatrix := getPitchMatrix(beta)
	rollMatrix := getRollMatrix(gamma)

	return matrixMatrixProduct3x3(matrixMatrixProduct3x3(yawMatrix, pitchMatrix), rollMatrix)
}

func getExtrinsicMatrix(alpha float64, beta float64, gamma float64, cameraPosition Vector3d) [3][4]float64 {
	R := getCameraRotationMatrix(alpha, beta, gamma)
	t := matrixVectorProduct3x3(R, vector3dToArray(scaleByScalar(cameraPosition, -1.0)))

	return [3][4]float64{
		{R[0][0], R[0][1], R[0][2], t[0]},
		{R[1][0], R[1][1], R[1][2], t[1]},
		{R[2][0], R[2][1], R[2][2], t[2]},
	}
}

func getCameraMatrix(alpha float64, beta float64, gamma float64, cameraPosition Vector3d) [3][4]float64 {
	intrinsicMatrix := getIntrinsicMatrix()
	extrinsicMatrix := getExtrinsicMatrix(alpha, beta, gamma, cameraPosition)

	return matrixMatrixProduct3x4(intrinsicMatrix, extrinsicMatrix)

}

type Vertex struct {
	id     int
	coord  Vector3d
	normal Vector3d
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
	faces    []Face
}

type Camera struct {
	position Vector3d
	alpha    float64
	beta     float64
	gamma    float64
}

func vector3dToArray(v Vector3d) [3]float64 {
	return [3]float64{v.x, v.y, v.z}
}

func plus(v1 Vector3d, v2 Vector3d) Vector3d {
	return Vector3d{v1.x + v2.x, v1.y + v2.y, v1.z + v2.z}
}

func minus(v1 Vector3d, v2 Vector3d) Vector3d {
	return Vector3d{v1.x - v2.x, v1.y - v2.y, v1.z - v2.z}
}

func dotProduct(v1 Vector3d, v2 Vector3d) float64 {
	return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z
}

func scaleByScalar(v Vector3d, s float64) Vector3d {
	return Vector3d{v.x * s, v.y * s, v.z * s}
}

func crossProduct(v1 Vector3d, v2 Vector3d) Vector3d {
	return Vector3d{
		v1.y*v2.z - v1.z*v2.y,
		v1.z*v2.x - v1.x*v2.z,
		v1.x*v2.y - v1.y*v2.x,
	}
}

func euclideanNorm(v Vector3d) float64 {
	return math.Sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
}

func matrixVectorProduct(A [][]float64, b []float64) []float64 {
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
		for j := 0; j < p; j++ {
			for k := 0; k < n; k++ {
				result[i][j] += A[i][k] * B[k][j]
			}
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

func transposeMatrix3x4(A [3][4]float64) [4][3]float64 {
	var result [4][3]float64
	for i := 0; i < 4; i++ {
		for j := 0; j < 3; j++ {
			result[i][j] = A[j][i]
		}
	}
	return result
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
		meanCoord = plus(scaleByScalar(meanCoord, i/(i+1)), scaleByScalar(vertex.coord, 1/(i+1)))
	}
	return meanCoord
}

func getCylinder(x0 float64, y0 float64) *Cylinder {
	var cylinder *Cylinder
	cylinder = new(Cylinder)

	zInc := CylinderHeight / float64(CylinderNumHeightDivisions-1)
	phiInc := (2.0 * math.Pi) / float64(CylinderNumRotationAngles-1)

	cylinder.vertices = make([]Vertex, CylinderNumHeightDivisions*CylinderNumRotationAngles+2)

	// Vertices
	for i := 0; i < CylinderNumHeightDivisions; i++ {
		z := float64(i) * zInc

		for j := 0; j < CylinderNumRotationAngles; j++ {
			phi := float64(j) * phiInc

			x := x0 + CylinderRadius*math.Cos(phi)
			y := y0 + CylinderRadius*math.Sin(phi)
			z := z

			nx := math.Cos(phi)
			ny := math.Sin(phi)
			nz := 0.0

			cylinder.vertices[i*CylinderNumRotationAngles+j] = Vertex{i*CylinderNumRotationAngles + j, Vector3d{x, y, z}, Vector3d{nx, ny, nz}}
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
	vcbi := (CylinderNumHeightDivisions-1)*CylinderNumRotationAngles + CylinderNumRotationAngles
	vcb := Vertex{id: vcbi, coord: Vector3d{x0, y0, 0}}
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
	vcti := (CylinderNumHeightDivisions-1)*CylinderNumRotationAngles + CylinderNumRotationAngles + 1
	vct := Vertex{id: vcti, coord: Vector3d{x0, y0, CylinderHeight}}
	cylinder.vertices[vcti] = vct
	for j := 0; j < CylinderNumRotationAngles; j++ {
		v1i := (CylinderNumHeightDivisions-1)*CylinderNumRotationAngles + j
		v2i := v1i - v1i%CylinderNumRotationAngles + (v1i+1)%CylinderNumRotationAngles

		v1 := &cylinder.vertices[v1i]
		v2 := &cylinder.vertices[v2i]
		v3 := &vct

		n := computeNormal(v1, v2, v3)
		face := Face{v1, v2, v3, n}
		cylinder.faces = append(cylinder.faces, face)
	}

	return cylinder
}

func gatherCoordinatesFromTriangleMesh(cylinder *Cylinder) [][3][4]float64 {
	numFaces := len(cylinder.faces)
	result := make([][3][4]float64, numFaces)
	var faceMatrix [3][4]float64

	for i, face := range cylinder.faces {
		a1 := vector3dToArray(face.v1.coord)
		a2 := vector3dToArray(face.v2.coord)
		a3 := vector3dToArray(face.v3.coord)

		copy(faceMatrix[0][:], append(a1[:], 1.0))
		copy(faceMatrix[1][:], append(a2[:], 1.0))
		copy(faceMatrix[2][:], append(a3[:], 1.0))

		result[i] = faceMatrix
	}

	return result
}

func gatherNormalsFromTriangleMesh(cylinder *Cylinder) [][3][4]float64 {
	numFaces := len(cylinder.faces)
	result := make([][3][4]float64, numFaces)
	var faceMatrix [3][4]float64

	for i, face := range cylinder.faces {
		a1 := vector3dToArray(face.v1.normal)
		a2 := vector3dToArray(face.v2.normal)
		a3 := vector3dToArray(face.v3.normal)

		copy(faceMatrix[0][:], append(a1[:], 1.0))
		copy(faceMatrix[1][:], append(a2[:], 1.0))
		copy(faceMatrix[2][:], append(a3[:], 1.0))

		result[i] = faceMatrix
	}

	return result
}

func flattenVector3d(arr [][3][4]float64) [][4]float64 {
	numFaces := len(arr)
	result := make([][4]float64, 3*numFaces)

	for i := 0; i < numFaces; i += 3 {
		result[i] = arr[i][0]
		result[i+1] = arr[i][1]
		result[i+2] = arr[i][2]
	}

	return result
}

func worldCoordinatesToImageCoordinates(vertexWorldCoords [][4]float64, cameraMatrix [3][4]float64) [][3]float64 {
	cameraMatrixTransposed := transposeMatrix3x4(cameraMatrix)
	vertexImageCoords := matrixMatrixProduct4x3(vertexWorldCoords, cameraMatrixTransposed)

	return vertexImageCoords
}

func normalsToLighting(normals [][3]float64) []float64 {
	numNormals := len(normals)
	intensity := make([]float64, numNormals)
	lightingDirection := getLightingDirection()
	for i, normal := range normals {
		intensity[i] = math.Abs(normal[0]*lightingDirection[0] + normal[1]*lightingDirection[1] + normal[2]*lightingDirection[2])
	}
	return intensity
}

func zBuffer(vertexImageCoords [][3]float64, vertexLightingIntensity []float64) [ScreenWidth][ScreenHeight]float64 {
	var image [ScreenWidth][ScreenHeight]float64

	numVertices := len(vertexImageCoords)

	var depth [ScreenWidth][ScreenHeight]float64
	for i := 0; i < ScreenWidth; i++ {
		for j := 0; j < ScreenHeight; j++ {
			depth[i][j] = math.Inf(1)
		}
	}

	for i := 0; i < numVertices; i++ {
		x := int(math.Round(vertexImageCoords[i][0]))
		y := int(math.Round(vertexImageCoords[i][1]))

		if (x < 0 || x >= ScreenWidth) || (y < 0 || y >= ScreenHeight) {
			continue
		}

		z := vertexImageCoords[i][2]

		if z < depth[x][y] {
			intensity := vertexLightingIntensity[i]
			image[x][y] = intensity
			fmt.Printf("%d %d %f %f\n", x, y, z, intensity)
			depth[x][y] = z
		}

	}
	return image

}

func imageTo8bit(img [ScreenWidth][ScreenHeight]float64) [ScreenWidth][ScreenHeight]uint8 {
	var result [ScreenWidth][ScreenHeight]uint8
	for i := 0; i < ScreenWidth; i++ {
		for j := 0; j < ScreenHeight; j++ {
			result[i][j] = uint8(255.0 * img[i][j])
		}
	}
	return result
}

func save8bitImage(img [ScreenWidth][ScreenHeight]uint8) {
	rgbImg := image.NewRGBA(image.Rect(0, 0, ScreenWidth, ScreenHeight))
	for i := 0; i < ScreenWidth; i++ {
		for j := 0; j < ScreenHeight; j++ {
			rgbImg.Set(i, j, color.RGBA{
				R: img[i][j],
				G: img[i][j],
				B: img[i][j],
				A: 255,
			})
		}
	}
	out, _ := os.Create("test.png")
	png.Encode(out, rgbImg)
	out.Close()
}

func homogeneousToInhomogeneous(vertexCoords [][4]float64) [][3]float64 {
	numCoords := len(vertexCoords)
	result := make([][3]float64, numCoords)
	for i, coord := range vertexCoords {
		result[i][0] = coord[0]
		result[i][1] = coord[1]
		result[i][2] = coord[2]
	}
	return result
}

func rasterizeLevelToImage(camera *Camera, cylinder *Cylinder, fileName string) {
	cameraMatrix := getCameraMatrix(camera.alpha, camera.beta, camera.gamma, camera.position)

	vertexWorldCoords := gatherCoordinatesFromTriangleMesh(cylinder)
	vertexWorldCoordsFlat := flattenVector3d(vertexWorldCoords)
	vertexImageCoords := worldCoordinatesToImageCoordinates(vertexWorldCoordsFlat, cameraMatrix)

	vertexWorldNormals := gatherNormalsFromTriangleMesh(cylinder)
	vertexWorldNormalsFlat := flattenVector3d(vertexWorldNormals)
	vertexNormals := homogeneousToInhomogeneous(vertexWorldNormalsFlat)

	vertexLightingIntensity := normalsToLighting(vertexNormals)

	image := zBuffer(vertexImageCoords, vertexLightingIntensity)
	image8bit := imageTo8bit(image)
	save8bitImage(image8bit)

	fmt.Println(len(image8bit))

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
		_, err := f.WriteString(fmt.Sprintf("f %d %d %d\n", face.v1.id+1, face.v2.id+1, face.v3.id+1))
		if err != nil {
			return
		}
	}

}

func main() {
	cylinder := getCylinder(0.0, 0.0)
	writeToObjFile(cylinder, "test.obj")
	camera := Camera{Vector3d{0, 0, 0}, 0.5, 0.25, 0.1}
	rasterizeLevelToImage(&camera, cylinder, "test.png")
}
