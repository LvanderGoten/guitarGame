package main

import (
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io/ioutil"
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
	CylinderNumRotationAngles  int     = 15
	CylinderNumHeightDivisions int     = 15
	ScreenWidth                int     = 128
	ScreenHeight               int     = 128
	DistanceToCameraPlane      float64 = 5.0
	PixelSize                  float64 = 100
	Pi2                        float64 = math.Pi / 2.0
	Pi4                        float64 = math.Pi / 4.0
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
	return [3]float64{1.0 / math.Sqrt(2.0), 0.0, 1.0 / math.Sqrt(2.0)}
}

func gatherCoordinatesFromTriangleMesh(cylinder *Cylinder) [][3][4]float64 {
	numFaces := len(cylinder.faces)
	result := make([][3][4]float64, numFaces)
	var faceMatrix [3][4]float64

	for i, face := range cylinder.faces {
		a1 := face.v1.coord
		a2 := face.v2.coord
		a3 := face.v3.coord

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
		a1 := face.v1.normal
		a2 := face.v2.normal
		a3 := face.v3.normal

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
	m := [3]float64{cameraMatrix[2][0], cameraMatrix[2][1], cameraMatrix[2][2]}
	mNorm := euclideanNorm(m)

	for i := 0; i < len(vertexImageCoords); i++ {
		vertexImageCoords[i][0] /= vertexImageCoords[i][2]
		vertexImageCoords[i][1] /= vertexImageCoords[i][2]
		vertexImageCoords[i][2] /= mNorm
	}

	return vertexImageCoords
}

func normalsToLighting(normals [][3]float64) []float64 {
	numNormals := len(normals)
	intensity := make([]float64, numNormals)
	lightingDirection := getLightingDirection()
	for i, normal := range normals {
		intensity[i] = (1.0 + normal[0]*lightingDirection[0] + normal[1]*lightingDirection[1] + normal[2]*lightingDirection[2]) / 2.0
	}
	return intensity
}

func zBuffer(vertexImageCoords [][3]float64, vertexLightingIntensity []float64) [ScreenWidth][ScreenHeight]float64 {
	var image [ScreenWidth][ScreenHeight]float64
	for i := 0; i < ScreenWidth; i++ {
		for j := 0; j < ScreenHeight; j++ {
			image[i][j] = 0.0
		}
	}

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

		if z >= 0 && z < depth[x][y] {
			intensity := vertexLightingIntensity[i]
			image[x][y] = intensity
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

func save8bitImage(img [ScreenWidth][ScreenHeight]uint8, fileName string) {
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
	out, _ := os.Create(fileName)
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

func computeRay(imgCoord [2]int, camera *Camera) [2][3]float64 {
	M := camera.M
	fmt.Println(len(M))
	var result [2][3]float64
	return result
}

func raytraceLevelToImage(camera *Camera, cylinder *Cylinder, fileName string) {
	//projectionMatrix := camera.ProjectionMatrix
	//faces := cylinder.faces

	// TODO: Check order
	var imgCoords [ScreenWidth * ScreenHeight][2]int
	for i := 0; i < ScreenWidth; i++ {
		for j := 0; j < ScreenHeight; j++ {
			k := i*ScreenHeight + j
			imgCoords[k][0] = i
			imgCoords[k][1] = j
		}
	}

	var rays [ScreenWidth * ScreenHeight][2][3]float64
	for i := 0; i < ScreenWidth; i++ {
		for j := 0; j < ScreenHeight; j++ {
			k := i*ScreenHeight + j

			rays[k] = computeRay(imgCoords[k], camera)
		}
	}
	fmt.Println()
}

func rasterizeLevelToImage(camera *Camera, cylinder *Cylinder, fileName string) {
	cameraMatrix := getCameraMatrix(camera.Alpha, camera.Beta, camera.Gamma, camera.Position)

	vertexWorldCoords := gatherCoordinatesFromTriangleMesh(cylinder)
	vertexWorldCoordsFlat := flattenVector3d(vertexWorldCoords)
	vertexImageCoords := worldCoordinatesToImageCoordinates(vertexWorldCoordsFlat, cameraMatrix)

	vertexWorldNormals := gatherNormalsFromTriangleMesh(cylinder)
	vertexWorldNormalsFlat := flattenVector3d(vertexWorldNormals)
	vertexNormals := homogeneousToInhomogeneous(vertexWorldNormalsFlat)

	vertexLightingIntensity := normalsToLighting(vertexNormals)

	image := zBuffer(vertexImageCoords, vertexLightingIntensity)
	image8bit := imageTo8bit(image)
	save8bitImage(image8bit, fileName)

	fmt.Println(len(image8bit))
}

func writeToObjFile(cylinder *Cylinder, camera *Camera, cylinderFileName string, cylinderSurfaceNormalsFname string, cameraFileName string) {
	fObj, err := os.Create(cylinderFileName)
	fCsv, err := os.Create(cylinderSurfaceNormalsFname)
	if err != nil {
		fmt.Println(err)
		return
	}

	for _, vertex := range cylinder.vertices {
		_, err := fObj.WriteString(fmt.Sprintf("v %.10f %.10f %.10f\n", vertex.coord[0], vertex.coord[1], vertex.coord[2]))
		if err != nil {
			return
		}
	}

	fCsv.WriteString(fmt.Sprintf("fx,fy,fz\n"))
	for _, face := range cylinder.faces {
		_, err := fObj.WriteString(fmt.Sprintf("f %d %d %d\n", face.v1.id+1, face.v2.id+1, face.v3.id+1))
		_, err = fCsv.WriteString(fmt.Sprintf("%f,%f,%f\n", face.n[0], face.n[1], face.n[2]))
		if err != nil {
			return
		}
	}

	dat, err := json.Marshal(*camera)
	err = ioutil.WriteFile(cameraFileName, dat, 0644)
}

func main() {
	cylinder := getCylinder(10.0, 10.0)
	camera := NewCamera([3]float64{0, 0, 2.5}, 0.0, -Pi4, Pi2)
	writeToObjFile(cylinder, camera, "cylinder.obj", "cylinderSurfaceNormals.csv", "camera.json")
	raytraceLevelToImage(camera, cylinder, "cylinder.png")
}
