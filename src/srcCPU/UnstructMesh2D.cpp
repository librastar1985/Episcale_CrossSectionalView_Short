/*
 * Mesh.cpp
 *
 *  Created on: Aug 20, 2014
 *      Author: wsun2
 */

#include "UnstructMesh2D.h"

namespace GEOMETRY {

UnstructMesh2D::UnstructMesh2D() {
}

std::vector<GEOMETRY::Point2D> UnstructMesh2D::outputTriangleCenters() {
	std::vector<GEOMETRY::Point2D> result;
	int triCount = triangles.size();
	for (int i = 0; i < triCount; i++) {
		Point2D center;
		center.setX(
				(points[triangles[i][0]].getX() + points[triangles[i][1]].getX()
						+ points[triangles[i][2]].getX()) / 3.0);
		center.setY(
				(points[triangles[i][0]].getY() + points[triangles[i][1]].getY()
						+ points[triangles[i][2]].getY()) / 3.0);
		result.push_back(center);
	}
	return result;
}

std::vector<GEOMETRY::Point2D> UnstructMesh2D::outputTriangleVerticies() {
	std::vector<GEOMETRY::Point2D> result;
	int triCount = triangles.size();
	for (int i = 0; i < triCount; i++) {
		Point2D center;
		center.setX(
				(points[triangles[i][0]].getX() + points[triangles[i][1]].getX()
						+ points[triangles[i][2]].getX()) / 3.0);
		center.setY(
				(points[triangles[i][0]].getY() + points[triangles[i][1]].getY()
						+ points[triangles[i][2]].getY()) / 3.0);
		result.push_back(center);
	}
	return result;
}

void UnstructMesh2D::setPointAsBdry(int index) {
	points[index].setIsOnBdry(true);
}

UnstructMesh2D::~UnstructMesh2D() {

}

} /* namespace GEOMETRY */

void GEOMETRY::UnstructMesh2D::insertVertex(const GEOMETRY::Point2D& point2D) {
	points.push_back(point2D);
}

void GEOMETRY::UnstructMesh2D::insertTriangle(
		const std::vector<int>& triangle) {
	triangles.push_back(triangle);
}

void GEOMETRY::UnstructMesh2D::insertEdge(const std::pair<int, int>& edge) {
	edges.push_back(edge);
}

void GEOMETRY::UnstructMesh2D::outputVtkFile(std::string outputFileName) {
	std::string vtkFileName = outputFileName + ".vtk";
	int totalNNum = points.size();

	std::ofstream fs;
	fs.open(vtkFileName.c_str());
	fs << "# vtk DataFile Version 3.0" << std::endl;
	fs << "Lines and points representing subcelluar element cells "
			<< std::endl;
	fs << "ASCII" << std::endl;
	fs << std::endl;
	fs << "DATASET UNSTRUCTURED_GRID" << std::endl;
	fs << "POINTS " << totalNNum << " float" << std::endl;
	for (int i = 0; i < totalNNum; i++) {
		fs << points[i].getX() << " " << points[i].getY() << " " << 0
				<< std::endl;
	}
	fs << std::endl;
	fs << "CELLS " << edges.size() << " " << 3 * edges.size() << std::endl;
	int linkSize = edges.size();
	for (int i = 0; i < linkSize; i++) {
		fs << 2 << " " << edges[i].first << " " << edges[i].second << std::endl;
	}
	fs << "CELL_TYPES " << linkSize << std::endl;
	for (int i = 0; i < linkSize; i++) {
		fs << "3" << std::endl;
	}
	fs << "POINT_DATA " << totalNNum << std::endl;
	fs << "SCALARS point_scalars float" << std::endl;
	fs << "LOOKUP_TABLE default" << std::endl;
	for (int i = 0; i < totalNNum; i++) {
		fs << points[i].isIsOnBdry() << std::endl;
	}
	fs.flush();
	fs.close();
}

std::vector<GEOMETRY::Point2D> GEOMETRY::UnstructMesh2D::getAllInsidePoints() {
	std::vector<GEOMETRY::Point2D> result;
	for (uint i = 0; i < points.size(); i++) {
		if (!points[i].isIsOnBdry()) {
			result.push_back(points[i]);
		}
	}
	return result;
}

std::vector<GEOMETRY::Point2D> GEOMETRY::UnstructMesh2D::getAllBdryPoints() {
	std::vector<GEOMETRY::Point2D> result;
	for (uint i = 0; i < points.size(); i++) {
		if (points[i].isIsOnBdry()) {
			result.push_back(points[i]);
		}
	}
	return result;
}

uint GEOMETRY::UnstructMesh2D::findIndexGivenPos(CVector pos) {
	assert(orderedBdryPts.size() > 0);
	uint index = 0;
	double minDis = 999999999;
	for (uint i = 0; i < orderedBdryPts.size(); i++) {
		CVector vec = CVector(orderedBdryPts[i].getX() - pos.GetX(),
				orderedBdryPts[i].getY() - pos.GetY(), 0);
		double dist = vec.getModul();
		if (dist < minDis) {
			index = i;
			minDis = dist;
		}
	}
	return index;
}

void GEOMETRY::UnstructMesh2D::generateFinalBdryAndProfilePoints(
		CVector posBegin, CVector posEnd) {
	uint index1 = findIndexGivenPos(posBegin);
	uint index2 = findIndexGivenPos(posEnd);
	finalBdryPts.insert(finalBdryPts.end(), orderedBdryPts.begin(),
			orderedBdryPts.begin() + index1);
	finalProfilePts.insert(finalProfilePts.end(),
			orderedBdryPts.begin() + index1,
			orderedBdryPts.begin() + index2 + 1);
	finalBdryPts.insert(finalBdryPts.end(), orderedBdryPts.begin() + index2 + 1,
			orderedBdryPts.end());
}
