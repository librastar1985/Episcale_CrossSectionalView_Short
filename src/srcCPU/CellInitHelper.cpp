
 /* CellInitHelper.cpp
 *
 *  Created on: Sep 22, 2013
 *      Authors: wsun2, Ali Nematbakhsh
 */
// To Do List: 
//1- The ID of cells to which asymmetric nuclear position is assigned, is given manually in generateInitIntnlNodes_M function. It should be updated if the number of cells are changed. It is better to be automatically detected.
//2-In the function generateInitIntnlNodes_M, the location of internal nodes is given randomly within certain radius from the cell center. This is modified manually in here. It should become an input paramter.
#include <fstream>
#include <stdexcept>
#include "CellInitHelper.h"
//Ali 
ForReadingData_M2 ReadFile_M2(std::string CellCentersFileName) {

          std::vector<GEOMETRY::Point2D> result;
          std::fstream inputc;
          ForReadingData_M2  ForReadingData1; 
          double TempPos_X,TempPos_Y,TempPos_Z ; 
		  string eCellTypeString ; 
          inputc.open(CellCentersFileName.c_str());

          if (inputc.is_open())
          {
            cout << "File successfully open";
          }
         else
         {
          cout << "Error opening file";
          }
          inputc >> ForReadingData1.CellNumber ; 
          for (int i = 0; i <ForReadingData1.CellNumber; i = i + 1) {
	    cout << "i=" << i << endl;		
	    inputc >> TempPos_X >> TempPos_Y >> TempPos_Z>> eCellTypeString ;	
	    ForReadingData1.TempSX.push_back(TempPos_X);
	    ForReadingData1.TempSY.push_back(TempPos_Y);
	    ForReadingData1.TempSZ.push_back(TempPos_Z);
		ECellType eCellType=StringToECellTypeConvertor (eCellTypeString) ; 
		ForReadingData1.eCellTypeV.push_back (eCellType) ; 
            }     
         cout << "Cell center positions read successfully";
		
          for (int i = 0; i <ForReadingData1.CellNumber; i = i + 1) {
			  cout << ForReadingData1.eCellTypeV.at(i) << "cell type read" << endl ;				 }
return ForReadingData1;
}

ECellType StringToECellTypeConvertor ( const string & eCellTypeString) {
	if (eCellTypeString=="bc") {return bc ;  }
		else if (eCellTypeString=="peri") {return peri ;  }
			else if (eCellTypeString=="pouch") {return pouch ;  }
//else { throw std:: invslid_argument ("recevied invalid cell type") ; } 

}
//Ali 

MembraneType1 StringToMembraneType1Convertor ( const string & mTypeString) {
	if (mTypeString=="lateralA") {return lateralA ;  }
		else if (mTypeString=="lateralB") {return lateralB ;  }
			else if (mTypeString=="basal1") {return basal1 ;  }
				else if (mTypeString=="apical1") {return apical1 ;  }
//else { throw std:: invslid_argument ("recevied invalid cell type") ; } 

}


CellInitHelper::CellInitHelper() {
	int type = globalConfigVars.getConfigValue("SimulationType").toInt();
	simuType = parseTypeFromConfig(type);
	if (simuType == Beak) {
		//initInternalBdry();  // Beak simulation is not active and I want to remove the dependency of the code on CGAL
	}
}

CVector CellInitHelper::getPointGivenAngle(double currentAngle, double r,
		CVector centerPos) {
	double xPos = centerPos.x + r * cos(currentAngle);
	double yPos = centerPos.y + r * sin(currentAngle);
	return CVector(xPos, yPos, 0);
}
void CellInitHelper::transformRawCartData(CartilageRawData& cartRawData,
		CartPara& cartPara, std::vector<CVector>& initNodePos) {
	// step 1, switch tip node1 to pos 0
	CVector tmpPos = cartRawData.tipVerticies[0];
	cartRawData.tipVerticies[0] =
			cartRawData.tipVerticies[cartRawData.growNode1Index_on_tip];
	cartRawData.tipVerticies[cartRawData.growNode1Index_on_tip] = tmpPos;
	cartPara.growNode1Index = 0;

	// step 2, switch tip node 2 to pos 1
	tmpPos = cartRawData.tipVerticies[1];
	cartRawData.tipVerticies[1] =
			cartRawData.tipVerticies[cartRawData.growNode2Index_on_tip];
	cartRawData.tipVerticies[cartRawData.growNode2Index_on_tip] = tmpPos;
	cartPara.growNode2Index = 1;
	cartPara.tipNodeStartPos = 2;
	cartPara.tipNodeIndexEnd = cartRawData.tipVerticies.size();

	// step 3, calculate size for tip nodes
	double tipMaxExpansionRatio = globalConfigVars.getConfigValue(
			"TipMaxExpansionRatio").toDouble();
	int maxTipSize = tipMaxExpansionRatio * cartRawData.tipVerticies.size();
	cartPara.nonTipNodeStartPos = maxTipSize;
	cartPara.nodeIndexEnd = cartPara.nonTipNodeStartPos
			+ cartRawData.nonTipVerticies.size();
	cartPara.pivotNode1Index = cartPara.nonTipNodeStartPos
			+ cartRawData.pivotNode1Index;
	cartPara.pivotNode2Index = cartPara.nonTipNodeStartPos
			+ cartRawData.pivotNode2Index;
	cartPara.growNodeBehind1Index = cartPara.nonTipNodeStartPos
			+ cartRawData.growNodeBehind1Index;
	cartPara.growNodeBehind2Index = cartPara.nonTipNodeStartPos
			+ cartRawData.growNodeBehind2Index;

	// step 4, calculate size for all nodes
	double cartmaxExpRatio = globalConfigVars.getConfigValue(
			"CartMaxExpansionRatio").toDouble();
	int maxCartNodeSize = (cartRawData.tipVerticies.size()
			+ cartRawData.nonTipVerticies.size()) * cartmaxExpRatio;
	cartPara.nodeIndexTotal = maxCartNodeSize;

	// step 5, initialize the first part of initNodePos
	initNodePos.resize(cartPara.nodeIndexTotal);
	for (uint i = 0; i < cartRawData.tipVerticies.size(); i++) {
		initNodePos[i] = cartRawData.tipVerticies[i];
	}

	// step 6, initialize the second part of initNodePos
	for (uint i = 0; i < cartRawData.nonTipVerticies.size(); i++) {
		initNodePos[i + cartPara.nonTipNodeStartPos] =
				cartRawData.nonTipVerticies[i];
	}

	for (uint i = 0; i < initNodePos.size(); i++) {
		initNodePos[i].Print();
	}

}

void CellInitHelper::initializeRawInput(RawDataInput& rawInput,
		std::vector<CVector>& cellCenterPoss) {

	for (unsigned int i = 0; i < cellCenterPoss.size(); i++) {
		CVector centerPos = cellCenterPoss[i];
		if (isMXType(centerPos)) {
			rawInput.MXCellCenters.push_back(centerPos);
		} else {
			rawInput.FNMCellCenters.push_back(centerPos);
		}
	}

	generateCellInitNodeInfo_v2(rawInput.initCellNodePoss);

}

/**
 * Initialize inputs for five different components.
 */

SimulationInitData CellInitHelper::initInputsV2(RawDataInput &rawData) {
	SimulationInitData initData;

	uint FnmCellCount = rawData.FNMCellCenters.size();
	uint MxCellCount = rawData.MXCellCenters.size();
	uint ECMCount = rawData.ECMCenters.size();

	uint maxNodePerCell =
			globalConfigVars.getConfigValue("MaxNodePerCell").toInt();
	uint maxNodePerECM = 0;
	if (rawData.simuType == Beak) {
		maxNodePerECM =
				globalConfigVars.getConfigValue("MaxNodePerECM").toInt();
	}

	uint initTotalCellCount = rawData.initCellNodePoss.size();
	initData.initBdryCellNodePosX.resize(rawData.bdryNodes.size(), 0.0);
	initData.initBdryCellNodePosY.resize(rawData.bdryNodes.size(), 0.0);
	initData.initProfileNodePosX.resize(rawData.profileNodes.size());
	initData.initProfileNodePosY.resize(rawData.profileNodes.size());
	initData.initECMNodePosX.resize(maxNodePerECM * ECMCount);
	initData.initECMNodePosY.resize(maxNodePerECM * ECMCount);
	initData.initFNMCellNodePosX.resize(maxNodePerCell * FnmCellCount, 0.0);
	initData.initFNMCellNodePosY.resize(maxNodePerCell * FnmCellCount, 0.0);
	initData.initMXCellNodePosX.resize(maxNodePerCell * MxCellCount, 0.0);
	initData.initMXCellNodePosY.resize(maxNodePerCell * MxCellCount, 0.0);

	for (uint i = 0; i < FnmCellCount; i++) {
		initData.cellTypes.push_back(FNM);
		initData.numOfInitActiveNodesOfCells.push_back(initTotalCellCount);
	}

	for (uint i = 0; i < MxCellCount; i++) {
		initData.cellTypes.push_back(MX);
		initData.numOfInitActiveNodesOfCells.push_back(initTotalCellCount);
	}

	for (uint i = 0; i < rawData.bdryNodes.size(); i++) {
		initData.initBdryCellNodePosX[i] = rawData.bdryNodes[i].x;
		initData.initBdryCellNodePosY[i] = rawData.bdryNodes[i].y;
	}

	for (uint i = 0; i < rawData.profileNodes.size(); i++) {
		initData.initProfileNodePosX[i] = rawData.profileNodes[i].x;
		initData.initProfileNodePosY[i] = rawData.profileNodes[i].y;
	}

	uint index;

	uint ECMInitNodeCount = rawData.initECMNodePoss.size();
	for (uint i = 0; i < ECMCount; i++) {
		vector<CVector> rotatedCoords = rotate2D(rawData.initECMNodePoss,
				rawData.ECMAngles[i]);
		for (uint j = 0; j < ECMInitNodeCount; j++) {
			index = i * maxNodePerECM + j;
			initData.initECMNodePosX[index] = rawData.ECMCenters[i].x
					+ rotatedCoords[j].x;
			initData.initECMNodePosY[index] = rawData.ECMCenters[i].y
					+ rotatedCoords[j].y;
		}
	}

	for (uint i = 0; i < FnmCellCount; i++) {
		for (uint j = 0; j < initTotalCellCount; j++) {
			index = i * maxNodePerCell + j;
			initData.initFNMCellNodePosX[index] = rawData.FNMCellCenters[i].x
					+ rawData.initCellNodePoss[j].x;
			initData.initFNMCellNodePosY[index] = rawData.FNMCellCenters[i].y
					+ rawData.initCellNodePoss[j].y;
		}
	}

	for (uint i = 0; i < MxCellCount; i++) {
		for (uint j = 0; j < initTotalCellCount; j++) {
			index = i * maxNodePerCell + j;
			initData.initMXCellNodePosX[index] = rawData.MXCellCenters[i].x
					+ rawData.initCellNodePoss[j].x;
			initData.initMXCellNodePosY[index] = rawData.MXCellCenters[i].y
					+ rawData.initCellNodePoss[j].y;
		}
	}

	return initData;
}

SimulationInitData_V2 CellInitHelper::initInputsV3(RawDataInput& rawData) {
	SimulationInitData_V2 initData;
	initData.isStab = rawData.isStab;
	initData.simuType = rawData.simuType;

	uint FnmCellCount = rawData.FNMCellCenters.size();
	uint MxCellCount = rawData.MXCellCenters.size();
	uint ECMCount = rawData.ECMCenters.size();

	uint maxNodePerCell =
			globalConfigVars.getConfigValue("MaxNodePerCell").toInt();
	uint maxNodePerECM = 0;
	if (initData.simuType == Beak) {
		maxNodePerECM =
				globalConfigVars.getConfigValue("MaxNodePerECM").toInt();
	}

	uint initTotalCellCount = rawData.initCellNodePoss.size();
	//uint initTotalECMCount = rawData.ECMCenters.size();
	initData.initBdryNodeVec.resize(rawData.bdryNodes.size());
	initData.initProfileNodeVec.resize(rawData.profileNodes.size());

	if (simuType == Beak && !initData.isStab) {
		transformRawCartData(rawData.cartilageData, initData.cartPara,
				initData.initCartNodeVec);
	}

	initData.initECMNodeVec.resize(maxNodePerECM * ECMCount);
	initData.initFNMNodeVec.resize(maxNodePerCell * FnmCellCount);
	initData.initMXNodeVec.resize(maxNodePerCell * MxCellCount);

	for (uint i = 0; i < FnmCellCount; i++) {
		initData.cellTypes.push_back(FNM);
		initData.numOfInitActiveNodesOfCells.push_back(initTotalCellCount);
	}

	for (uint i = 0; i < MxCellCount; i++) {
		initData.cellTypes.push_back(MX);
		initData.numOfInitActiveNodesOfCells.push_back(initTotalCellCount);
	}

	for (uint i = 0; i < rawData.bdryNodes.size(); i++) {
		initData.initBdryNodeVec[i] = rawData.bdryNodes[i];
	}

	for (uint i = 0; i < rawData.profileNodes.size(); i++) {
		initData.initProfileNodeVec[i] = rawData.profileNodes[i];
	}

	uint index;

	uint ECMInitNodeCount = rawData.initECMNodePoss.size();
	for (uint i = 0; i < ECMCount; i++) {
		vector<CVector> rotatedCoords = rotate2D(rawData.initECMNodePoss,
				rawData.ECMAngles[i]);
		for (uint j = 0; j < ECMInitNodeCount; j++) {
			index = i * maxNodePerECM + j;
			initData.initECMNodeVec[index] = rawData.ECMCenters[i]
					+ rotatedCoords[j];
		}
	}

	for (uint i = 0; i < FnmCellCount; i++) {
		for (uint j = 0; j < initTotalCellCount; j++) {
			index = i * maxNodePerCell + j;
			initData.initFNMNodeVec[index] = rawData.FNMCellCenters[i]
					+ rawData.initCellNodePoss[j];
		}
	}

	for (uint i = 0; i < MxCellCount; i++) {
		for (uint j = 0; j < initTotalCellCount; j++) {
			index = i * maxNodePerCell + j;
			initData.initMXNodeVec[index] = rawData.MXCellCenters[i]
					+ rawData.initCellNodePoss[j];
		}
	}

	return initData;
}
// This function reformat the the files read from input file to a format suitable to convert to GPU vectors. Initially the files are in the format of Vector <vector <**>>, it will be converted to Vector <> with standard format.
SimulationInitData_V2_M CellInitHelper::initInputsV3_M(
		RawDataInput_M& rawData_m) {
// This function is called Ali 
	if (rawData_m.simuType != Disc_M) {
		throw SceException("V2_M data can be used for Disc_M simulation only!",
				ConfigValueException);
	}

	uint maxMembrNodePerCell = globalConfigVars.getConfigValue(
			"MaxMembrNodeCountPerCell").toInt();
	uint maxAllNodeCountPerCell = globalConfigVars.getConfigValue(
			"MaxAllNodeCountPerCell").toInt();
	uint maxCellInDomain =
			globalConfigVars.getConfigValue("MaxCellInDomain").toInt();
	uint maxNodeInDomain = maxCellInDomain * maxAllNodeCountPerCell;
	uint initCellCount = rawData_m.initCellCenters.size();

	uint initMaxNodeCount = initCellCount * maxAllNodeCountPerCell;

	uint cellRank, nodeRank, activeMembrNodeCountThisCell,
			activeIntnlNodeCountThisCell;

	SimulationInitData_V2_M initData;
	initData.isStab = rawData_m.isStab;
	initData.simuType = rawData_m.simuType;

	initData.nodeTypes.resize(maxNodeInDomain);
	initData.mDppV.resize(maxNodeInDomain); // Ali 
	initData.mTypeV.resize(maxNodeInDomain); // Ali 
	initData.initNodeVec.resize(initMaxNodeCount);
	initData.initNodeMultip_actomyo.resize(initMaxNodeCount);
	initData.initNodeMultip_integrin.resize(initMaxNodeCount);
	initData.initIsActive.resize(initMaxNodeCount, false);
	//initData.initGrowProgVec.resize(initCellCount, 0);
    ECellType eCellTypeTmp2 ; // Ali
	
	for (uint i = 0; i < initCellCount; i++) {
		initData.initActiveMembrNodeCounts.push_back(
				rawData_m.initMembrNodePoss[i].size()); // the size of this vector will be used to count initial number of active nodes Ali 
		initData.initActiveIntnlNodeCounts.push_back(
				rawData_m.initIntnlNodePoss[i].size());
		initData.initGrowProgVec.push_back(rawData_m.cellGrowProgVec[i]);
		eCellTypeTmp2=rawData_m.cellsTypeCPU.at(i) ;  // Ali
		initData.eCellTypeV1.push_back(eCellTypeTmp2); // Ali

	}

	for (uint i = 0; i < maxNodeInDomain; i++) {
		nodeRank = i % maxAllNodeCountPerCell;
		if (nodeRank < maxMembrNodePerCell) {
			initData.nodeTypes[i] = CellMembr;
		} else {
			initData.nodeTypes[i] = CellIntnl;
		}
	}

	for (uint i = 0; i < initMaxNodeCount; i++) {
		cellRank = i / maxAllNodeCountPerCell;
		nodeRank = i % maxAllNodeCountPerCell;
		activeMembrNodeCountThisCell =
				rawData_m.initMembrNodePoss[cellRank].size();
		activeIntnlNodeCountThisCell =
				rawData_m.initIntnlNodePoss[cellRank].size();
		if (nodeRank < maxMembrNodePerCell) {
			if (nodeRank < activeMembrNodeCountThisCell) {
				initData.initNodeVec[i] =
						rawData_m.initMembrNodePoss[cellRank][nodeRank];
				initData.initNodeMultip_actomyo[i] =
						rawData_m.initMembrMultip_actomyo[cellRank][nodeRank];
				initData.initNodeMultip_integrin[i] =
						rawData_m.initMembrMultip_integrin[cellRank][nodeRank];
				initData.initIsActive[i] = true;
				initData.mDppV[i]=rawData_m.mDppV2[cellRank][nodeRank] ;  //Ali
				initData.mTypeV[i]=rawData_m.mTypeV2[cellRank][nodeRank] ;  //Ali
			} else {
				initData.initIsActive[i] = false;
				initData.mDppV[i]=0.0 ; //Ali
				initData.mTypeV[i]=notAssigned1 ; //Ali
			}
		} else {
			uint intnlIndex = nodeRank - maxMembrNodePerCell;
			if (intnlIndex < activeIntnlNodeCountThisCell) {
				initData.initNodeVec[i] =
				rawData_m.initIntnlNodePoss[cellRank][intnlIndex];
				initData.initIsActive[i] = true;
				initData.mDppV[i]=0.0 ; //Ali
				initData.mTypeV[i]=notAssigned1 ; //Ali
			} else {
				initData.initIsActive[i] = false;
				initData.mDppV[i]=0.0 ; //Ali
				initData.mTypeV[i]=notAssigned1 ; //Ali
			}
		}
	}

	return initData;
}

vector<CVector> CellInitHelper::rotate2D(vector<CVector> &initECMNodePoss,
		double angle) {
	uint inputVectorSize = initECMNodePoss.size();
	CVector centerPosOfInitVector = CVector(0);
	for (uint i = 0; i < inputVectorSize; i++) {
		centerPosOfInitVector = centerPosOfInitVector + initECMNodePoss[i];
	}
	centerPosOfInitVector = centerPosOfInitVector / inputVectorSize;
	for (uint i = 0; i < inputVectorSize; i++) {
		initECMNodePoss[i] = initECMNodePoss[i] - centerPosOfInitVector;
	}
	vector<CVector> result;
	for (uint i = 0; i < inputVectorSize; i++) {
		CVector posNew;
		posNew.x = cos(angle) * initECMNodePoss[i].x
				- sin(angle) * initECMNodePoss[i].y;
		posNew.y = sin(angle) * initECMNodePoss[i].x
				+ cos(angle) * initECMNodePoss[i].y;
		result.push_back(posNew);
	}
	return result;
}
/* CGAL Deactivation
RawDataInput CellInitHelper::generateRawInput_stab() {
	RawDataInput rawData;
	rawData.simuType = simuType;
	vector<CVector> insideCellCenters;
	vector<CVector> outsideBdryNodePos;
	std::string bdryInputFileName = globalConfigVars.getConfigValue(
			"Bdry_InputFileName").toString();

	GEOMETRY::MeshGen meshGen;

	GEOMETRY::UnstructMesh2D mesh = meshGen.generateMesh2DFromFile(
			bdryInputFileName);

	std::vector<GEOMETRY::Point2D> insideCenterPoints =
			mesh.getAllInsidePoints();

	double fine_Ratio =
			globalConfigVars.getConfigValue("StabBdrySpacingRatio").toDouble();

	for (uint i = 0; i < insideCenterPoints.size(); i++) {
		insideCellCenters.push_back(
				CVector(insideCenterPoints[i].getX(),
						insideCenterPoints[i].getY(), 0));
	}

	mesh = meshGen.generateMesh2DFromFile(bdryInputFileName, fine_Ratio);

	std::vector<GEOMETRY::Point2D> bdryPoints = mesh.getOrderedBdryPts();

	for (uint i = 0; i < bdryPoints.size(); i++) {
		outsideBdryNodePos.push_back(
				CVector(bdryPoints[i].getX(), bdryPoints[i].getY(), 0));
	}

	for (unsigned int i = 0; i < insideCellCenters.size(); i++) {
		CVector centerPos = insideCellCenters[i];
		rawData.MXCellCenters.push_back(centerPos);
		centerPos.Print();
	}

	for (uint i = 0; i < outsideBdryNodePos.size(); i++) {
		rawData.bdryNodes.push_back(outsideBdryNodePos[i]);
	}

	generateCellInitNodeInfo_v2(rawData.initCellNodePoss);

	rawData.isStab = true;
	return rawData;
}
*/
RawDataInput_M CellInitHelper::generateRawInput_M() {   // an Important function in cell inithelper
	RawDataInput_M rawData;

	rawData.simuType = simuType;
	vector<CVector> insideCellCenters;
	vector<CVector> outsideBdryNodePos;
	std::string bdryInputFileName = globalConfigVars.getConfigValue(
			"Bdry_InputFileName").toString();

	std::string CellCentersFileName = globalConfigVars.getConfigValue(
			"CellCenters_FileName").toString() ;
    //Ali
	//This function read the cells centers coordinates and their type
    ForReadingData_M2 ForReadingData2 = ReadFile_M2(CellCentersFileName);

    GEOMETRY::Point2D Point2D1[ForReadingData2.CellNumber];
	//not used for now  Ali 
	//GEOMETRY::MeshGen meshGen;
	//GEOMETRY::UnstructMesh2D mesh = meshGen.generateMesh2DFromFile(
	//		bdryInputFileName);
	////////////////////////

         //Ali
    std::vector<GEOMETRY::Point2D> insideCenterCenters ; 
    for (int ii = 0; ii <ForReadingData2.CellNumber; ii = ii + 1) {
		
		Point2D1[ii].Assign_M2(ForReadingData2.TempSX[ii], ForReadingData2.TempSY[ii]);
		cout << "x coordinate=" << Point2D1[ii].getX() << "y coordinate=" << Point2D1[ii].getY() << "Is on Boundary=" << Point2D1[ii].isIsOnBdry() << endl;
		insideCenterCenters.push_back(Point2D1[ii]); 
	}
         
        //Ali 


         //Ali comment
//	std::vector<GEOMETRY::Point2D> insideCenterCenters =
//			mesh.getAllInsidePoints();

       //Ali comment

	uint initCellCt = insideCenterCenters.size();

	for (uint i = 0; i < initCellCt; i++) {
		insideCellCenters.push_back(
				CVector(insideCenterCenters[i].getX(),
						insideCenterCenters[i].getY(), 0));
	}



	double randNum;
	double progDivStart =
			globalConfigVars.getConfigValue("GrowthPrgrCriVal").toDouble();
	for (uint i = 0; i < initCellCt; i++) {
		randNum = (double) rand() / ((double) RAND_MAX + 1) * (progDivStart-0.25)   ; //Ali here 
		//randNum = (double) rand() / ((double) RAND_MAX + 1)*0.98   ; //Ali here 
		//std::cout << "rand init growth progress = " << randNum << std::endl;
//Ali to make the initial progree of all nodes zero

 
	//	rawData.cellGrowProgVec.push_back(randNum);
		rawData.cellGrowProgVec.push_back(0.7);
		ECellType eCellTypeTmp=ForReadingData2.eCellTypeV.at(i);  
		rawData.cellsTypeCPU.push_back(eCellTypeTmp);
	}

	std::cout << "Printing initial cell center positions ......" << std::endl;
	for (unsigned int i = 0; i < insideCellCenters.size(); i++) {
		CVector centerPos = insideCellCenters[i];
		rawData.initCellCenters.push_back(centerPos);
		std::cout << "    ";
		centerPos.Print();
	}
	// This functions reads membrane nodes coordinates, dpp levels and types, and generates internal nodes coordinates
	generateCellInitNodeInfo_v3(rawData.initCellCenters,
			rawData.cellGrowProgVec, rawData.initMembrNodePoss,
			rawData.initIntnlNodePoss, rawData.initMembrMultip_actomyo, rawData.initMembrMultip_integrin, rawData.mDppV2,rawData.mTypeV2); 

	//std::cout << "finished generate raw data" << std::endl;
	//std::cout.flush();

	rawData.isStab = true;
	return rawData;
}

void CellInitHelper::generateRandomAngles(vector<double> &randomAngles,
		int initProfileNodeSize) {
	static const double PI = acos(-1.0);
	randomAngles.clear();
	for (int i = 0; i < initProfileNodeSize; i++) {
		double randomNum = rand() / ((double) RAND_MAX + 1);
		randomAngles.push_back(randomNum * 2.0 * PI);
	}
}

/**
 * Initially, ECM nodes are alignd vertically.
 */
void CellInitHelper::generateECMInitNodeInfo(vector<CVector> &initECMNodePoss,
		int initNodeCountPerECM) {
	initECMNodePoss.clear();
	double ECMInitNodeInterval = globalConfigVars.getConfigValue(
			"ECM_Init_Node_Interval").toDouble();
	int numOfSegments = initNodeCountPerECM - 1;
//double totalLength = ECMInitNodeInterval * numOfSegments;
	if (numOfSegments % 2 == 0) {
		CVector initPt = CVector(0, 0, 0);
		initECMNodePoss.push_back(initPt);
		for (int i = 1; i <= numOfSegments / 2; i++) {
			CVector posSide = initPt + CVector(0, i * ECMInitNodeInterval, 0);
			CVector negSide = initPt - CVector(0, i * ECMInitNodeInterval, 0);
			initECMNodePoss.push_back(posSide);
			initECMNodePoss.push_back(negSide);
		}
	} else {
		CVector initPosPt = CVector(0, ECMInitNodeInterval / 2.0, 0);
		CVector initNegPt = CVector(0, -ECMInitNodeInterval / 2.0, 0);
		initECMNodePoss.push_back(initPosPt);
		initECMNodePoss.push_back(initNegPt);
		for (int i = 1; i <= numOfSegments / 2; i++) {
			CVector posSide = initPosPt
					+ CVector(0, i * ECMInitNodeInterval, 0);
			CVector negSide = initNegPt
					- CVector(0, i * ECMInitNodeInterval, 0);
			initECMNodePoss.push_back(posSide);
			initECMNodePoss.push_back(negSide);
		}
	}
}

void CellInitHelper::generateECMCenters(vector<CVector> &ECMCenters,
		vector<CVector> &CellCenters, vector<CVector> &bdryNodes) {
	ECMCenters.clear();
	vector<double> angles;
	vector<CVector> vecs;
	static const double PI = acos(-1.0);

	const int numberOfECMAroundCellCenter = globalConfigVars.getConfigValue(
			"ECM_Around_Cell_Center").toInt();
	const double distFromCellCenter = globalConfigVars.getConfigValue(
			"Dist_From_Cell_Center").toDouble();
	double unitAngle = 2 * PI / numberOfECMAroundCellCenter;
	for (int i = 0; i < numberOfECMAroundCellCenter; i++) {
		angles.push_back(i * unitAngle);
		CVector vec(sin(i * unitAngle), cos(i * unitAngle), 0);
		vec = vec * distFromCellCenter;
		vecs.push_back(vec);
	}
	for (uint i = 0; i < CellCenters.size(); i++) {
		for (int j = 0; j < numberOfECMAroundCellCenter; j++) {
			CVector pos = CellCenters[i] + vecs[j];
			if (anyCellCenterTooClose(CellCenters, pos)) {
				continue;
			}
			if (anyECMCenterTooClose(ECMCenters, pos)) {
				continue;
			}
			if (anyBoundaryNodeTooClose(bdryNodes, pos)) {
				continue;
			}
			ECMCenters.push_back(pos);
			if (std::isnan(pos.GetX())) {
				throw SceException("number is NAN!", InputInitException);
			}
		}
	}
}

bool CellInitHelper::anyCellCenterTooClose(vector<CVector> &cellCenters,
		CVector position) {
	double MinDistToOtherCellCenters = globalConfigVars.getConfigValue(
			"MinDistToCellCenter").toDouble();
	int size = cellCenters.size();
	for (int i = 0; i < size; i++) {
		if (Modul(cellCenters[i] - position) < MinDistToOtherCellCenters) {
			return true;
		}
	}
	return false;
}

bool CellInitHelper::anyECMCenterTooClose(vector<CVector> &ecmCenters,
		CVector position) {
	double MinDistToOtherECMCenters = globalConfigVars.getConfigValue(
			"MinDistToECMCenter").toDouble();
	int size = ecmCenters.size();
	for (int i = 0; i < size; i++) {
		if (Modul(ecmCenters[i] - position) < MinDistToOtherECMCenters) {
			return true;
		}
	}
	return false;
}

bool CellInitHelper::anyBoundaryNodeTooClose(vector<CVector> &bdryNodes,
		CVector position) {
	double MinDistToOtherBdryNodes = globalConfigVars.getConfigValue(
			"MinDistToBdryNodes").toDouble();
	int size = bdryNodes.size();
	for (int i = 0; i < size; i++) {
		if (Modul(bdryNodes[i] - position) < MinDistToOtherBdryNodes) {
			return true;
		}
	}
	return false;
}

CellInitHelper::~CellInitHelper() {
}

void CellInitHelper::generateCellInitNodeInfo_v2(vector<CVector>& initPos) {
	initPos = generateInitCellNodes();
}

void CellInitHelper::generateCellInitNodeInfo_v3(vector<CVector>& initCenters,   //This function is called //Ali 
		vector<double>& initGrowProg, vector<vector<CVector> >& initMembrPos,
		vector<vector<CVector> >& initIntnlPos, 
		vector<vector<CVector> >& initMembrMultip_actomyo,
		vector<vector<CVector> >& initMembrMultip_integrin,
		vector<vector<double> >& mDppV2, 
		vector<vector<MembraneType1> >& mTypeV2 )
{
	// vector<double> multip_info; //Note: the weighted intensity here represents the actomyosin intensity.
	// multip_info.push_back(0.6932);
	// multip_info.push_back(0.7895);
	// multip_info.push_back(0.8259);
	// multip_info.push_back(0.9003);
	// multip_info.push_back(0.9115);
	// multip_info.push_back(0.9868);
	// multip_info.push_back(0.8688);
	// multip_info.push_back(1.0000);
	// multip_info.push_back(0.8834);
	// multip_info.push_back(0.9042);
	// multip_info.push_back(0.9368);
	// multip_info.push_back(0.7465);
	// multip_info.push_back(0.8535);
	// multip_info.push_back(0.8026);
	// multip_info.push_back(0.8325);
	// multip_info.push_back(0.7171);
	// multip_info.push_back(0.6745);
	// multip_info.push_back(0.6003);
	// multip_info.push_back(0.6159);
	// multip_info.push_back(0.6449);
	// multip_info.push_back(0.5711);
	// multip_info.push_back(0.5144);
	// multip_info.push_back(0.4960);
	// multip_info.push_back(0.5026);
	// multip_info.push_back(0.5172);
	// multip_info.push_back(0.5261);
	// multip_info.push_back(0.5330);
	// multip_info.push_back(0.5733);
	// multip_info.push_back(0.5609);
	// multip_info.push_back(0.5874);
	// multip_info.push_back(0.6210);
	// multip_info.push_back(0.6499);
	// multip_info.push_back(0.6540);
	// multip_info.push_back(0.6798);
	// multip_info.push_back(0.7186);
	// multip_info.push_back(0.7509);
	// multip_info.push_back(0.7037);
	// multip_info.push_back(0.7228);
	// multip_info.push_back(0.7725);
	// multip_info.push_back(0.8035);
	// multip_info.push_back(0.7611);
	// multip_info.push_back(0.8351);
	// multip_info.push_back(0.7737);
	// multip_info.push_back(0.9053);
	// multip_info.push_back(0.8444);
	// multip_info.push_back(0.8266);
	// multip_info.push_back(0.8132);
	// multip_info.push_back(0.9219);
	// multip_info.push_back(0.8528);
	// multip_info.push_back(0.8072);
	// multip_info.push_back(0.8008);
	// multip_info.push_back(0.8564);
	// multip_info.push_back(0.8507);
	// multip_info.push_back(0.7850);
	// multip_info.push_back(0.8535);
	// multip_info.push_back(0.8771);
	// multip_info.push_back(0.7674);
	// multip_info.push_back(0.8777);
	// multip_info.push_back(0.9007);
	// multip_info.push_back(0.9127);
	// multip_info.push_back(0.9237);

	// (1/11/2021) arbitrary step functions to adjust scaling.
	// (1/08/2021) basal actomyo from new data of Disc1 out of seven discs in [...]. The scaling is based on a new calculation method.
	// (1/07/2021) basal actomyo from new data of Disc1 out of seven discs in ProteinConcentration xlsx file
	vector<double> multip_info;
	double step1 = 1.0;
	int step1_size = 10;
	double step2 = 1.0;
	int step2_size = 10;
	double step3 = 1.0;
	int step3_size = 21;
	double step4 = 1.0;
	int step4_size = 10;
	double step5 = 1.0;
	int step5_size = 10;
	// cout<<"Scaling applied for step1 ="<<step1<<", "<<"Size of step1 = "<<step1_size<<endl;
	// cout<<"Scaling applied for step2 ="<<step2<<", "<<"Size of step2 = "<<step2_size<<endl;
	// cout<<"Scaling applied for step3 ="<<step3<<", "<<"Size of step3 = "<<step3_size<<endl;
	// cout<<"Scaling applied for step4 ="<<step4<<", "<<"Size of step4 = "<<step4_size<<endl;
	// cout<<"Scaling applied for step5 ="<<step5<<", "<<"Size of step5 = "<<step5_size<<endl;

	int Contractility_Scaling_Assigned = 0;

	for (int i = 0; i < step1_size; i++){
		multip_info.push_back(step1);
		Contractility_Scaling_Assigned += 1;
	}
	for (int i = 0; i < step2_size; i++){
		multip_info.push_back(step2);
		Contractility_Scaling_Assigned += 1;
	}
	for (int i = 0; i < step3_size; i++){
		multip_info.push_back(step3);
		Contractility_Scaling_Assigned += 1;
	}
	for (int i = 0; i < step4_size; i++){
		multip_info.push_back(step4);
		Contractility_Scaling_Assigned += 1;
	}
	for (int i = 0; i < step5_size; i++){
		multip_info.push_back(step5);
		Contractility_Scaling_Assigned += 1;
	}

	if (Contractility_Scaling_Assigned != 61){
		cout<<"INCORRECT NUMBER OF CONTRACTILITY SCALER ASSIGNED!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
		cout<<"PLEASE CHECK IMMEDIATELY!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
		cout<<"OR THE SIMULATION RESULTS WILL BE INVALID!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
	}
	// multip_info.push_back(step1);//step1);//(1/11/2021)		//0.774811967);//(1/8/2021) 	//(1/7/2021) 302.08);
    // multip_info.push_back(step1);//step1);//(1/11/2021)		//0.761620724);//(1/8/2021) 	//(1/7/2021) 299.84);
    // multip_info.push_back(step1);//step1);//(1/11/2021)		//0.846785241);//(1/8/2021) 	//(1/7/2021) 330.32);
    // multip_info.push_back(step1);//step1);//(1/11/2021)		//0.916907113);//(1/8/2021) 	//(1/7/2021) 349.56);
    // multip_info.push_back(step1);//step1);//(1/11/2021)		//0.923387022);//(1/8/2021) 	//(1/7/2021) 349.68);
    // multip_info.push_back(step1);//step1);//(1/11/2021)		//0.997443123);//(1/8/2021) 	//(1/7/2021) 383.84);
    // multip_info.push_back(step1);//step1);//(1/11/2021)		//1.017345701);//(1/8/2021) 	//(1/7/2021) 389.80);
    // multip_info.push_back(step1);//step1);//(1/11/2021)		//1.178957263);//(1/8/2021) 	//(1/7/2021) 443.72);
    // multip_info.push_back(step1);//step1);//(1/11/2021)		//1.189589142);//(1/8/2021) 	//(1/7/2021) 447.72);
    // multip_info.push_back(step1);//step1);//(1/11/2021)		//1.188604709);//(1/8/2021) 	//(1/7/2021) 449.00);
    // multip_info.push_back(step1);//step1);//(1/11/2021)		//1.143241074);//(1/8/2021) 	//(1/7/2021) 437.96);
    // multip_info.push_back(step1);//step1);//(1/11/2021)		//1.220768556);//(1/8/2021) 	//(1/7/2021) 463.28);
    // multip_info.push_back(step2);//step1);//(1/11/2021)		//1.252199097);//(1/8/2021) 	//(1/7/2021) 494.92);
    // multip_info.push_back(step2);//step1);//(1/11/2021)		//1.255152397);//(1/8/2021) 	//(1/7/2021) 491.96);
    // multip_info.push_back(step2);//step1);//(1/11/2021)		//1.122644221);//(1/8/2021) 	//(1/7/2021) 442.80);
    // multip_info.push_back(step2);//step1);//(1/11/2021)		//1.149720983);//(1/8/2021) 	//(1/7/2021) 447.20);
    // multip_info.push_back(step2);//step1);//(1/11/2021)		//1.130744107);//(1/8/2021) 	//(1/7/2021) 437.56);
    // multip_info.push_back(step2);//step1);//(1/11/2021)		//1.171012113);//(1/8/2021) 	//(1/7/2021) 451.08);
    // multip_info.push_back(step2);//step1);//(1/11/2021)		//1.131669808);//(1/8/2021) 	//(1/7/2021) 429.20);
    // multip_info.push_back(step2);//step1);//(1/11/2021)		//0.945141001);//(1/8/2021) 	//(1/7/2021) 369.72);
    // multip_info.push_back(step2);//step2);//(1/11/2021)		//0.912741457);//(1/8/2021) 	//(1/7/2021) 368.16);
    // multip_info.push_back(step2);//step2);//(1/11/2021)		//1.061085086);//(1/8/2021) 	//(1/7/2021) 399.88);
    // multip_info.push_back(step2);//step2);//(1/11/2021)		//0.939936869);//(1/8/2021) 	//(1/7/2021) 354.04);
    // multip_info.push_back(step2);//step2);//(1/11/2021)		//0.963423602);//(1/8/2021) 	//(1/7/2021) 366.08);
    // multip_info.push_back(step3);//step2);//(1/11/2021)		//0.943983875);//(1/8/2021) 	//(1/7/2021) 386.80);
    // multip_info.push_back(step3);//step2);//(1/11/2021)		//0.991521172);//(1/8/2021) 	//(1/7/2021) 372.92);
    // multip_info.push_back(step3);//step2);//(1/11/2021)		//0.859513634);//(1/8/2021) 	//(1/7/2021) 332.52);
    // multip_info.push_back(step3);//step2);//(1/11/2021)		//0.919452791);//(1/8/2021) 	//(1/7/2021) 363.48);
    // multip_info.push_back(step3);//step2);//(1/11/2021)		//0.815311398);//(1/8/2021) 	//(1/7/2021) 308.32);
    // multip_info.push_back(step3);//step2);//(1/11/2021)		//0.913750945);//(1/8/2021) 	//(1/7/2021) 349.24);
    // multip_info.push_back(step3);//step2);//(1/11/2021)		//0.939586794);//(1/8/2021) 	//(1/7/2021) 359.68);
    // multip_info.push_back(step3);//step2);//(1/11/2021)		//0.996049565);//(1/8/2021) 	//(1/7/2021) 400.72);
    // multip_info.push_back(step3);//step2);//(1/11/2021)		//0.959257946);//(1/8/2021) 	//(1/7/2021) 376.88);
    // multip_info.push_back(step3);//step2);//(1/11/2021)		//1.000914503);//(1/8/2021) 	//(1/7/2021) 380.92);
    // multip_info.push_back(step3);//step2);//(1/11/2021)		//0.962578834);//(1/8/2021) 	//(1/7/2021) 364.12);
    // multip_info.push_back(step3);//step2);//(1/11/2021)		//0.90024449);//(1/8/2021) 	//(1/7/2021) 343.80);
    // multip_info.push_back(step4);//step2);//(1/11/2021)		//0.957175118);//(1/8/2021) 	//(1/7/2021) 365.36);
    // multip_info.push_back(step4);//step2);//(1/11/2021)		//0.775043392);//(1/8/2021) 	//(1/7/2021) 302.52);
    // multip_info.push_back(step4);//step2);//(1/11/2021)		//0.810682891);//(1/8/2021) 	//(1/7/2021) 311.44);
    // multip_info.push_back(step4);//step2);//(1/11/2021)		//0.910403872);//(1/8/2021) 	//(1/7/2021) 360.12);
    // multip_info.push_back(step4);//step3);//(1/11/2021)		//0.882642854);//(1/8/2021) 	//(1/7/2021) 332.52);
    // multip_info.push_back(step4);//step3);//(1/11/2021)		//0.934958287);//(1/8/2021) 	//(1/7/2021) 359.72);
    // multip_info.push_back(step4);//step3);//(1/11/2021)		//0.870159198);//(1/8/2021) 	//(1/7/2021) 344.32);
    // multip_info.push_back(step4);//step3);//(1/11/2021)		//0.989749192);//(1/8/2021) 	//(1/7/2021) 374.84);
    // multip_info.push_back(step4);//step3);//(1/11/2021)		//0.955555141);//(1/8/2021) 	//(1/7/2021) 372.68);
    // multip_info.push_back(step4);//step3);//(1/11/2021)		//0.933833383);//(1/8/2021) 	//(1/7/2021) 363.72);
    // multip_info.push_back(step4);//step3);//(1/11/2021)		//0.961200627);//(1/8/2021) 	//(1/7/2021) 378.16);
    // multip_info.push_back(step4);//step3);//(1/11/2021)		//0.923989051);//(1/8/2021) 	//(1/7/2021) 354.76);
    // multip_info.push_back(step5);//step3);//(1/11/2021)		//0.896031146);//(1/8/2021) 	//(1/7/2021) 340.68);
    // multip_info.push_back(step5);//step3);//(1/11/2021)		//0.888549453);//(1/8/2021) 	//(1/7/2021) 336.76);
    // multip_info.push_back(step5);//step3);//(1/11/2021)		//0.913667158);//(1/8/2021) 	//(1/7/2021) 343.96);
    // multip_info.push_back(step5);//step3);//(1/11/2021)		//1.013180045);//(1/8/2021) 	//(1/7/2021) 383.48);
    // multip_info.push_back(step5);//step3);//(1/11/2021)		//0.96631968);//(1/8/2021) 	//(1/7/2021) 374.08);
    // multip_info.push_back(step5);//step3);//(1/11/2021)		//1.0545249);//(1/8/2021) 	//(1/7/2021) 412.80);
    // multip_info.push_back(step5);//step3);//(1/11/2021)		//1.019085303);//(1/8/2021) 	//(1/7/2021) 406.48);
    // multip_info.push_back(step5);//step3);//(1/11/2021)		//1.066338099);//(1/8/2021) 	//(1/7/2021) 426.00);
    // multip_info.push_back(step5);//step3);//(1/11/2021)		//1.029126522);//(1/8/2021) 	//(1/7/2021) 403.12);
    // multip_info.push_back(step5);//step3);//(1/11/2021)		//0.875554935);//(1/8/2021) 	//(1/7/2021) 350.92);
    // multip_info.push_back(step5);//step3);//(1/11/2021)		//0.818064033);//(1/8/2021) 	//(1/7/2021) 327.84);
    // multip_info.push_back(step5);//step3);//(1/11/2021)		//0.8141263);//(1/8/2021) 	//(1/7/2021) 323.16);
    // multip_info.push_back(step5);//step3);//(1/11/2021)		//0.883581867);//(1/8/2021) 	//(1/7/2021) 335.88);

	//apical actomyo from new data on 11/29/2020
	vector<double> multip_info2;
	multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);
    multip_info2.push_back(0.00);

	double total_net_intensity = 0.0;
	double alpha, beta;
	alpha = 1.0;
	beta = 1.0;
	// cout<<"Coefficients of linear combination (alpha*basal + beta*apical) of basal and apical actomyosin intensity are alpha = "<<alpha<<" & beta = "<<beta<<endl;
	for (uint i = 0; i < multip_info.size(); i++){
		if (multip_info.size() != multip_info2.size()){
			cout<<"Basal and apical ctomyosin intensities have different dimension! Something is wrong!"<<endl;
			break;
		}
		// multip_info[i] = alpha*multip_info[i] + beta*multip_info2[i];
		// if (multip_info[i] < 0){
			// multip_info[i] = 0.0; //Does not allow negative net intensity between the basal and apical.
		// }
		// total_net_intensity += multip_info[i];
		// total_net_intensity += multip_info[i] + multip_info2[i];
	}

	double max_weighted_intensity = -1000.0;
	for (uint i = 0; i < multip_info.size(); i++){
		// multip_info[i] = multip_info[i]/total_net_intensity;
		if (multip_info[i] >= max_weighted_intensity){
			max_weighted_intensity = multip_info[i];
		}
		if (multip_info2[i] >= max_weighted_intensity){
			max_weighted_intensity = multip_info2[i];
		}
	}

	for (uint i = 0; i < multip_info.size(); i++){
		multip_info[i] = pow(multip_info[i],1.0);//alpha*pow(multip_info[i]/max_weighted_intensity, 3.0);
		multip_info2[i] = beta*pow(multip_info2[i]/max_weighted_intensity, 1.0);
	}

	vector<double> multip_info_integrin; //Weighted basal integrin from new data on 11/29/2020
	double step1_integrin = 1.0;
	int step1_size_integrin = 10;
	double step2_integrin = 1.0;
	int step2_size_integrin = 10;
	double step3_integrin = 1.0;
	int step3_size_integrin = 21;
	double step4_integrin = 1.0;
	int step4_size_integrin = 10;
	double step5_integrin = 1.0;
	int step5_size_integrin = 10;
	// cout<<"Scaling applied for step1 (integrin) ="<<step1_integrin<<", "<<"Size of step1 = "<<step1_size_integrin<<endl;
	// cout<<"Scaling applied for step2 (integrin) ="<<step2_integrin<<", "<<"Size of step2 = "<<step2_size_integrin<<endl;
	// cout<<"Scaling applied for step3 (integrin) ="<<step3_integrin<<", "<<"Size of step3 = "<<step3_size_integrin<<endl;
	// cout<<"Scaling applied for step4 (integrin) ="<<step4_integrin<<", "<<"Size of step4 = "<<step4_size_integrin<<endl;
	// cout<<"Scaling applied for step5 (integrin) ="<<step5_integrin<<", "<<"Size of step5 = "<<step5_size_integrin<<endl;

	int Integrin_Scaling_Assigned = 0;

	for (int i = 0; i < step1_size; i++){
		multip_info_integrin.push_back(step1_integrin);
		Integrin_Scaling_Assigned += 1;
	}
	for (int i = 0; i < step2_size; i++){
		multip_info_integrin.push_back(step2_integrin);
		Integrin_Scaling_Assigned += 1;
	}
	for (int i = 0; i < step3_size; i++){
		multip_info_integrin.push_back(step3_integrin);
		Integrin_Scaling_Assigned += 1;
	}
	for (int i = 0; i < step4_size; i++){
		multip_info_integrin.push_back(step4_integrin);
		Integrin_Scaling_Assigned += 1;
	}
	for (int i = 0; i < step5_size; i++){
		multip_info_integrin.push_back(step5_integrin);
		Integrin_Scaling_Assigned += 1;
	}

	if (Integrin_Scaling_Assigned != 61){
		cout<<"INCORRECT NUMBER OF INTEGRIN SCALER ASSIGNED!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
		cout<<"PLEASE CHECK IMMEDIATELY!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
		cout<<"OR THE SIMULATION RESULTS WILL BE INVALID!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
	}
	// multip_info_integrin.push_back(0.6626);//0.6802);
	// multip_info_integrin.push_back(    0.6728);//0.7486);
	// multip_info_integrin.push_back(    0.7889);//0.7479);
	// multip_info_integrin.push_back(    0.8750);//0.7976);
	// multip_info_integrin.push_back(    0.8350);//0.7696);
	// multip_info_integrin.push_back(    0.9224);//0.8024);
	// multip_info_integrin.push_back(    0.9446);//0.7205);
	// multip_info_integrin.push_back(    1.0000);//0.7455);
	// multip_info_integrin.push_back(    1.0000);//0.7118);
	// multip_info_integrin.push_back(    1.0000);//0.7124);
	// multip_info_integrin.push_back(    1.0000);//0.7234);
	// multip_info_integrin.push_back(    1.0000);//0.6424);
	// multip_info_integrin.push_back(    1.0000);//0.6778);
	// multip_info_integrin.push_back(    1.0000);//0.6242);
	// multip_info_integrin.push_back(    1.0000);//0.5983);
	// multip_info_integrin.push_back(    1.0000);//0.5942);
	// multip_info_integrin.push_back(    0.9975);//0.5364);
	// multip_info_integrin.push_back(    1.0000);//0.5429);
	// multip_info_integrin.push_back(    0.8765);//0.5252);
	// multip_info_integrin.push_back(    0.8333);//0.5304);
	// multip_info_integrin.push_back(    0.8334);//0.5245);
	// multip_info_integrin.push_back(    0.8014);//0.4571);
	// multip_info_integrin.push_back(    0.7978);//0.4590);
	// multip_info_integrin.push_back(    0.7763);//0.4560);
	// multip_info_integrin.push_back(    0.8430);//0.4628);
	// multip_info_integrin.push_back(    0.7556);//0.4643);
	// multip_info_integrin.push_back(    0.7338);//0.4490);
	// multip_info_integrin.push_back(    0.7594);//0.4615);
	// multip_info_integrin.push_back(    0.7627);//0.4766);
	// multip_info_integrin.push_back(    0.6574);//0.4522);
	// multip_info_integrin.push_back(    0.7115);//0.4518);
	// multip_info_integrin.push_back(    0.7148);//0.4518);
	// multip_info_integrin.push_back(    0.7473);//0.5053);
	// multip_info_integrin.push_back(    0.7939);//0.4974);
	// multip_info_integrin.push_back(    0.8839);//0.5543);
	// multip_info_integrin.push_back(    0.9415);//0.5600);
	// multip_info_integrin.push_back(    0.7523);//0.5696);
	// multip_info_integrin.push_back(    0.5744);//0.5957);
	// multip_info_integrin.push_back(    0.6795);//0.6031);
	// multip_info_integrin.push_back(    0.7583);//0.6064);
	// multip_info_integrin.push_back(    0.8990);//0.6430);
	// multip_info_integrin.push_back(    0.8602);//0.6967);
	// multip_info_integrin.push_back(    0.8482);//0.7063);
	// multip_info_integrin.push_back(    0.8682);//0.6791);
	// multip_info_integrin.push_back(    0.9078);//0.7059);
	// multip_info_integrin.push_back(    0.8573);//0.7659);
	// multip_info_integrin.push_back(    0.7818);//0.6978);
	// multip_info_integrin.push_back(    0.8695);//0.7431);
	// multip_info_integrin.push_back(    0.8427);//0.7486);
	// multip_info_integrin.push_back(    0.7631);//0.7306);
	// multip_info_integrin.push_back(    0.8541);//0.7286);
	// multip_info_integrin.push_back(    0.9187);//0.7685);
	// multip_info_integrin.push_back(    0.9622);//0.7197);
	// multip_info_integrin.push_back(    0.9729);//0.7024);
	// multip_info_integrin.push_back(    0.9581);//0.7466);
	// multip_info_integrin.push_back(    0.9733);//0.7529);
	// multip_info_integrin.push_back(    0.9332);//0.7357);
	// multip_info_integrin.push_back(    0.8471);//0.7512);
	// multip_info_integrin.push_back(    0.8496);//0.7483);
	// multip_info_integrin.push_back(    0.7934);//0.7729);
	// multip_info_integrin.push_back(    0.7933);//0.8480);

	vector<double> multip_info_integrin_apical;
	multip_info_integrin_apical.push_back(1.0);//0.5116);//0.6316);
    multip_info_integrin_apical.push_back(1.0);//    0.5345);//0.5806);
    multip_info_integrin_apical.push_back(1.0);//    0.5698);//0.5933);
    multip_info_integrin_apical.push_back(1.0);//    0.5275);//0.7223);
    multip_info_integrin_apical.push_back(1.0);//    0.5964);//0.7116);
    multip_info_integrin_apical.push_back(1.0);//    0.6576);//0.5848);
    multip_info_integrin_apical.push_back(1.0);//    0.7648);//0.5480);
    multip_info_integrin_apical.push_back(1.0);//    0.6967);//0.6923);
    multip_info_integrin_apical.push_back(1.0);//    0.7585);//0.7782);
    multip_info_integrin_apical.push_back(1.0);//    0.8361);//0.9288);
    multip_info_integrin_apical.push_back(1.0);//    0.6972);//0.8486);
	for (int h = 0; h < multip_info_integrin_apical.size();h++){
		cout<<"Apical integrin scaling factor = "<<multip_info_integrin_apical[h]<<endl;
	}

	vector<CVector> initIntnlPosTmp;

	assert(initCenters.size() == initGrowProg.size());
	
	uint resumeSimulation = globalConfigVars.getConfigValue(
			"ResumeSimulation").toInt();
	uint initMembrNodeCount = globalConfigVars.getConfigValue(
			"InitMembrNodeCount").toInt();
	uint maxMembrNodeCountPerCell = globalConfigVars.getConfigValue(
			"MaxMembrNodeCountPerCell").toInt();
	uint maxIntnlNodeCountPerCell = globalConfigVars.getConfigValue(
			"MaxIntnlNodeCountPerCell").toInt();
	std::string MembraneNodesFileName = globalConfigVars.getConfigValue(
			"MembraneNodes_FileName").toString() ;
	std::string intnlNodesFileNameResume = globalConfigVars.getConfigValue(
			"IntnlNodes_FileName_Resume").toString() ;
	std::string membNodesFileNameResume = globalConfigVars.getConfigValue(
			"MembraneNodes_FileName_Resume").toString() ;
    std::string uniqueSymbol=globalConfigVars.getConfigValue(
	        "UniqueSymbol").toString() ;

	if (resumeSimulation==0) {
		cout<< " The simulation is in start mode" << endl ; 
		for (uint i = 0; i < initCenters.size(); i++) {
			//	initMembrPosTmp = generateInitMembrNodes(initCenters[i],   // to generate  membrane node positions
			//			initGrowProg[i]);
			//	initIntnlPosTmp = generateInitIntnlNodes(initCenters[i],   // to generate internal node positions
			//			initGrowProg[i]);
			initIntnlPosTmp = generateInitIntnlNodes_M(initCenters[i],   // to generate internal node positions
				initGrowProg[i],int(i)); //Ali
			//	initMembrPos.push_back(initMembrPosTmp);
			initIntnlPos.push_back(initIntnlPosTmp);
		}

		initMembrPos=readMembNodes(initCenters.size(),maxMembrNodeCountPerCell,mTypeV2, mDppV2, MembraneNodesFileName ); 	
		// initMembrMultip=readMembNodes_multip_info(initCenters.size(),maxMembrNodeCountPerCell,mTypeV2, mDppV2, MembraneNodesFileName, multip_info, multip_info2, multip_info_ECM, multip_info_ECM_apical ); 	
		initMembrMultip_actomyo=readMembNodes_multip_actomyo(initCenters.size(),maxMembrNodeCountPerCell,mTypeV2, mDppV2, MembraneNodesFileName, multip_info, multip_info2); 
		initMembrMultip_integrin=readMembNodes_multip_integrin(initCenters.size(),maxMembrNodeCountPerCell,mTypeV2, mDppV2, MembraneNodesFileName, multip_info_integrin, multip_info_integrin_apical ); 		
	}
	else if (resumeSimulation==1) {
		cout<< " The simulation is in Resume mode" << endl ;

        std::string intnlFileName = "./resources/" + intnlNodesFileNameResume + uniqueSymbol + "Resume.cfg";
		initIntnlPos=readResumeIntnlNodes( initCenters.size(),maxIntnlNodeCountPerCell,intnlFileName) ;  

        std:: string membFileName = "./resources/" + membNodesFileNameResume  + uniqueSymbol + "Resume.cfg";
		initMembrPos=readMembNodes(initCenters.size(),maxMembrNodeCountPerCell,mTypeV2, mDppV2, membFileName); 			
		// initMembrMultip=readMembNodes_multip_info(initCenters.size(),maxMembrNodeCountPerCell,mTypeV2, mDppV2, membFileName, multip_info, multip_info2, multip_info_ECM, multip_info_ECM_apical); 			
		initMembrMultip_actomyo=readMembNodes_multip_actomyo(initCenters.size(),maxMembrNodeCountPerCell,mTypeV2, mDppV2, MembraneNodesFileName, multip_info, multip_info2); 	
		initMembrMultip_integrin=readMembNodes_multip_integrin(initCenters.size(),maxMembrNodeCountPerCell,mTypeV2, mDppV2, MembraneNodesFileName, multip_info_integrin, multip_info_integrin_apical ); 	
	}
	else {
		throw std::invalid_argument("ResumeSimulation parameter in the input file must be either 1 or 0"); 
	}

		
}

double CellInitHelper::getRandomNum(double min, double max) {
	double rand01 = rand() / ((double) RAND_MAX + 1);
	double randNum = min + (rand01 * (max - min));
	return randNum;
}

vector<CVector> CellInitHelper::generateInitCellNodes() {
	bool isSuccess = false;
	vector<CVector> attemptedPoss;
	while (!isSuccess) {
		attemptedPoss = tryGenInitCellNodes();
		if (isPosQualify(attemptedPoss)) {
			isSuccess = true;
		}
	}
	// also need to make sure center point is (0,0,0).
	CVector tmpSum(0, 0, 0);
	for (uint i = 0; i < attemptedPoss.size(); i++) {
		tmpSum = tmpSum + attemptedPoss[i];
	}
	tmpSum = tmpSum / (double) (attemptedPoss.size());
	for (uint i = 0; i < attemptedPoss.size(); i++) {
		attemptedPoss[i] = attemptedPoss[i] - tmpSum;
	}
	return attemptedPoss;
}

//Not Active function
vector<CVector> CellInitHelper::generateInitIntnlNodes(CVector& center,
		double initProg) {
	bool isSuccess = false;

	uint minInitNodeCount =
			globalConfigVars.getConfigValue("InitCellNodeCount").toInt();
	uint maxInitNodeCount = globalConfigVars.getConfigValue(
			"MaxIntnlNodeCountPerCell").toInt();
//Ali

//	uint initIntnlNodeCt = minInitNodeCount ; 
//Ali
//Ali comment
	uint initIntnlNodeCt = minInitNodeCount
			+ (maxInitNodeCount - minInitNodeCount) * initProg;

	vector<CVector> attemptedPoss;
	while (!isSuccess) {
		attemptedPoss = tryGenInitCellNodes(initIntnlNodeCt);
		if (isPosQualify(attemptedPoss)) {
			isSuccess = true;
		}
	}
	/*
	 // also need to make sure center point is (0,0,0).
	 CVector tmpSum(0, 0, 0);
	 for (uint i = 0; i < attemptedPoss.size(); i++) {
	 tmpSum = tmpSum + attemptedPoss[i];
	 }
	 tmpSum = tmpSum / (double) (attemptedPoss.size());
	 for (uint i = 0; i < attemptedPoss.size(); i++) {
	 attemptedPoss[i] = attemptedPoss[i] - tmpSum;
	 }
	 */

	 // Input for nuclear pattern
double initRadius =
			globalConfigVars.getConfigValue("InitMembrRadius").toDouble();

	//double	noiseNucleusY=getRandomNum(0.4*initRadius,3.5*initRadius);  
	double	noiseNucleusY=getRandomNum(-0.2*initRadius,0.2*initRadius);  
		center.y=center.y+ noiseNucleusY ; 


	for (uint i = 0; i < attemptedPoss.size(); i++) {
		attemptedPoss[i] = attemptedPoss[i] + center;
	}
	return attemptedPoss;
}

vector<CVector> CellInitHelper::generateInitIntnlNodes_M(CVector& center,
		double initProg, int cellRank) {
	bool isSuccess = false;

	uint minInitNodeCount =
			globalConfigVars.getConfigValue("InitCellNodeCount").toInt();
	uint maxInitNodeCount = globalConfigVars.getConfigValue(
			"MaxIntnlNodeCountPerCell").toInt();
//Ali

//	uint initIntnlNodeCt = minInitNodeCount ; 
//Ali
//Ali comment
	uint initIntnlNodeCt = minInitNodeCount
			+ (maxInitNodeCount - minInitNodeCount) * initProg;

	vector<CVector> attemptedPoss;
	while (!isSuccess) {
		attemptedPoss = tryGenInitCellNodes(initIntnlNodeCt);
		if (isPosQualify(attemptedPoss)) {
			isSuccess = true;
		}
	}
		 // Input for nuclear pattern
	double initRadius =
			globalConfigVars.getConfigValue("InitMembrRadius").toDouble();
	
	double	noiseNucleusY ; 
	if ( cellRank>1 && cellRank<63) {
		double std=0.16 ; 
		double mean=0.6 ;
		double pouchCellH=25 ; 
		double upperLimit=mean+std ; 
		double lowerLimit=mean-std  ;
		// generate random number in y direction respect to cell center (0.5*pouhCellH)
		noiseNucleusY=getRandomNum( (upperLimit-0.5)*pouchCellH
		                           , (0.5-lowerLimit)*pouchCellH ); 
		//noiseNucleusY=getRandomNum(-0.75*initRadius,7.25*initRadius); 
	}
	else {

		noiseNucleusY=getRandomNum(-0.25*initRadius,0.25*initRadius);  
	}

	center.y=center.y+ noiseNucleusY ; 


	for (uint i = 0; i < attemptedPoss.size(); i++) {
		attemptedPoss[i] = attemptedPoss[i] + center;
	}
	return attemptedPoss;
}


vector<CVector> CellInitHelper::generateInitMembrNodes(CVector& center,
		double initProg) {
	double initRadius =
			globalConfigVars.getConfigValue("InitMembrRadius").toDouble();
	uint initMembrNodeCount = globalConfigVars.getConfigValue(
			"InitMembrNodeCount").toInt();
	vector<CVector> initMembrNodes;
	double unitAngle = 2 * acos(-1.0) / (double) (initMembrNodeCount);
	for (uint i = 0; i < initMembrNodeCount; i++) {
		CVector node;
		node.x = initRadius * cos(unitAngle * i) + center.x;
		node.y = initRadius * sin(unitAngle * i) + center.y;  //actual assignment
		initMembrNodes.push_back(node);
	}
	return initMembrNodes;
}

//I don't think it is active now
vector<CVector> CellInitHelper::tryGenInitCellNodes() { 
	// Not active right now 
	double radius =
			globalConfigVars.getConfigValue("InitCellRadius").toDouble();
	//int initCellNodeCount =
	//    globalConfigVars.getConfigValue("InitCellNodeCount").toInt();
	// now we need non-uniform initial growth progress.
	int initCellNodeCount =
			globalConfigVars.getConfigValue("InitCellNodeCount").toInt();
	vector<CVector> poss;
	int foundCount = 0;
	double randX, randY;
	while (foundCount < initCellNodeCount) {
		bool isInCircle = false;
               //Ali
               cout << "I am in the wrong one" << endl ; 
               //Ali
		while (!isInCircle) {
			randX = getRandomNum(-0.2*radius, 0.2*radius);
			randY = getRandomNum(-0.2*radius, 0.2*radius);
			isInCircle = (sqrt(randX * randX + randY * randY) < 0.2*radius);
		}
		poss.push_back(CVector(randX, randY, 0));
		foundCount++;
	}
	return poss;
}

//It is active
vector<CVector> CellInitHelper::tryGenInitCellNodes(uint initNodeCt) {
	double radius =
			globalConfigVars.getConfigValue("InitCellRadius").toDouble();
	vector<CVector> poss;
	uint foundCount = 0;
	double randX, randY;

               //Ali
            //   cout << "I am in the right one" << endl ; 
             //  cout << "# of internal Nodes" << initNodeCt <<endl ; 
               //Ali
	while (foundCount < initNodeCt) {
		bool isInCircle = false;
		//while (!isInCircle) {
			randX = getRandomNum(-0.3*radius, 0.3*radius);
			randY = getRandomNum(-0.3*radius, 0.3*radius);
			isInCircle = (sqrt(randX * randX + randY * randY) < radius);
	//	}
                //Ali
                 if (isInCircle) {
                //Ali
		 poss.push_back(CVector(randX, randY, 0));
		 foundCount++;
         //      cout << "#internal nodes location" << foundCount<<"isInCircle"<<isInCircle <<endl ; 
               //Ali
                 }
               //Ali 
	}
	return poss;
}

bool CellInitHelper::isPosQualify(vector<CVector>& poss) {
	double minInitDist = globalConfigVars.getConfigValue(
			"MinInitDistToOtherNodes").toDouble();
	bool isQualify = true;
	for (uint i = 0; i < poss.size(); i++) {
		for (uint j = 0; j < poss.size(); j++) {
			if (i == j) {
				continue;
			} else {
				CVector distDir = poss[i] - poss[j];
				if (distDir.getModul() < minInitDist) {
					isQualify = false;
					return isQualify;
				}
			}
		}
	}
	return isQualify;
}

bool CellInitHelper::isMXType(CVector position) {
	if (simuType != Beak) {
		return true;
	}
	if (position.y >= internalBdryPts[0].y) {
		return false;
	}
	if (position.x >= internalBdryPts[2].x) {
		return false;
	}
	for (uint i = 0; i < internalBdryPts.size() - 1; i++) {
		CVector a = internalBdryPts[i + 1] - internalBdryPts[i];
		CVector b = position - internalBdryPts[i];
		CVector crossProduct = Cross(a, b);
		if (crossProduct.z > 0) {
			return false;
		}
	}
	return true;
}

/* beak simulation is not active and I want to remove dependency of the code on CGAL
void CellInitHelper::initInternalBdry() {
	GEOMETRY::MeshGen meshGen;
	GEOMETRY::MeshInput input = meshGen.obtainMeshInput();
	internalBdryPts = input.internalBdryPts;
}
*/
/* This function is used for DiskMain project which is not active and I want to remove the dependency of the code on CGAL
SimulationInitData_V2 CellInitHelper::initStabInput() {
	RawDataInput rawInput = generateRawInput_stab();
	SimulationInitData_V2 initData = initInputsV3(rawInput);
	initData.isStab = true;
	return initData;
}
*/
//RawDataInput rawInput = generateRawInput_stab();
SimulationInitData_V2_M CellInitHelper::initInput_M() {   //Ali: This function is called by the main function of the code which is discMain_M.cpp
	RawDataInput_M rawInput_m = generateRawInput_M();     //Ali: This function includes reading cell centers and membrane nodes locations
	SimulationInitData_V2_M initData = initInputsV3_M(rawInput_m); // This function reformat the files read in the input to be easily movable to GPU
	initData.isStab = false;
	return initData;
}
/* this function is needed for laserAblation and discMain. cpp which none of them are acitve and I want to remove dependency on CGAL
SimulationInitData_V2 CellInitHelper::initSimuInput(
		std::vector<CVector> &cellCenterPoss) {
	RawDataInput rawInput = generateRawInput_simu(cellCenterPoss);  //This function call MeshGen which is heavily using CGAL
	SimulationInitData_V2 simuInitData = initInputsV3(rawInput);
	simuInitData.isStab = false;
	return simuInitData;
}
*/
void SimulationGlobalParameter::initFromConfig() {
	int type = globalConfigVars.getConfigValue("SimulationType").toInt();
	SimulationType simuType = parseTypeFromConfig(type);

	animationNameBase =
			globalConfigVars.getConfigValue("AnimationFolder").toString()
					+ globalConfigVars.getConfigValue("AnimationName").toString()
					+ globalConfigVars.getConfigValue("UniqueSymbol").toString();
        //A & A 
	InitTimeStage=
			globalConfigVars.getConfigValue("InitTimeStage").toDouble();
        //A & A 
	totalSimuTime =
			globalConfigVars.getConfigValue("SimulationTotalTime").toDouble();

	dt = globalConfigVars.getConfigValue("SimulationTimeStep").toDouble();
	Damp_Coef= globalConfigVars.getConfigValue("DampingCoef").toDouble();

	totalTimeSteps = totalSimuTime / dt;

	totalFrameCount =
			globalConfigVars.getConfigValue("TotalNumOfOutputFrames").toInt();

	aniAuxVar = totalTimeSteps / totalFrameCount;

	aniCri.pairDisplayDist = globalConfigVars.getConfigValue(
			"IntraLinkDisplayRange").toDouble();

	aniCri.animationType = parseAniTpFromConfig(
			globalConfigVars.getConfigValue("AnimationType").toInt());

	aniCri.threshold = globalConfigVars.getConfigValue("DeltaValue").toDouble();
	if (simuType != Disc_M) {
		aniCri.arrowLength = globalConfigVars.getConfigValue(
				"DisplayArrowLength").toDouble();
	}

	if (simuType != Beak && simuType != Disc_M) {
		dataOutput =
				globalConfigVars.getConfigValue("PolygonStatFileName").toString()
						+ globalConfigVars.getConfigValue("UniqueSymbol").toString()
						+ ".txt";
		imgOutput =
				globalConfigVars.getConfigValue("DataOutputFolder").toString()
						+ globalConfigVars.getConfigValue("ImgSubFolder").toString()
						+ globalConfigVars.getConfigValue("ImgFileNameBase").toString();
		dataFolder = globalConfigVars.getConfigValue("DataFolder").toString();
		dataName = dataFolder
				+ globalConfigVars.getConfigValue("DataName").toString();
	}

}



vector<vector<CVector> >  CellInitHelper::readResumeIntnlNodes(int numCells, int maxIntnlNodeCountPerCell, string intnlFileName) {

   CVector iCoordinate ;
   vector <CVector> intnlPosTmp ;
   vector<vector<CVector> > intnlPos ;
   ifstream inputc ; 
   inputc.open(intnlFileName.c_str());
   if (inputc.is_open()){
      cout << "File for reading internal nodes coordinates in resume mode is opened successfully " << endl ; 
   }
   else{
      cout << "Failed opening internal nodes coordinates in resme mode " << endl ; 
   }

   int cellIDOld=-1;
   int cellID ;
   for (int j=0 ; j<numCells  ; j++) {
		intnlPosTmp.clear() ;
		cellIDOld++  ;
		if (j!=0) {
	    	intnlPosTmp.push_back(iCoordinate);
		}
        for (int i = 0; i <maxIntnlNodeCountPerCell; i++) {
	    	inputc >> cellID >> iCoordinate.x >> iCoordinate.y >> iCoordinate.z  ;
			if (cellID != cellIDOld) {
				break ;// for reading the next cell's internal coordinates
			}
			if (inputc.eof()) {  
				break ; // to not push backing data when the read file is finished.
			}
	    	intnlPosTmp.push_back(iCoordinate);
		//	cout <<"cell ID= "<<cellID<<"x internal= "<<iCoordinate.x << " y internal= "<<iCoordinate.y <<" z internal="<<iCoordinate.z << endl ;  
      	}
		intnlPos.push_back(intnlPosTmp);
	}
	cout << " I read internal nodes successfully in resume mode" << endl ; 	
	
	return intnlPos ; 
}

  

vector<vector<CVector> > CellInitHelper::readMembNodes(int numCells,int maxMembrNodeCountPerCell,
                                                           vector<vector<MembraneType1> >& mTypeV2,vector<vector<double> >& mDppV2, string membFileName) 
{
    vector<CVector> initMembrPosTmp;
    vector<vector<CVector> > initMembrPos ; 
    vector<double> mDppVTmp;  
    vector<MembraneType1> mTypeVTmp;  
    std::fstream inputc;

    inputc.open(membFileName.c_str());
    if (inputc.is_open()){
       cout << "File for reading membrane nodes coordinates opened successfully " << endl ; 
    }
	else{
       cout << "failed opening membrane nodes coordinates " << endl ; 
    }

	int cellIDOld=-1;
	int cellID ;
    CVector mCoordinate ;
	double mDpp ; 
	string mTypeString ; 
	MembraneType1 mType ; 
	for (int j=0 ; j<numCells ; j++) {
		initMembrPosTmp.clear() ;
		mDppVTmp.clear() ; 
		mTypeVTmp.clear() ; 
		cellIDOld++  ;
		if (j!=0) {
	    	initMembrPosTmp.push_back(mCoordinate);
	    	mDppVTmp.push_back(mDpp);
	    	mTypeVTmp.push_back(mType);
		}
        for (int i = 0; i <maxMembrNodeCountPerCell; i++) {
	    	inputc >> cellID >> mCoordinate.x >> mCoordinate.y >> mDpp >> mTypeString ;
			mType =StringToMembraneType1Convertor (mTypeString) ; 
			if (cellID != cellIDOld) {
				break ;// for reading the next cell's membrane coordinates
			}
			
			if (inputc.eof()) {  
				break ; // to not push backing data when the read file is finished.
			}
	    	initMembrPosTmp.push_back(mCoordinate);
	    	mDppVTmp.push_back(mDpp);
	    	mTypeVTmp.push_back(mType);
	//		cout <<"cell ID= "<<cellID<<"x membrane= "<<mCoordinate.x << " y membrane= "<<mCoordinate.y <<" dpp level=" << mDpp <<" type membrane="<<mType << endl ;  
      	}
		initMembrPos.push_back(initMembrPosTmp);
		mDppV2.push_back(mDppVTmp);
		mTypeV2.push_back(mTypeVTmp);
	}
	cout << " I read membrane nodes successfully" << endl ; 	
	return initMembrPos ;  
}


vector<vector<CVector> > CellInitHelper::readMembNodes_multip_actomyo(int numCells,int maxMembrNodeCountPerCell,
                                                           vector<vector<MembraneType1> >& mTypeV2,vector<vector<double> >& mDppV2, string membFileName, vector<double> multip_info, vector<double> multip_info2) 
{
	vector<CVector> initMembrMultipTmp;
	vector<vector<CVector> > initMembrMultip ; 
    std::fstream inputc;

    inputc.open(membFileName.c_str());
    if (inputc.is_open()){
       cout << "File for reading membrane nodes coordinates opened successfully " << endl ; 
    }
	else{
       cout << "failed opening membrane nodes coordinates " << endl ; 
    }

	int cellIDOld=-1;
	int cellID ;
	CVector dummy;
	CVector mMultip_info;
	double mDpp ; 
	string mTypeString ; 
	MembraneType1 mType ; 
	for (int j=0 ; j<numCells ; j++) {
		initMembrMultipTmp.clear();
		cellIDOld++  ;
		if (j!=0) {
			initMembrMultipTmp.push_back(mMultip_info);
		}
        for (int i = 0; i <maxMembrNodeCountPerCell; i++) {
	    	inputc >> cellID >> dummy.x >> dummy.y >> mDpp >> mTypeString ;
			mType =StringToMembraneType1Convertor (mTypeString) ; 
			if (cellID != cellIDOld) {
				break ;// for reading the next cell's membrane coordinates
			}
			
			if (inputc.eof()) {  
				break ; // to not push backing data when the read file is finished.
			}
			if (cellID>1 && cellID <63){
				mMultip_info.x = multip_info[cellID-2];// 
				mMultip_info.y = 0.0;//multip_info2[cellID-2];// pow(multip_info2[cellID-2],1.0);
			}
			else{
				mMultip_info.x = 0.0;//0.0;
				mMultip_info.y = 0.0;
			}
			if (j == 1 || j == 10 || j == 32 || j == 70 || j == 78){
				// std::cout<<"cell["<<cellID<<"] with nodeID("<<i<<") apical & basal actomyo multiplier = "<<mMultip_info.y<<" & "<<mMultip_info.x<<std::endl;
			}
			initMembrMultipTmp.push_back(mMultip_info);
	//		cout <<"cell ID= "<<cellID<<"x membrane= "<<mCoordinate.x << " y membrane= "<<mCoordinate.y <<" dpp level=" << mDpp <<" type membrane="<<mType << endl ;  
      	}
		initMembrMultip.push_back(initMembrMultipTmp);
	
	}
	cout << " I read membrane nodes successfully" << endl ; 	
	return initMembrMultip ;  
}

vector<vector<CVector> > CellInitHelper::readMembNodes_multip_integrin(int numCells,int maxMembrNodeCountPerCell,
                                                           vector<vector<MembraneType1> >& mTypeV2,vector<vector<double> >& mDppV2, string membFileName, vector<double> multip_info_ECM, vector<double> multip_info_ECM_apical) 
{
	vector<CVector> initMembrMultipTmp;
	vector<vector<CVector> > initMembrMultip ; 
    std::fstream inputc;

    inputc.open(membFileName.c_str());
    if (inputc.is_open()){
       cout << "File for reading membrane nodes coordinates opened successfully " << endl ; 
    }
	else{
       cout << "failed opening membrane nodes coordinates " << endl ; 
    }

	int cellIDOld=-1;
	int cellID ;
	CVector dummy;
	CVector mMultip_info;
	double mDpp ; 
	string mTypeString ; 
	MembraneType1 mType ; 
	for (int j=0 ; j<numCells ; j++) {
		initMembrMultipTmp.clear();
		cellIDOld++  ;
		if (j!=0) {
			initMembrMultipTmp.push_back(mMultip_info);
		}
        for (int i = 0; i <maxMembrNodeCountPerCell; i++) {
	    	inputc >> cellID >> dummy.x >> dummy.y >> mDpp >> mTypeString ;
			mType =StringToMembraneType1Convertor (mTypeString) ; 
			if (cellID != cellIDOld) {
				break ;// for reading the next cell's membrane coordinates
			}
			
			if (inputc.eof()) {  
				break ; // to not push backing data when the read file is finished.
			}
			if (cellID>1 && cellID <63){
				mMultip_info.x = pow(multip_info_ECM[cellID-2],1.0);
				// mMultip_info.y = 0.0;
			}
			else if (cellID>69 && cellID<81){
				// mMultip_info.x = 0.0;//0.0;
				mMultip_info.x = pow(multip_info_ECM_apical[cellID%70],1.0);
			}
			else{
				mMultip_info.x = 1.0;//0.5;;
				// mMultip_info.y = 0.5;
			}
			if (j == 1 || j == 10 || j == 32 || j == 70 || j == 78){
				// std::cout<<"cell["<<cellID<<"] with nodeID("<<i<<") integrin multiplier = "<<mMultip_info.x<<std::endl;//" & "<<mMultip_info.y<<std::endl;
			}
			initMembrMultipTmp.push_back(mMultip_info);
	//		cout <<"cell ID= "<<cellID<<"x membrane= "<<mCoordinate.x << " y membrane= "<<mCoordinate.y <<" dpp level=" << mDpp <<" type membrane="<<mType << endl ;  
      	}
		initMembrMultip.push_back(initMembrMultipTmp);
	
	}
	cout << " I read membrane nodes successfully" << endl ; 	
	return initMembrMultip ;  
}


/* CGAL DEACTIVATION
RawDataInput CellInitHelper::generateRawInput_singleCell() {
	RawDataInput rawData;
	rawData.simuType = simuType;

	std::string initPosFileName = globalConfigVars.getConfigValue(
			"SingleCellCenterPos").toString();

	fstream fs(initPosFileName.c_str());
	vector<CVector> insideCellCenters = GEOMETRY::MeshInputReader::readPointVec(
			fs);
	fs.close();

	for (unsigned int i = 0; i < insideCellCenters.size(); i++) {
		CVector centerPos = insideCellCenters[i];
		rawData.MXCellCenters.push_back(centerPos);
	}

	generateCellInitNodeInfo_v2(rawData.initCellNodePoss);
	rawData.isStab = true;
	return rawData;
}
*/
/* CGAL deactivation
SimulationInitData_V2 CellInitHelper::initSingleCellTest() {
	RawDataInput rawInput = generateRawInput_singleCell();
	SimulationInitData_V2 initData = initInputsV3(rawInput);
	initData.isStab = true;
	return initData;
}
*/
