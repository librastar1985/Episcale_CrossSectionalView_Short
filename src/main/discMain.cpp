//============================================================================
// Name        : MeshGen.cpp
// Author      : Wenzhao Sun
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>

#include "MeshGen.h"
#include "commonData.h"
#include "CellInitHelper.h"
#include <vector>
#include "SimulationDomainGPU.h"

using namespace std;

GlobalConfigVars globalConfigVars;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort =
		true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
				line);
		if (abort)
			exit(code);
	}
}

void initializeSlurmConfig(int argc, char* argv[]) {
	// read configuration.
	ConfigParser parser;
	std::string configFileNameDefault = "./resources/disc_master.cfg";
	globalConfigVars = parser.parseConfigFile(configFileNameDefault);
	std::string configFileNameBaseL = "./resources/disc_";
	std::string configFileNameBaseR = ".cfg";

	// Unknown number of input arguments.
	if (argc != 1 && argc != 3) {
		std::cout << "ERROR: Incorrect input argument count.\n"
				<< "Expect either no command line argument or three arguments"
				<< std::endl;
		exit(0);
	}
	// one input argument. It has to be "-slurm".
	else if (argc == 3) {
		if (strcmp(argv[1], "-slurm") != 0) {
			std::cout
					<< "ERROR: one argument received from commandline but it's not recognized.\n"
					<< "Currently, the argument value must be -slurm"
					<< std::endl;
			exit(0);
		} else {
			std::string configFileNameM(argv[2]);
			std::string configFileNameCombined = configFileNameBaseL
					+ configFileNameM + configFileNameBaseR;
			parser.updateConfigFile(globalConfigVars, configFileNameCombined);
		}
	}
	// no input argument. Take default.
	else if (argc == 1) {

		// set GPU device.
		int myDeviceID =
				globalConfigVars.getConfigValue("GPUDeviceNumber").toInt();
		gpuErrchk(cudaSetDevice(myDeviceID));
	}
}

int main(int argc, char* argv[]) {
	/*
	// initialize random seed.
	srand(time(NULL));

	initializeSlurmConfig(argc, argv);

	// initialize simulation control related parameters from config file.
	SimulationGlobalParameter mainPara;
	mainPara.initFromConfig();

	// initialize post-processing related parameters from config file.
	PixelizePara pixelPara;
	pixelPara.initFromConfigFile();

	// initialize simulation initialization helper.
	CellInitHelper initHelper;
	// initialize simulation domain.
	SimulationDomainGPU simuDomain;

	// initialize stabilization inputs.
	SimulationInitData_V2 initData = initHelper.initStabInput();
	// Stabilize the initial inputs and output stablized inputs.
	std::vector<CVector> stabilizedCenters = simuDomain.stablizeCellCenters(
			initData);

	// Generate initial inputs for simulation domain.
	SimulationInitData_V2 simuData2 = initHelper.initSimuInput(
			stabilizedCenters);
	// initialize domain based on initial inputs.
	simuDomain.initialize_v2(simuData2);

	// delete old data file.
	std::remove(mainPara.dataOutput.c_str());

	// preparation.
	uint aniFrame = 0;
	// main simulation steps.
	for (int i = 0; i <= mainPara.totalTimeSteps; i++) {
		cout << "step number = " << i << endl;
		if (i % mainPara.aniAuxVar == 0) {
			simuDomain.outputVtkFilesWithCri(mainPara.animationNameBase,
					aniFrame, mainPara.aniCri);
			cout << "finished output Animation" << endl;
			vector<vector<int> > labelMatrix = simuDomain.outputLabelMatrix(
					mainPara.dataName, aniFrame, pixelPara);
			cout << "finished writing label matrix" << endl;
			simuDomain.analyzeLabelMatrix(labelMatrix, aniFrame,
					mainPara.imgOutput, mainPara.dataOutput);
			cout << "finished output matrix analysis" << endl;
			aniFrame++;
		}
		// for each step, run all logics of the domain.
		simuDomain.runAllLogic(mainPara.dt);
	}
	*/
	return 0;
}
