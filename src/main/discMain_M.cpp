//============================================================================
// Name        : Main.cpp
// Author      : Wenzhao Sun, Ali Nematbakhsh
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>

//#include "MeshGen.h"
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
//test here
void initializeSlurmConfig(int argc, char* argv[]) {
	// read configuration.
	ConfigParser parser;
	std::string configFileNameDefault = "./resources/disc_M.cfg";
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
		        int myDeviceID =
				globalConfigVars.getConfigValue("GPUDeviceNumber").toInt();
		        gpuErrchk(cudaSetDevice(myDeviceID));
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

void updateDivThres(double& curDivThred, uint& i, double& curTime,  //Ali
		double& decayCoeff, double& divThreshold) {
        
        //cout<<"The value of initial time stage in updateDivThres is"<<curTime<<endl ;  
	double decay = exp(-curTime * decayCoeff);
	//curDivThred = 1.0 - (1.0 - divThreshold) * decay;
	curDivThred = divThreshold ;
}

int main(int argc, char* argv[]) {
	// initialize random seed.
	srand(time(NULL));

	// Slurm is computer-cluster management system.
	initializeSlurmConfig(argc, argv);

        cout<< "I am in main file after slurm "<<endl; 
	// initialize simulation control related parameters from config file.
	SimulationGlobalParameter mainPara;

        cout<< "I am in main file after simulation global parameter "<<endl; 
	mainPara.initFromConfig();

        cout<< "I am in main file before Cell IniHelper instance creation"<<endl; 
	// initialize simulation initialization helper.
	CellInitHelper initHelper;
        cout<< "I am in main file after Cell IniHelper instance creation"<<endl; 

	// initialize simulation domain.
	SimulationDomainGPU simuDomain;

        cout<< "I am in main file after simulationDomainGPU instance creation"<<endl;

	// comment: initInput_M() goes in cellInitHelper.cpp. There the information are read from input txt files and saved in CPU of the computer.
	// Then initialize_v2_M will go to the simuDomain.cu and initially create some vector spaces for GPU values and then assigned the CPU values to GPU vectors either in SceNodes or SceCells. The next two functions below are the main functions for initialization of the code from txt and transfer them to GPU.
	SimulationInitData_V2_M initData = initHelper.initInput_M(); // it will go inside cellinithelper.cpp  Ali 

        cout<< "I am in main file after initInput_M creation"<<endl; 
	simuDomain.initialize_v2_M(initData,mainPara.InitTimeStage);
	
        cout<< "I am in main file after initInput_v2_M creation"<<endl; 
	std::string polyStatFileNameBase = globalConfigVars.getConfigValue(
			"PolygonStatFileName").toString();
	std::string uniqueSymbol =
			globalConfigVars.getConfigValue("UniqueSymbol").toString();
	std::string polyStatFileName = polyStatFileNameBase + uniqueSymbol + ".txt";

	std::remove(polyStatFileName.c_str());

	std::string detailStatFileNameBase = globalConfigVars.getConfigValue(
			"DetailStatFileNameBase").toString() + uniqueSymbol;
	double divThreshold =
			globalConfigVars.getConfigValue("DivThreshold").toDouble();
	double decayCoeff =
			globalConfigVars.getConfigValue("ProlifDecayCoeff").toDouble();
	double curDivThred;

	int maxStepTraceBack =
			globalConfigVars.getConfigValue("MaxStepTraceBack").toInt();

	uint aniFrame = 0;
	// main simulation steps.
    std::string stressStrainFileNameBase="StressStrain" ;
    std::string stressStrainFileName=stressStrainFileNameBase +uniqueSymbol+ ".CSV" ; 
    SingleCellData singleCellData(stressStrainFileName);
	double current_Time;
	double total_Time = mainPara.totalTimeSteps*mainPara.dt;
	std::cout<<"mainPara.totalTimeStpes = "<<mainPara.totalTimeSteps<<std::endl;
	std::cout<<"mainPara.dt = "<<mainPara.dt<<std::endl;
	double timeRatio;
	double timeRatio_Crit_actomyo = 2.0;
	// std::cout<<"Critical timeRatio for actomyosin strength reduction = "<<timeRatio_Crit_actomyo<<std::endl;
	bool reduced_actomyo_triggered = false;
	double timeRatio_Crit_ECM = 2.0;
	// std::cout<<"Critical timeRatio for ecm strength reduction = "<<timeRatio_Crit_ECM<<std::endl;
	double timeRatio_Crit_Division = 0.2;
	// std::cout<<"Critical timeRatio for growth and cell division = "<<timeRatio_Crit_Division<<std::endl;
	bool reduced_ecm_triggered = false;
	bool Division_triggered = false;
	double volume_Increase_Target_Ratio = 1.41;
	std::cout<<"Target ratio for cell volume increase = "<<volume_Increase_Target_Ratio<<std::endl;
	double volume_Increase_Scale = 1.0;
	std::cout<<"How fast is the volume increase happening = x"<<volume_Increase_Scale<<" rate of change"<<std::endl;
	double postDivision_restorationRateScale = 1.0;//0.5;//0.1;
	//I changed postDivision_restorationRateScale to 2.0 due to extended simulation steps (1000/0.002), make sure this is changed back
	//later when using a smaller number of simulation time steps.
	std::cout<<"How fast is the volume restoration happening post division = x"<<postDivision_restorationRateScale<<" rate of change"<<std::endl;
	bool volume_restoration_rate_restore = false;

	double thresholdToIntroduceNewCell = 0.25;//-1.0;
	std::cout<<"The likelihood (probability) a new cell will be introduced in the same cross section after cell division = "<<thresholdToIntroduceNewCell<<std::endl;

	double distFromNucleus_max = 0.0;//4.0;
	double distFromNucleus_min = 0.3;//-14.0;
	double distFromNucleus_normalMax = 0.20;//0.355;//-8.0;
	double distFromNucleus_normalMax_apical = 0.20;//0.295;//7.5;
	double percentage_before_timeRatio_Crit_Division_scaling = 4.0; //No longer in use, but don't delete yet since it is still passed into several functions. //Kevin
	double mitoRndActomyoStrengthScaling = 5.0; //Please consult SceNodes.cu to see what are the scaling applied to non-dividing cells and disc_NXX_X.cfg file to see what is the corresponding default spring constant. //Kevin
	std::cout<<"Cell division requires the contractile spring to increase strength by "<<percentage_before_timeRatio_Crit_Division_scaling<<" fold."<<std::endl;
	std::cout<<"Contractile spring minimum at "<<distFromNucleus_min<<" and maximum at "<<distFromNucleus_max<<" away from the cell center."<<std::endl;
	std::cout<<"But under non-growth (stationary) circumstances, the basal contractile spring is set at "<<distFromNucleus_normalMax<<"*cellheight away from the cell center."<<std::endl;
	std::cout<<"But under non-growth (stationary) circumstances, the apical contractile spring is set at "<<distFromNucleus_normalMax_apical<<"*cellheight away from the cell center."<<std::endl;

	double growthProgressSpeed = 3.84e-7;//3.84e-7;//(0.02)*0.25*(0.001*0.002); // 0.002 is the default time step of the simulation.
	std::cout<<"Growth speed for cell cycle per time step is = "<<growthProgressSpeed<<std::endl;

	int maxApicalBasalNodeNum = 18;//9999;
	std::cout<<"Max number of apical and basal nodes, respectively, for columnar cells = "<<maxApicalBasalNodeNum<<std::endl;
	// int minApicalBasalNodeNum = 21;
	// std::cout<<"Min number of apical and basal nodes, respectively, for columnar cells = "<<minApicalBasalNodeNum<<std::endl;

	double maxLengthToAddMemNodes = 0.195;//0.26;
	std::cout<<"Max length for each edge to qualify for new node additions = "<<maxLengthToAddMemNodes<<std::endl;

	std::vector<int> cycle_vec;
	cycle_vec.push_back(-1);
	// uint maxNumCycle = 10;

	// for (int k = 0; k < maxNumCycle; k++){
	// 	cycle_vec.push_back(k);
	// }
	// if (maxNumCycle == 0){
	// 	cycle_vec.push_back(-1);
	// }
	// cycle_vec.push_back(0);
	// cycle_vec.push_back(1);
	// cycle_vec.push_back(2);
	int cycle;

	double fixed_dt = mainPara.dt;
	for (int i = 0; i < cycle_vec.size(); i++){	
		Division_triggered = false;
		cycle = cycle_vec[i];
		for (uint i = 0; i <= (uint) (mainPara.totalTimeSteps); i++) {
			
			// current_Time = i*mainPara.dt + mainPara.InitTimeStage;
			current_Time = i*fixed_dt + mainPara.InitTimeStage;
			timeRatio = current_Time/total_Time;
			// if (timeRatio > timeRatio_Crit_actomyo && reduced_actomyo_triggered == false){
			// 	std::cout<<"Reduced actomyosin strength in the tissue midsection triggered."<<std::endl;
			// 	std::cout<<"Current timeRatio for reduced actomyosin ="<<timeRatio<<std::endl;
			// 	reduced_actomyo_triggered = true;
			// }
			// if (timeRatio > timeRatio_Crit_ECM && reduced_ecm_triggered == false){
			// 	std::cout<<"Reduced ECM strength in the tissue midsection triggered."<<std::endl;
			// 	std::cout<<"Current timeRatio for reduced ecm ="<<timeRatio<<std::endl;
			// 	reduced_ecm_triggered = true;
			// }
			// if (timeRatio == timeRatio_Crit_Division && Division_triggered == false){
			// 	std::cout<<"Division triggered in discMain.cpp"<<std::endl;
			// 	std::cout<<"Current timeRatio for division ="<<timeRatio<<std::endl;
			// 	Division_triggered = true;
			// 	// mainPara.dt = 1e-6;
			// 	volume_restoration_rate_restore = false;
			// }
			// if (timeRatio > (timeRatio_Crit_Division+0.05)){
			// 	mainPara.dt = fixed_dt;
			// 	// if (volume_restoration_rate_restore == false){
			// 	// 	std::cout<<"Volume restoration rate restored to normal speed"<<std::endl;
			// 	// }
			// 	// volume_restoration_rate_restore = true;
			// 	// postDivision_restorationRateScale = 0.2;
			// 	// break;
			// }
			// this if is just for output data// 
			if (i % mainPara.aniAuxVar == 0) {
				// if (i == 0 && cycle != 0){

				// }
				// else{
					double curTime=i*mainPara.dt + mainPara.InitTimeStage;  //Ali - Abu
					updateDivThres(curDivThred, i, curTime, decayCoeff,divThreshold);

					// std::cout << "substep 1" << std::flush;

					CellsStatsData polyData = simuDomain.outputPolyCountData();  //Ali comment
					singleCellData=simuDomain.OutputStressStrain() ;              
					singleCellData.printStressStrainToFile(stressStrainFileName,curTime) ;
								

					// std::cout << "substep 2 " << std::endl;
					// prints brief polygon counting statistics to file
					polyData.printPolyCountToFile(polyStatFileName, curDivThred);
					// prints detailed individual cell statistics to file
					polyData.printDetailStatsToFile(detailStatFileNameBase, aniFrame);
					// prints the animation frames to file. They can be open by Paraview

					
					//if (i != 0) {
						//simuDomain.processT1Info(maxStepTraceBack, polyData);
					//}

					// std::cout << "substep 3 " << std::endl;
					//simuDomain.outputVtkFilesWithCri_M(mainPara.animationNameBase,
					//		aniFrame, mainPara.aniCri);
					//simuDomain.outputVtkColorByCell_T1(mainPara.animationNameBase,
					//		aniFrame, mainPara.aniCri);
					simuDomain.outputVtkColorByCell_polySide(mainPara.animationNameBase,
							aniFrame, mainPara.aniCri);
					// std::cout << "in ani step " << aniFrame << std::endl;
					// std::cout << "substep 4 " << std::endl;
					simuDomain.outputResumeData(aniFrame) ; 
					aniFrame++;
				// }
			}
			simuDomain.runAllLogic_M(mainPara.dt,mainPara.Damp_Coef,mainPara.InitTimeStage,
									timeRatio, timeRatio_Crit_actomyo, timeRatio_Crit_ECM, timeRatio_Crit_Division,
										volume_Increase_Target_Ratio, volume_Increase_Scale, postDivision_restorationRateScale, cycle,
										distFromNucleus_max, distFromNucleus_min, distFromNucleus_normalMax, distFromNucleus_normalMax_apical,
										percentage_before_timeRatio_Crit_Division_scaling, growthProgressSpeed, maxApicalBasalNodeNum, maxLengthToAddMemNodes,mitoRndActomyoStrengthScaling, thresholdToIntroduceNewCell);  //Ali //Kevin
			// simuDomain.runAllLogic_M(mainPara.dt,mainPara.Damp_Coef,mainPara.InitTimeStage,
			// 						timeRatio, timeRatio_Crit_actomyo, timeRatio_Crit_ECM, timeRatio_Crit_Division,
			// 							volume_Increase_Target_Ratio, volume_Increase_Scale, postDivision_restorationRateScale, cycle,
			// 							distFromNucleus_max, distFromNucleus_min, distFromNucleus_normalMax, distFromNucleus_normalMax_apical,
			// 							percentage_before_timeRatio_Crit_Division_scaling, growthProgressSpeed, maxApicalBasalNodeNum, minApicalBasalNodeNum, maxLengthToAddMemNodes);  //Ali //Kevin
		}
	}

	return 0;
}
