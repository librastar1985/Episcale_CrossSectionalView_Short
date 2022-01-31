CUDA code for developmental biology using Subcellular Element Method

Hardware requirement: 
Nvidia video card that supports SM 2.0+ and CUDA 4.0 

Software environment requirement: 
CMAKE ----- Build system.
CUDA  ----- Provide runtime support for parallel GPU computation.
CGAL  ----- Computational geometry library.
Thrust ---- Build-in library of cuda, similar to STL of C++
Paraview -- (Optional) Visualization software for animation purpose. 

To compile:
 (1) In project root folder, type "cmake ." ("sudo cmake ." preferred)
 (2) type "make" 
Please note that CMake, CUDA, CGAL, Thrust, are all required for compilation.  

To run unit test from project root folder:
 Option 1: Simple run: type "make test"
 Option 2: See more details about unit test: type "./bin/UnitTest"
 
To run performance test from project root folder:
 In project root folder, type "./bin/PerfTest"

To run simulation:
 In project root folder, type "./bin/run***Simulation"
 Currently, two simulations are available: Beak and Disc.


************************
To run simulation on slurm cluster (acms-gpu is powered by slurm) 
 (1) In project root folder, cd ./scripts
 (2) sbatch *.sh, for example, sbatch discN01G02.sh means take 
     the first configuration file and then submit it to gpu02 compute node 
     so. The actual GPU device number that is going to run the program is 
     controled by slurm, and specified GPUDevice in the config file is ignored 
     if running on cluster.

Location of configuration files:
 ./resources


********************************************

To run simulation on HPCC clusters @ UCR:
   (1) Obtain all files in this repository and place these in a desired directory on the cluster.
   (2) srun -p gpu --gres=gpu:1 --mem=20g --time=12:00:00 --pty bash -l (note: this calls the interactive session to compile the code, YOU WON'T HAVE ENOUGH RESOURCE TO COMPILE WITHOUT CALLING RESOURCES FROM A GPU BY THIS LINE)
   (3) cd [NAME OF THE DIRECTORY THE FILES ARE PLACED IN] (note: of course, don't include "[" and "]" when typing the name.)
   (4) module load extra
   (5) module load GCC
   (6) module load cuda
   (7) make -j 4 (note: or just "make" would be enough)
   (7.5) It will take "a while" to compile the code if any changes are made to SceCells.cu and SceNodes.cu.
   (8) sbatch -p gpu --gres=gpu:1 --time=432:00:00 EpiScale_run.sh
   

