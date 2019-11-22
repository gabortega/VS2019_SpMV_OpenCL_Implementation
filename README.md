# VS2019_SpMV_OpenCL_Implementation
Repository containing a VS2019 solution for testing out different SpMV solutions in OpenCL with the VS CUDA Profiler

-------------------------------- Installation --------------------------------

Requirements:
   - Visual Studio 2019 (other versions not tested)
   - CUDA Toolkit (implementation is based on v10.1)
   - NVIDIA GPU compatible with OpenCL 1.2
   - pthread lib files (may not be necessary to everyone but it was for me apparently)
   
The required lib/hpp files mentioned above have been included with this repo, however, it is wise to still go through the installation process of all the listed dependencies in case I missed anything.

--------------------------------- How to use ----------------------------------

Open VS solution file "SpMVM_OpenCL.sln". 
Right-click on the solution in the Solution Explorer window and click on "Rebuild Solution".

If all went well, you should be able to find the executable of each implementation in the Release folder (i.e. ...\VS2019_SpMV_OpenCL_Implementation\x64\Release). 
However, it is much easier to just run each implementation from VS itself.

To do this, right click on the project you wish to run (by project, I mean implementation but in VS terminology these are labeled as 'projects') and select "Select as StartUp Project". 
Now, whenever you press either CTRL+F5 (Release mode) or F5 (Debug mode), VS will compile the selected project and run it.

The implementations are entirely configurable via the compiler_config.h located at ...\VS2019_SpMV_OpenCL_Implementation\SpMVM_OpenCL\config
