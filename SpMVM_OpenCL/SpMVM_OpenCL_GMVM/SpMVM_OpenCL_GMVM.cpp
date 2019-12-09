#include<compiler_config.h>

#if GMVM_SEQ || GMVM

#include<stdio.h>
#include<string>
#include<windows.h>

#include<util_misc.hpp>
#include<CL/cl.h>
#include<JC/util.hpp>
#include<IO/mmio.h>
#include<IO/convert_input.h>
#include<SEQ/GMVM.hpp>

#if PRECISION == 2
#define CL_REAL cl_double
#else
//#elif PRECISION == 1
#define CL_REAL cl_float
//#else
//#define CL_REAL cl_half // TODO?
#endif

#if GMVM_SEQ
std::vector<REAL> spmv_GMVM_sequential(struct mat_t* d_mat, const std::vector<REAL> d_x)
{
	std::vector<REAL> dst_y(d_x.size(), 0);
	//
	unsigned long nanoseconds = 0, total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		std::fill(dst_y.begin(), dst_y.end(), 0);
		nanoseconds = GMVM_sequential(d_mat, d_x, dst_y);
		printRunInfo(r + 1, nanoseconds, (d_mat->nnz));
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_mat->nnz));

	return dst_y;
}
#endif

#if GMVM
std::vector<CL_REAL> spmv_GMVM(struct mat_t* d_mat, const std::vector<CL_REAL> d_x)
{
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	//Macro
	std::string macro = "-DPRECISION=" + std::to_string(PRECISION) +
						" -DN_MATRIX=" + std::to_string(d_mat->n) +
						" -DNN_MATRIX=" + std::to_string(d_mat->n * d_mat->n) +
						" -DN_WORKGROUPS=" + std::to_string((d_mat->n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) +
						" -DWORKGROUP_SIZE=" + std::to_string(WORKGROUP_SIZE);
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + GMVM_KERNEL_FILE, context, device, macro.c_str());
	cl::Kernel kernel{ program, "spmv_gmvm" };
	//
	size_t byte_size_d_val = (unsigned long)d_mat->n * d_mat->n * sizeof(CL_REAL);
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_val_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_val };
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	cl_ulong size;
	device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
	//
	size_t local_byte_size_shx = WORKGROUP_SIZE * sizeof(CL_REAL);
	//
	queue.enqueueWriteBuffer(d_val_buffer, CL_TRUE, 0, byte_size_d_val, d_mat->val);
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	kernel.setArg(0, d_val_buffer);
	kernel.setArg(1, d_x_buffer);
	kernel.setArg(2, dst_y_buffer);
	kernel.setArg(3, cl::Local(local_byte_size_shx));
	//
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	std::cout << "!!! A work-group uses " << local_byte_size_shx << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
	for (int r = 0; r < REPEAT; r++)
	{
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
		nanoseconds =
			jc::run_and_time_kernel(kernel,
				queue,
				cl::NDRange(jc::best_fit(d_mat->n, WORKGROUP_SIZE)),
				cl::NDRange(WORKGROUP_SIZE));
		printRunInfo(r + 1, nanoseconds, (d_mat->nnz));
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_mat->nnz));

	return dst_y;
}
#endif

int main(void)
{
	// Error checking
	// TODO?

	FILE* f;
	struct coo_t coo;
	struct mat_t mat;

	std::string input_filename = (INPUT_FOLDER + (std::string)"/" + INPUT_FILE);

	if (createOutputDirectory(OUTPUT_FOLDER, GMVM_OUTPUT_FOLDER))
		exit(1);
	std::string output_file = (OUTPUT_FOLDER + (std::string)"/" + GMVM_OUTPUT_FOLDER + (std::string)"/" + OUTPUT_FILENAME + getTimeOfRun() + OUTPUT_FILEFORMAT);

	std::cout << "!!! OUTPUT IS BEING WRITTEN TO " << output_file << " !!!" << std::endl;
	std::cout << "!!! PROGRAM WILL EXIT AUTOMATICALLY AFTER PROCESSING; PRESS CTRL-C TO ABORT !!!" << std::endl;
	system("PAUSE");
	freopen_s(&f, output_file.c_str(), "w", stdout);

	std::cout << "-- LOADING INPUT FILE " << input_filename << " --" << std::endl;
	MM_To_COO(input_filename.c_str(), &coo, COO_LOG);
	IndexType n = coo.n;
	std::cout << "-- INPUT FILE LOADED --" << std::endl << std::endl;
	std::cout << "-- PRE-PROCESSING INPUT --" << std::endl;
	COO_To_MAT(&coo, &mat, MAT_LOG);
	FreeCOO(&coo);
	std::cout << "-- DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;

	std::vector<CL_REAL> x = std::vector<CL_REAL>();
	for (IndexType i = 0; i < n; i++)
		x.push_back(i);

#if GMVM_SEQ
	std::cout << std::endl << "-- STARTING GMVM SEQUENTIAL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y1 = spmv_GMVM_sequential(&mat, x);
	std::cout << std::endl << "-- FINISHED GMVM SEQUENTIAL OPERATION --" << std::endl << std::endl;
	if (GMVM_SEQ_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y1.size(); i++)
			std::cout << y1[i] << " ";
		std::cout << std::endl;
	}
#endif
#if GMVM
	std::cout << std::endl << "-- STARTING GMVM KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y2 = spmv_GMVM(&mat, x);
	std::cout << std::endl << "-- FINISHED GMVM KERNEL OPERATION --" << std::endl << std::endl;
	if (GMVM_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y2.size(); i++)
			std::cout << y2[i] << " ";
		std::cout << std::endl;
	}
#endif

	x.clear();
	FreeMAT(&mat);
#if GMVM_SEQ
	y1.clear();
#endif
#if GMVM
	y2.clear();
#endif
#if DEBUG
	system("PAUSE"); // for debugging
#endif

	return 0;
}

#endif