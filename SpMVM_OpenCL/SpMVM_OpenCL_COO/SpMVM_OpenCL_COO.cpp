#include<compiler_config.h>

#if COO_SEQ || COO

#include<stdio.h>
#include<string>
#include<windows.h>
#include<vector>

#include<util_misc.hpp>
#include<CL/cl.h>
#include<JC/util.hpp>
#include<IO/mmio.h>
#include<IO/convert_input.h>
#include<SEQ/COO.hpp>

#if PRECISION == 2
#define CL_REAL cl_double
#else
//#elif PRECISION == 1
#define CL_REAL cl_float
//#else
//#define CL_REAL cl_half // TODO?
#endif

#if COO_SEQ
std::vector<REAL> spmv_COO_sequential(struct coo_t* d_coo, const std::vector<REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_coo->nnz; i++) d_coo->ir[i]--, d_coo->jc[i]--;
	//
	std::vector<REAL> dst_y(d_x.size(), 0);
	//
	unsigned long nanoseconds = 0, total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		std::fill(dst_y.begin(), dst_y.end(), 0);
		nanoseconds = COO_sequential(d_coo, d_x, dst_y);
		printRunInfo(r + 1, nanoseconds, (d_coo->nnz));
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_coo->nnz));
	//increment all values
	for (IndexType i = 0; i < d_coo->nnz; i++) d_coo->ir[i]++, d_coo->jc[i]++;

	return dst_y;
}
#endif

#if COO
std::vector<CL_REAL> spmv_COO_serial(struct coo_t* d_coo, const std::vector<CL_REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_coo->nnz; i++) d_coo->ir[i]--, d_coo->jc[i]--;
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	//Macro
	std::string macro = "-DPRECISION=" + std::to_string(PRECISION) +
						" -DNNZ_MATRIX=" + std::to_string(d_coo->nnz);
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + COO_KERNEL_FILE, context, device, macro.c_str());
	cl::Kernel kernel{ program, "spmv_coo_serial" };
	//
	size_t byte_size_d_ir = d_coo->nnz * sizeof(cl_uint);
	size_t byte_size_d_jc = d_coo->nnz * sizeof(cl_uint);
	size_t byte_size_d_val = d_coo->nnz * sizeof(CL_REAL);
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_ir_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ir };
	cl::Buffer d_jc_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jc };
	cl::Buffer d_val_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_val };
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	queue.enqueueWriteBuffer(d_ir_buffer, CL_TRUE, 0, byte_size_d_ir, d_coo->ir);
	queue.enqueueWriteBuffer(d_jc_buffer, CL_TRUE, 0, byte_size_d_jc, d_coo->jc);
	queue.enqueueWriteBuffer(d_val_buffer, CL_TRUE, 0, byte_size_d_val, d_coo->val);
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	kernel.setArg(0, d_ir_buffer);
	kernel.setArg(1, d_jc_buffer);
	kernel.setArg(2, d_val_buffer);
	kernel.setArg(3, d_x_buffer);
	kernel.setArg(4, dst_y_buffer);
	//
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_d_x, dst_y.data());
		nanoseconds =
			jc::run_and_time_kernel(kernel,
				queue,
				cl::NDRange(1),
				cl::NDRange(1));
		printRunInfo(r + 1, nanoseconds, (d_coo->nnz));
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_coo->nnz));
	//increment all values
	for (IndexType i = 0; i < d_coo->nnz; i++) d_coo->ir[i]++, d_coo->jc[i]++;

	return dst_y;
}
#endif

int main(void)
{
	// Error checking
	// TODO?

	FILE* f;
	struct coo_t coo;

	std::string input_filename = (INPUT_FOLDER + (std::string)"/" + INPUT_FILE);

	if (createOutputDirectory(OUTPUT_FOLDER, COO_OUTPUT_FOLDER))
		exit(1);
	std::string output_file = (OUTPUT_FOLDER + (std::string)"/" + COO_OUTPUT_FOLDER + (std::string)"/" + OUTPUT_FILENAME + getTimeOfRun() + OUTPUT_FILEFORMAT);

	std::cout << "!!! OUTPUT IS BEING WRITTEN TO " << output_file << " !!!" << std::endl;
	std::cout << "!!! PROGRAM WILL EXIT AUTOMATICALLY AFTER PROCESSING; PRESS CTRL-C TO ABORT !!!" << std::endl;
	system("PAUSE");
	freopen_s(&f, output_file.c_str(), "w", stdout);

	std::cout << "-- LOADING INPUT FILE " << input_filename << " --" << std::endl;
	MM_To_COO(input_filename.c_str(), &coo, COO_LOG);
	std::cout << "-- INPUT FILE LOADED --" << std::endl << std::endl;

	std::vector<CL_REAL> x = std::vector<CL_REAL>();
	for (IndexType i = 0; i < coo.n; i++)
		x.push_back(i);

#if COO_SEQ
	std::cout << std::endl << "-- STARTING COO SEQUENTIAL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y1 = spmv_COO_sequential(&coo, x);
	std::cout << std::endl << "-- FINISHED COO SEQUENTIAL OPERATION --" << std::endl << std::endl;
	if (COO_SEQ_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y1.size(); i++)
			std::cout << y1[i] << " ";
		std::cout << std::endl;
	}
#endif
#if COO
	std::cout << std::endl << "-- STARTING COO (SERIAL) KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y2 = spmv_COO_serial(&coo, x);
	std::cout << std::endl << "-- FINISHED COO (SERIAL) KERNEL OPERATION --" << std::endl << std::endl;
	if (COO_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y2.size(); i++)
			std::cout << y2[i] << " ";
		std::cout << std::endl;
	}
#endif

	x.clear();
	FreeCOO(&coo);
#if COO_SEQ
	y1.clear();
#endif
#if COO
	y2.clear();
#endif
#if DEBUG
	system("PAUSE"); // for debugging
#endif

	return 0;
}

#endif