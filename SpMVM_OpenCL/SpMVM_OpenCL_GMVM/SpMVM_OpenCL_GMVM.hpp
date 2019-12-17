#include<compiler_config.h>

#if GMVM_SEQ || GMVM
#ifndef OPENCL_GMVM_H
#define OPENCL_GMVM_H

#include<stdio.h>
#include<string>
#include<windows.h>

#include<util_misc.hpp>
#include<CL/cl.h>
#include<JC/util.hpp>
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
	//d_mat->val + d_x + dst_y
	unsigned long long units_REAL = 3 * d_mat->n * d_mat->n;
	//
	unsigned long long nanoseconds = 0, total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		std::fill(dst_y.begin(), dst_y.end(), 0);
		nanoseconds = GMVM_sequential(d_mat, d_x, dst_y);
		printRunInfo(r + 1, nanoseconds, (d_mat->nnz), units_REAL, 0);
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_mat->nnz), units_REAL, 0);

	return dst_y;
}
#endif

#if GMVM
std::vector<CL_REAL> spmv_GMVM(struct mat_t* d_mat, const std::vector<CL_REAL> d_x)
{
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//d_mat->val + d_x + dst_y
	unsigned long long units_REAL = d_mat->n * d_mat->n + (((d_mat->n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) * d_mat->n) + d_mat->n;
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
	std::cout << "Kernel macros: " << macro << std::endl << std::endl;
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
		printRunInfo(r + 1, nanoseconds, (d_mat->nnz), units_REAL, 0);
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_mat->nnz), units_REAL, 0);

	return dst_y;
}
#endif

#endif
#endif