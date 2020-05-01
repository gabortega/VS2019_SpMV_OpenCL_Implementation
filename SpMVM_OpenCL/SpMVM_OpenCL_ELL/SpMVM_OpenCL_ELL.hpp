#include<compiler_config.h>

#if ELL_SEQ || ELL || ELLG_SEQ || ELLG || HLL_SEQ || HLL || TRANSPOSED_ELL || TRANSPOSED_ELLG
#ifndef OPENCL_ELL_H
#define OPENCL_ELL_H

#include<windows.h>

#include<util_misc.hpp>
#include<CL/cl.h>
#include<JC/util.hpp>
#include<SEQ/ELL.hpp>
#include<SEQ/ELLG.hpp>
#include<SEQ/HLL.hpp>

#if PRECISION == 2
#define CL_REAL cl_double
#else
//#elif PRECISION == 1
#define CL_REAL cl_float
//#else
//#define CL_REAL cl_half // TODO?
#endif

#if ELL_SEQ
std::vector<REAL> spmv_ELL_sequential(struct ellg_t* d_ell, const std::vector<REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_ell->stride * *(d_ell->nell + d_ell->n); i++) d_ell->jcoeff[i]--;
	//
	std::vector<REAL> dst_y(d_x.size(), 0);
	//
	printHeaderInfoSEQ(d_ell->n, d_ell->nnz);
	//
	//d_ell->a + d_x + dst_y
	unsigned long long units_REAL = 2 * d_ell->n * d_ell->nell[d_ell->n] + d_ell->n;
	//d_ell->jcoeff
	unsigned long long units_IndexType = d_ell->n * d_ell->nell[d_ell->n];
	//
	unsigned long long nanoseconds = 0, total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		std::fill(dst_y.begin(), dst_y.end(), 0);
		nanoseconds = ELL_sequential(d_ell, d_x, dst_y);
		printRunInfoSEQ(r + 1, nanoseconds, (d_ell->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoSEQ(average_nanoseconds, (d_ell->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < d_ell->stride * *(d_ell->nell + d_ell->n); i++) d_ell->jcoeff[i]++;

	return dst_y;
}
#endif

#if ELL
std::vector<CL_REAL> spmv_ELL_param(struct ellg_t* d_ell, const std::vector<CL_REAL> d_x, unsigned int workgroup_size, unsigned int local_mem_size, unsigned int thread_count)
{
	//decrement all values
	for (IndexType i = 0; i < d_ell->stride * *(d_ell->nell + d_ell->n); i++) d_ell->jcoeff[i]--;
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//d_ell->a + d_x + dst_y
	unsigned long long units_REAL = 2 * d_ell->n * d_ell->nell[d_ell->n] + d_ell->n;
	//d_ell->jcoeff
	unsigned long long units_IndexType = d_ell->n * d_ell->nell[d_ell->n];
#if !OVERRIDE_THREADS
	//
	//Instruction count
	long double instr_count = 4 + 1 + *(d_ell->nell + d_ell->n) * 4 + 2 + *(d_ell->nell + d_ell->n) * 9 + 2;
	//
#else
	//
	//Instruction count
	long double instr_count = 5 + 1 + *(d_ell->nell + d_ell->n) * 4 + 2 + *(d_ell->nell + d_ell->n) * 9 + 4;
	//
#endif
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	//
	//Print GPU used
	std::string deviceName;
	device.getInfo<std::string>(CL_DEVICE_NAME, &deviceName);
	//
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	//Macro
	std::string macro = getGlobalConstants() +
		" -DOVERRIDE_MEM=" + std::to_string(OVERRIDE_MEM) +
		" -DNELL=" + std::to_string(*(d_ell->nell + d_ell->n)) +
		" -DN_MATRIX=" + std::to_string(d_ell->n) +
		" -DSTRIDE_MATRIX=" + std::to_string(d_ell->stride);
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + ELL_KERNEL_FILE, context, device, macro.c_str());
#if !OVERRIDE_THREADS
	cl::Kernel kernel{ program, "spmv_ell" };
#else
	cl::Kernel kernel{ program, "occ_spmv_ell" };
#endif
	//
	printHeaderInfoGPU(d_ell->n, d_ell->nnz, deviceName, macro, instr_count);
	//
	size_t byte_size_d_jcoeff = d_ell->stride * *(d_ell->nell + d_ell->n) * sizeof(cl_uint);
	size_t byte_size_d_a = d_ell->stride * *(d_ell->nell + d_ell->n) * sizeof(CL_REAL);
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_jcoeff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jcoeff };
	cl::Buffer d_a_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_a };
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	queue.enqueueWriteBuffer(d_jcoeff_buffer, CL_TRUE, 0, byte_size_d_jcoeff, d_ell->jcoeff);
	queue.enqueueWriteBuffer(d_a_buffer, CL_TRUE, 0, byte_size_d_a, d_ell->a);
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	kernel.setArg(0, d_jcoeff_buffer);
	kernel.setArg(1, d_a_buffer);
	kernel.setArg(2, d_x_buffer);
	kernel.setArg(3, dst_y_buffer);
	//
#if OVERRIDE_MEM
	cl_ulong size;
	device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
	//
	size_t local_byte_size_dummy_mem = local_mem_size;
	//
	kernel.setArg(4, cl::Local(local_byte_size_dummy_mem));
	//
	std::cout << "!!! A work-group uses " << local_byte_size_dummy_mem << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
#endif
	//
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
		nanoseconds =
			jc::run_and_time_kernel(kernel,
				queue,
#if !OVERRIDE_THREADS
				cl::NDRange(jc::best_fit(d_ell->n, workgroup_size)),
#else
				cl::NDRange(jc::best_fit(thread_count, workgroup_size)),
#endif
				cl::NDRange(workgroup_size));
		printRunInfoGPU(r + 1, nanoseconds, (d_ell->nnz), units_REAL, units_IndexType, instr_count);
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoGPU(average_nanoseconds, (d_ell->nnz), units_REAL, units_IndexType, instr_count);
	//increment all values
	for (IndexType i = 0; i < d_ell->stride * *(d_ell->nell + d_ell->n); i++) d_ell->jcoeff[i]++;

	return dst_y;
}

std::vector<CL_REAL> spmv_ELL(struct ellg_t* d_ell, const std::vector<CL_REAL> d_x)
{
	return spmv_ELL_param(d_ell, d_x, WORKGROUP_SIZE, 0, 0);
}
#endif

#if TRANSPOSED_ELL
std::vector<CL_REAL> spmv_TRANSPOSED_ELL_param(struct ellg_t* d_ell, const std::vector<CL_REAL> d_x, unsigned int workgroup_size, unsigned int local_mem_size, unsigned int thread_count)
{
	//decrement all values
	for (IndexType i = 0; i < d_ell->stride * *(d_ell->nell + d_ell->n); i++) d_ell->jcoeff[i]--;
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//d_ell->a + d_x + dst_y
	unsigned long long units_REAL = 2 * d_ell->n * d_ell->nell[d_ell->n] + d_ell->n;
	//d_ell->jcoeff
	unsigned long long units_IndexType = d_ell->n * d_ell->nell[d_ell->n];
#if !OVERRIDE_THREADS
	//
	//Instruction count
	long double instr_count = 4 + 1 + *(d_ell->nell + d_ell->n) * 4 + 2 + *(d_ell->nell + d_ell->n) * 9 + 2;
	//
#else
	//
	//Instruction count
	long double instr_count = 5 + 1 + *(d_ell->nell + d_ell->n) * 4 + 2 + *(d_ell->nell + d_ell->n) * 9 + 4;
	//
#endif
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	//
	//Print GPU used
	std::string deviceName;
	device.getInfo<std::string>(CL_DEVICE_NAME, &deviceName);
	//
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	//Macro
	std::string macro = getGlobalConstants() +
		" -DOVERRIDE_MEM=" + std::to_string(OVERRIDE_MEM) +
		" -DNELL=" + std::to_string(*(d_ell->nell + d_ell->n)) +
		" -DN_MATRIX=" + std::to_string(d_ell->n) +
		" -DSTRIDE_MATRIX=" + std::to_string(d_ell->stride);
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + TRANSPOSED_ELL_KERNEL_FILE, context, device, macro.c_str());
#if !OVERRIDE_THREADS
	cl::Kernel kernel{ program, "spmv_transposed_ell" };
#else
	cl::Kernel kernel{ program, "occ_spmv_transposed_ell" };
#endif
	//
	printHeaderInfoGPU(d_ell->n, d_ell->nnz, deviceName, macro, instr_count);
	//
	size_t byte_size_d_jcoeff = d_ell->stride * *(d_ell->nell + d_ell->n) * sizeof(cl_uint);
	size_t byte_size_d_a = d_ell->stride * *(d_ell->nell + d_ell->n) * sizeof(CL_REAL);
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_jcoeff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jcoeff };
	cl::Buffer d_a_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_a };
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	queue.enqueueWriteBuffer(d_jcoeff_buffer, CL_TRUE, 0, byte_size_d_jcoeff, d_ell->jcoeff);
	queue.enqueueWriteBuffer(d_a_buffer, CL_TRUE, 0, byte_size_d_a, d_ell->a);
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	kernel.setArg(0, d_jcoeff_buffer);
	kernel.setArg(1, d_a_buffer);
	kernel.setArg(2, d_x_buffer);
	kernel.setArg(3, dst_y_buffer);
	//
#if OVERRIDE_MEM
	cl_ulong size;
	device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
	//
	size_t local_byte_size_dummy_mem = local_mem_size;
	//
	kernel.setArg(4, cl::Local(local_byte_size_dummy_mem));
	//
	std::cout << "!!! A work-group uses " << local_byte_size_dummy_mem << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
#endif
	//
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
		nanoseconds =
			jc::run_and_time_kernel(kernel,
				queue,
#if !OVERRIDE_THREADS
				cl::NDRange(jc::best_fit(d_ell->n, workgroup_size)),
#else
				cl::NDRange(jc::best_fit(thread_count, workgroup_size)),
#endif
				cl::NDRange(workgroup_size));
		printRunInfoGPU(r + 1, nanoseconds, (d_ell->nnz), units_REAL, units_IndexType, instr_count);
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoGPU(average_nanoseconds, (d_ell->nnz), units_REAL, units_IndexType, instr_count);
	//increment all values
	for (IndexType i = 0; i < d_ell->stride * *(d_ell->nell + d_ell->n); i++) d_ell->jcoeff[i]++;

	return dst_y;
}

std::vector<CL_REAL> spmv_TRANSPOSED_ELL(struct ellg_t* d_ell, const std::vector<CL_REAL> d_x)
{
	return spmv_TRANSPOSED_ELL_param(d_ell, d_x, WORKGROUP_SIZE, 0, 0);
}
#endif

#if ELLG_SEQ
std::vector<REAL> spmv_ELLG_sequential(struct ellg_t* d_ellg, const std::vector<REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_ellg->stride * *(d_ellg->nell + d_ellg->n); i++) d_ellg->jcoeff[i]--;
	//
	std::vector<REAL> dst_y(d_x.size(), 0);
	//
	printHeaderInfoSEQ(d_ellg->n, d_ellg->nnz);
	//
	unsigned long long total_nell;
	IndexType i;
	for (i = 0, total_nell = 0; i < d_ellg->n; i++) total_nell += d_ellg->nell[i];
	//d_ellg->a + d_x + dst_y
	unsigned long long units_REAL = 2 * total_nell + d_ellg->n;
	//d_ellg->jcoeff + d_ellg->nell
	unsigned long long units_IndexType = total_nell + d_ellg->n;
	//
	unsigned long long nanoseconds = 0, total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		std::fill(dst_y.begin(), dst_y.end(), 0);
		nanoseconds = ELLG_sequential(d_ellg, d_x, dst_y);
		printRunInfoSEQ(r + 1, nanoseconds, (d_ellg->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoSEQ(average_nanoseconds, (d_ellg->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < d_ellg->stride * *(d_ellg->nell + d_ellg->n); i++) d_ellg->jcoeff[i]++;

	return dst_y;
}
#endif

#if ELLG
std::vector<CL_REAL> spmv_ELLG_param(struct ellg_t* d_ellg, const std::vector<CL_REAL> d_x, unsigned int workgroup_size, unsigned int local_mem_size, unsigned int thread_count)
{
    //decrement all values
    for (IndexType i = 0; i < d_ellg->stride * *(d_ellg->nell + d_ellg->n); i++) d_ellg->jcoeff[i]--;
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	unsigned long long total_nell;
	IndexType i;
	for (i = 0, total_nell = 0; i < d_ellg->n; i++) total_nell += d_ellg->nell[i];
	//d_ellg->a + d_x + dst_y
	unsigned long long units_REAL = 2 * total_nell + d_ellg->n;
	//d_ellg->jcoeff + d_ellg->nell
	unsigned long long units_IndexType = total_nell + d_ellg->n;
#if !OVERRIDE_THREADS
	//
	//Instruction count
	long double instr_count = 6 + 1 + *(d_ellg->nell + d_ellg->n) * 4 + 2 + *(d_ellg->nell + d_ellg->n) * 9 + 2;
	//
#else
	//
	//Instruction count
	long double instr_count = 7 + 1 + *(d_ellg->nell + d_ellg->n) * 4 + 2 + *(d_ellg->nell + d_ellg->n) * 9 + 4;
	//
#endif
    cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	//
	//Print GPU used
	std::string deviceName;
	device.getInfo<std::string>(CL_DEVICE_NAME, &deviceName);
	//
    cl::Context context{ device };
    cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	//Macro
	std::string macro = getGlobalConstants() +
						" -DOVERRIDE_MEM=" + std::to_string(OVERRIDE_MEM) +
						" -DN_MATRIX=" + std::to_string(d_ellg->n) +
						" -DSTRIDE_MATRIX=" + std::to_string(d_ellg->stride);
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + ELLG_KERNEL_FILE, context, device, macro.c_str());
#if !OVERRIDE_THREADS
	cl::Kernel kernel{ program, "spmv_ellg" };
#else
	cl::Kernel kernel{ program, "occ_spmv_ellg" };
#endif
	//
	printHeaderInfoGPU(d_ellg->n, d_ellg->nnz, deviceName, macro, instr_count);
	//
    size_t byte_size_d_nell = (d_ellg->n + 1) * sizeof(cl_uint);
    size_t byte_size_d_jcoeff = d_ellg->stride * *(d_ellg->nell + d_ellg->n) * sizeof(cl_uint);
    size_t byte_size_d_a = d_ellg->stride * *(d_ellg->nell + d_ellg->n) * sizeof(CL_REAL);
    size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
    size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
    //
    cl::Buffer d_nell_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_nell };
    cl::Buffer d_jcoeff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jcoeff };
    cl::Buffer d_a_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_a };
    cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
    cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
    //
    queue.enqueueWriteBuffer(d_nell_buffer, CL_TRUE, 0, byte_size_d_nell, d_ellg->nell);
    queue.enqueueWriteBuffer(d_jcoeff_buffer, CL_TRUE, 0, byte_size_d_jcoeff, d_ellg->jcoeff);
    queue.enqueueWriteBuffer(d_a_buffer, CL_TRUE, 0, byte_size_d_a, d_ellg->a);
    queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
    //
    kernel.setArg(0, d_nell_buffer);
    kernel.setArg(1, d_jcoeff_buffer);
    kernel.setArg(2, d_a_buffer);
    kernel.setArg(3, d_x_buffer);
    kernel.setArg(4, dst_y_buffer);
	//
#if OVERRIDE_MEM
	cl_ulong size;
	device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
	//
	size_t local_byte_size_dummy_mem = local_mem_size;
	//
	kernel.setArg(5, cl::Local(local_byte_size_dummy_mem));
	//
	std::cout << "!!! A work-group uses " << local_byte_size_dummy_mem << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
#endif
	//
    cl_ulong nanoseconds;
    cl_ulong total_nanoseconds = 0;
    //
    for (int r = 0; r < REPEAT; r++)
    {
        queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
        nanoseconds =
            jc::run_and_time_kernel(kernel,
                queue,
#if !OVERRIDE_THREADS
				cl::NDRange(jc::best_fit(d_ellg->n, workgroup_size)),
#else
				cl::NDRange(jc::best_fit(thread_count, workgroup_size)),
#endif
				cl::NDRange(workgroup_size));
		printRunInfoGPU(r + 1, nanoseconds, (d_ellg->nnz), units_REAL, units_IndexType, instr_count);
        total_nanoseconds += nanoseconds;
    }
    queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
    double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoGPU(average_nanoseconds, (d_ellg->nnz), units_REAL, units_IndexType, instr_count);
	//increment all values
	for (IndexType i = 0; i < d_ellg->stride * *(d_ellg->nell + d_ellg->n); i++) d_ellg->jcoeff[i]++;

    return dst_y;
}

std::vector<CL_REAL> spmv_ELLG(struct ellg_t* d_ellg, const std::vector<CL_REAL> d_x)
{
	return spmv_ELLG_param(d_ellg, d_x, WORKGROUP_SIZE, 0, 0);
}
#endif

#if TRANSPOSED_ELLG
std::vector<CL_REAL> spmv_TRANSPOSED_ELLG_param(struct ellg_t* d_ellg, const std::vector<CL_REAL> d_x, unsigned int workgroup_size, unsigned int local_mem_size, unsigned int thread_count)
{
	//decrement all values
	for (IndexType i = 0; i < d_ellg->stride * *(d_ellg->nell + d_ellg->n); i++) d_ellg->jcoeff[i]--;
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	unsigned long long total_nell;
	IndexType i;
	for (i = 0, total_nell = 0; i < d_ellg->n; i++) total_nell += d_ellg->nell[i];
	//d_ellg->a + d_x + dst_y
	unsigned long long units_REAL = 2 * total_nell + d_ellg->n;
	//d_ellg->jcoeff + d_ellg->nell
	unsigned long long units_IndexType = total_nell + d_ellg->n;
#if !OVERRIDE_THREADS
	//
	//Instruction count
	long double instr_count = 6 + 1 + *(d_ellg->nell + d_ellg->n) * 4 + 2 + *(d_ellg->nell + d_ellg->n) * 9 + 2;
	//
#else
	//
	//Instruction count
	long double instr_count = 7 + 1 + *(d_ellg->nell + d_ellg->n) * 4 + 2 + *(d_ellg->nell + d_ellg->n) * 9 + 4;
	//
#endif
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	//
	//Print GPU used
	std::string deviceName;
	device.getInfo<std::string>(CL_DEVICE_NAME, &deviceName);
	//
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	//Macro
	std::string macro = getGlobalConstants() +
						" -DOVERRIDE_MEM=" + std::to_string(OVERRIDE_MEM) +
						" -DN_MATRIX=" + std::to_string(d_ellg->n) +
						" -DSTRIDE_MATRIX=" + std::to_string(d_ellg->stride) +
						" -DMAX_NELL=" + std::to_string(*(d_ellg->nell + d_ellg->n));
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + TRANSPOSED_ELLG_KERNEL_FILE, context, device, macro.c_str());
#if !OVERRIDE_THREADS
	cl::Kernel kernel{ program, "spmv_transposed_ellg" };
#else
	cl::Kernel kernel{ program, "occ_spmv_transposed_ellg" };
#endif
	//
	printHeaderInfoGPU(d_ellg->n, d_ellg->nnz, deviceName, macro, instr_count);
	//
	size_t byte_size_d_nell = (d_ellg->n + 1) * sizeof(cl_uint);
	size_t byte_size_d_jcoeff = d_ellg->stride * *(d_ellg->nell + d_ellg->n) * sizeof(cl_uint);
	size_t byte_size_d_a = d_ellg->stride * *(d_ellg->nell + d_ellg->n) * sizeof(CL_REAL);
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_nell_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_nell };
	cl::Buffer d_jcoeff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jcoeff };
	cl::Buffer d_a_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_a };
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	queue.enqueueWriteBuffer(d_nell_buffer, CL_TRUE, 0, byte_size_d_nell, d_ellg->nell);
	queue.enqueueWriteBuffer(d_jcoeff_buffer, CL_TRUE, 0, byte_size_d_jcoeff, d_ellg->jcoeff);
	queue.enqueueWriteBuffer(d_a_buffer, CL_TRUE, 0, byte_size_d_a, d_ellg->a);
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	kernel.setArg(0, d_nell_buffer);
	kernel.setArg(1, d_jcoeff_buffer);
	kernel.setArg(2, d_a_buffer);
	kernel.setArg(3, d_x_buffer);
	kernel.setArg(4, dst_y_buffer);
	//
#if OVERRIDE_MEM
	cl_ulong size;
	device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
	//
	size_t local_byte_size_dummy_mem = local_mem_size;
	//
	kernel.setArg(5, cl::Local(local_byte_size_dummy_mem));
	//
	std::cout << "!!! A work-group uses " << local_byte_size_dummy_mem << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
#endif
	//
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
		nanoseconds =
			jc::run_and_time_kernel(kernel,
				queue,
#if !OVERRIDE_THREADS
				cl::NDRange(jc::best_fit(d_ellg->n, workgroup_size)),
#else
				cl::NDRange(jc::best_fit(thread_count, workgroup_size)),
#endif
				cl::NDRange(workgroup_size));
		printRunInfoGPU(r + 1, nanoseconds, (d_ellg->nnz), units_REAL, units_IndexType, instr_count);
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoGPU(average_nanoseconds, (d_ellg->nnz), units_REAL, units_IndexType, instr_count);
	//increment all values
	for (IndexType i = 0; i < d_ellg->stride * *(d_ellg->nell + d_ellg->n); i++) d_ellg->jcoeff[i]++;

	return dst_y;
}

std::vector<CL_REAL> spmv_TRANSPOSED_ELLG(struct ellg_t* d_ellg, const std::vector<CL_REAL> d_x)
{
	return spmv_TRANSPOSED_ELLG_param(d_ellg, d_x, WORKGROUP_SIZE, 0, 0);
}
#endif

#if HLL_SEQ
std::vector<REAL> spmv_HLL_sequential(struct hll_t* d_hll, const std::vector<REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_hll->total_mem; i++) d_hll->jcoeff[i]--;
	for (IndexType i = 0; i < d_hll->nhoff; i++) d_hll->hoff[i]--;
	//
	std::vector<REAL> dst_y(d_x.size(), 0);
	//
	printHeaderInfoSEQ(d_hll->n, d_hll->nnz);
	//
	unsigned long long total_nell;
	IndexType i;
	for (i = 0, total_nell = 0; i < d_hll->nhoff; i++) total_nell += d_hll->nell[i];
	//d_hll->a + d_x + dst_y
	unsigned long long units_REAL = 2 * total_nell + d_hll->n;
	//d_hll->jcoeff + d_hll->nell + d_hll->hoff
	unsigned long long units_IndexType = total_nell + d_hll->n + d_hll->n;
	//
	unsigned long long nanoseconds = 0, total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		std::fill(dst_y.begin(), dst_y.end(), 0);
		nanoseconds = HLL_sequential(d_hll, d_x, dst_y);
		printRunInfoSEQ(r + 1, nanoseconds, (d_hll->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoSEQ(average_nanoseconds, (d_hll->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < d_hll->total_mem; i++) d_hll->jcoeff[i]++;
	for (IndexType i = 0; i < d_hll->nhoff; i++) d_hll->hoff[i]++;

	return dst_y;
}
#endif

#if HLL
std::vector<CL_REAL> spmv_HLL_param(struct hll_t* d_hll, const std::vector<CL_REAL> d_x, unsigned int workgroup_size, unsigned int local_mem_size, unsigned int thread_count)
{
	//decrement all values
	for (IndexType i = 0; i < d_hll->total_mem; i++) d_hll->jcoeff[i]--;
	for (IndexType i = 0; i < d_hll->nhoff; i++) d_hll->hoff[i]--;
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	unsigned long long total_nell;
	IndexType i;
	for (i = 0, total_nell = 0; i < d_hll->nhoff; i++) total_nell += d_hll->nell[i];
	//d_hll->a + d_x + dst_y
	unsigned long long units_REAL = 2 * total_nell + d_hll->n;
	//d_hll->jcoeff + d_hll->nell + d_hll->hoff
	unsigned long long units_IndexType = total_nell + d_hll->n + d_hll->n;
#if !OVERRIDE_THREADS
	//
	//Instruction count
	long double instr_count = 12 + 1 + *(d_hll->nell + d_hll->nhoff - 1) * 4 + 2 + *(d_hll->nell + d_hll->nhoff - 1) * 10 + 2;
	//
#else
	//
	//Instruction count
	long double instr_count = 11 + 1 + *(d_hll->nell + d_hll->nhoff - 1) * 4 + 2 + *(d_hll->nell + d_hll->nhoff - 1) * 10 + 4;
	//
#endif
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	//
	//Print GPU used
	std::string deviceName;
	device.getInfo<std::string>(CL_DEVICE_NAME, &deviceName);
	//
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	IndexType unroll_val;
	for (unroll_val = 1; (*(d_hll->nell + d_hll->nhoff) / 2) >= unroll_val; unroll_val <<= 1);
	//
	//Macro
	std::string macro = getGlobalConstants() +
						" -DOVERRIDE_MEM=" + std::to_string(OVERRIDE_MEM) +
						" -DHACKSIZE=" + std::to_string(HLL_HACKSIZE) +
						" -DN_MATRIX=" + std::to_string(d_hll->n) +
						" -DUNROLL=" + std::to_string(unroll_val);
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + HLL_KERNEL_FILE, context, device, macro.c_str());
#if !OVERRIDE_THREADS
	cl::Kernel kernel{ program, "spmv_hll" };
#else
	cl::Kernel kernel{ program, "occ_spmv_hll" };
#endif
	//
	printHeaderInfoGPU(d_hll->n, d_hll->nnz, deviceName, macro, instr_count);
	//
	size_t byte_size_d_nell = d_hll->nhoff * sizeof(cl_uint);
	size_t byte_size_d_jcoeff = d_hll->total_mem * sizeof(cl_uint);
	size_t byte_size_d_hoff = d_hll->nhoff * sizeof(cl_uint);
	size_t byte_size_d_a = d_hll->total_mem * sizeof(CL_REAL);
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_nell_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_nell };
	cl::Buffer d_jcoeff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jcoeff };
	cl::Buffer d_hoff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_hoff };
	cl::Buffer d_a_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_a };
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	queue.enqueueWriteBuffer(d_nell_buffer, CL_TRUE, 0, byte_size_d_nell, d_hll->nell);
	queue.enqueueWriteBuffer(d_jcoeff_buffer, CL_TRUE, 0, byte_size_d_jcoeff, d_hll->jcoeff);
	queue.enqueueWriteBuffer(d_hoff_buffer, CL_TRUE, 0, byte_size_d_hoff, d_hll->hoff);
	queue.enqueueWriteBuffer(d_a_buffer, CL_TRUE, 0, byte_size_d_a, d_hll->a);
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	kernel.setArg(0, d_nell_buffer);
	kernel.setArg(1, d_jcoeff_buffer);
	kernel.setArg(2, d_hoff_buffer);
	kernel.setArg(3, d_a_buffer);
	kernel.setArg(4, d_x_buffer);
	kernel.setArg(5, dst_y_buffer);
	//
#if OVERRIDE_MEM
	cl_ulong size;
	device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
	//
	size_t local_byte_size_dummy_mem = local_mem_size;
	//
	kernel.setArg(6, cl::Local(local_byte_size_dummy_mem));
	//
	std::cout << "!!! A work-group uses " << local_byte_size_dummy_mem << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
#endif
	//
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
		nanoseconds =
			jc::run_and_time_kernel(kernel,
				queue,
#if !OVERRIDE_THREADS
				cl::NDRange(jc::best_fit(d_hll->n, workgroup_size)),
#else
				cl::NDRange(jc::best_fit(thread_count, workgroup_size)),
#endif
				cl::NDRange(workgroup_size));
		printRunInfoGPU(r + 1, nanoseconds, (d_hll->nnz), units_REAL, units_IndexType, instr_count);
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoGPU(average_nanoseconds, (d_hll->nnz), units_REAL, units_IndexType, instr_count);
	//increment all values
	for (IndexType i = 0; i < d_hll->total_mem; i++) d_hll->jcoeff[i]++;
	for (IndexType i = 0; i < d_hll->nhoff; i++) d_hll->hoff[i]++;

	return dst_y;
}

std::vector<CL_REAL> spmv_HLL(struct hll_t* d_hll, const std::vector<CL_REAL> d_x)
{
	return spmv_HLL_param(d_hll, d_x, WORKGROUP_SIZE, 0, 0);
}
#endif

#endif
#endif