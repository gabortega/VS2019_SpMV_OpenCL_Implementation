#include<compiler_config.h>

#if ELL_SEQ || ELL || ELLG_SEQ || ELLG || HLL_SEQ || HLL
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
		printRunInfo(r + 1, nanoseconds, (d_ell->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_ell->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < d_ell->stride * *(d_ell->nell + d_ell->n); i++) d_ell->jcoeff[i]++;

	return dst_y;
}
#endif

#if ELL
std::vector<CL_REAL> spmv_ELL(struct ellg_t* d_ell, const std::vector<CL_REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_ell->stride * *(d_ell->nell + d_ell->n); i++) d_ell->jcoeff[i]--;
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//d_ell->a + d_x + dst_y
	unsigned long long units_REAL = 2 * d_ell->n * d_ell->nell[d_ell->n] + d_ell->n;
	//d_ell->jcoeff
	unsigned long long units_IndexType = d_ell->n * d_ell->nell[d_ell->n];
	//
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
	std::string macro = "-DPRECISION=" + std::to_string(PRECISION) +
		" -DNELL=" + std::to_string(*(d_ell->nell + d_ell->n)) +
		" -DN_MATRIX=" + std::to_string(d_ell->n) +
		" -DSTRIDE_MATRIX=" + std::to_string(d_ell->stride);
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + ELL_KERNEL_FILE, context, device, macro.c_str());
	cl::Kernel kernel{ program, "spmv_ell" };
	//
	printHeaderInfoGPU(d_ell->n, d_ell->nnz, deviceName, macro);
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
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
		nanoseconds =
			jc::run_and_time_kernel(kernel,
				queue,
				cl::NDRange(jc::best_fit(d_ell->n, WORKGROUP_SIZE)),
				cl::NDRange(WORKGROUP_SIZE));
		printRunInfo(r + 1, nanoseconds, (d_ell->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_ell->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < d_ell->stride * *(d_ell->nell + d_ell->n); i++) d_ell->jcoeff[i]++;

	return dst_y;
}
#endif

#if TRANSPOSED_ELL
std::vector<CL_REAL> spmv_TRANSPOSED_ELL(struct ellg_t* d_ell, const std::vector<CL_REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_ell->stride * *(d_ell->nell + d_ell->n); i++) d_ell->jcoeff[i]--;
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//d_ell->a + d_x + dst_y
	unsigned long long units_REAL = 2 * d_ell->n * d_ell->nell[d_ell->n] + d_ell->n;
	//d_ell->jcoeff
	unsigned long long units_IndexType = d_ell->n * d_ell->nell[d_ell->n];
	//
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
	std::string macro = "-DPRECISION=" + std::to_string(PRECISION) +
		" -DNELL=" + std::to_string(*(d_ell->nell + d_ell->n)) +
		" -DN_MATRIX=" + std::to_string(d_ell->n) +
		" -DSTRIDE_MATRIX=" + std::to_string(d_ell->stride);
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + TRANSPOSED_ELL_KERNEL_FILE, context, device, macro.c_str());
	cl::Kernel kernel{ program, "spmv_transposed_ell" };
	//
	printHeaderInfoGPU(d_ell->n, d_ell->nnz, deviceName, macro);
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
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
		nanoseconds =
			jc::run_and_time_kernel(kernel,
				queue,
				cl::NDRange(jc::best_fit(d_ell->n, WORKGROUP_SIZE)),
				cl::NDRange(WORKGROUP_SIZE));
		printRunInfo(r + 1, nanoseconds, (d_ell->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_ell->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < d_ell->stride * *(d_ell->nell + d_ell->n); i++) d_ell->jcoeff[i]++;

	return dst_y;
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
		printRunInfo(r + 1, nanoseconds, (d_ellg->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_ellg->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < d_ellg->stride * *(d_ellg->nell + d_ellg->n); i++) d_ellg->jcoeff[i]++;

	return dst_y;
}
#endif

#if ELLG
std::vector<CL_REAL> spmv_ELLG(struct ellg_t* d_ellg, const std::vector<CL_REAL> d_x)
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
    //
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
	std::string macro = "-DPRECISION=" + std::to_string(PRECISION) +
						" -DN_MATRIX=" + std::to_string(d_ellg->n) +
						" -DSTRIDE_MATRIX=" + std::to_string(d_ellg->stride);
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + ELLG_KERNEL_FILE, context, device, macro.c_str());
	cl::Kernel kernel{ program, "spmv_ellg" };
	//
	printHeaderInfoGPU(d_ellg->n, d_ellg->nnz, deviceName, macro);
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
    cl_ulong nanoseconds;
    cl_ulong total_nanoseconds = 0;
    //
    for (int r = 0; r < REPEAT; r++)
    {
        queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
        nanoseconds =
            jc::run_and_time_kernel(kernel,
                queue,
				cl::NDRange(jc::best_fit(d_ellg->n, WORKGROUP_SIZE)),
				cl::NDRange(WORKGROUP_SIZE));
		printRunInfo(r + 1, nanoseconds, (d_ellg->nnz), units_REAL, units_IndexType);
        total_nanoseconds += nanoseconds;
    }
    queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
    double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_ellg->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < d_ellg->stride * *(d_ellg->nell + d_ellg->n); i++) d_ellg->jcoeff[i]++;

    return dst_y;
}
#endif

#if TRANSPOSED_ELLG
std::vector<CL_REAL> spmv_TRANSPOSED_ELLG(struct ellg_t* d_ellg, const std::vector<CL_REAL> d_x)
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
	//
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
	std::string macro = "-DPRECISION=" + std::to_string(PRECISION) +
		" -DN_MATRIX=" + std::to_string(d_ellg->n) +
		" -DSTRIDE_MATRIX=" + std::to_string(d_ellg->stride) +
		" -DMAX_NELL=" + std::to_string(*(d_ellg->nell + d_ellg->n));
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + TRANSPOSED_ELLG_KERNEL_FILE, context, device, macro.c_str());
	cl::Kernel kernel{ program, "spmv_transposed_ellg" };
	//
	printHeaderInfoGPU(d_ellg->n, d_ellg->nnz, deviceName, macro);
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
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
		nanoseconds =
			jc::run_and_time_kernel(kernel,
				queue,
				cl::NDRange(jc::best_fit(d_ellg->n, WORKGROUP_SIZE)),
				cl::NDRange(WORKGROUP_SIZE));
		printRunInfo(r + 1, nanoseconds, (d_ellg->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_ellg->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < d_ellg->stride * *(d_ellg->nell + d_ellg->n); i++) d_ellg->jcoeff[i]++;

	return dst_y;
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
		printRunInfo(r + 1, nanoseconds, (d_hll->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_hll->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < d_hll->total_mem; i++) d_hll->jcoeff[i]++;
	for (IndexType i = 0; i < d_hll->nhoff; i++) d_hll->hoff[i]++;

	return dst_y;
}
#endif

#if HLL
std::vector<CL_REAL> spmv_HLL(struct hll_t* d_hll, const std::vector<CL_REAL> d_x)
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
	//
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
	std::string macro = "-DPRECISION=" + std::to_string(PRECISION) + 
						" -DHACKSIZE=" + std::to_string(HLL_HACKSIZE) +
						" -DN_MATRIX=" + std::to_string(d_hll->n) +
						" -DUNROLL=" + std::to_string(unroll_val);
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + HLL_KERNEL_FILE, context, device, macro.c_str());
	cl::Kernel kernel{ program, "spmv_hll" };
	//
	printHeaderInfoGPU(d_hll->n, d_hll->nnz, deviceName, macro);
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
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
		nanoseconds =
			jc::run_and_time_kernel(kernel,
				queue,
				cl::NDRange(jc::best_fit(d_hll->n, WORKGROUP_SIZE)),
				cl::NDRange(WORKGROUP_SIZE));
		printRunInfo(r + 1, nanoseconds, (d_hll->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_hll->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < d_hll->total_mem; i++) d_hll->jcoeff[i]++;
	for (IndexType i = 0; i < d_hll->nhoff; i++) d_hll->hoff[i]++;

	return dst_y;
}
#endif

#endif
#endif