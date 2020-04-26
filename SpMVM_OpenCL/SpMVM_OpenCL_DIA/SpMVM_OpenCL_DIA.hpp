#include<compiler_config.h>

#if DIA_SEQ || DIA || TRANSPOSED_DIA || HDIA_SEQ || HDIA || HDIA_OLD
#ifndef OPENCL_DIA_H
#define OPENCL_DIA_H

#include<windows.h>

#include<util_misc.hpp>
#include<CL/cl.h>
#include<JC/util.hpp>
#include<SEQ/DIA.hpp>
#include<SEQ/HDIA.hpp>

#if PRECISION == 2
#define CL_REAL cl_double
#else
//#elif PRECISION == 1
#define CL_REAL cl_float
//#else
//#define CL_REAL cl_half // TODO?
#endif

#if DIA_SEQ
std::vector<REAL> spmv_DIA_sequential(struct dia_t* d_dia, const std::vector<REAL> d_x)
{
	std::vector<REAL> dst_y(d_x.size(), 0);
	//
	printHeaderInfoSEQ(d_dia->n, d_dia->nnz);
	//
	unsigned long long ioff_access = 0;
	for (IndexType i = 0; i < d_dia->n; i++)
	{
		for (IndexType j = 0; j < d_dia->ndiags; j++)
		{
			long q = i + d_dia->ioff[j];
			if (q >= 0 && q < d_dia->n)
				ioff_access += 1;
		}
	}
	//d_dia->diags + d_x + dst_y
	unsigned long long units_REAL = 2 * ioff_access + d_dia->n;
	//d_dia->ioff
	unsigned long long units_IndexType = d_dia->ndiags * d_dia->n;
	//
	unsigned long long nanoseconds = 0, total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		std::fill(dst_y.begin(), dst_y.end(), 0);
		nanoseconds = DIA_sequential(d_dia, d_x, dst_y);
		printRunInfoGPUSEQ(r + 1, nanoseconds, (d_dia->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoGPUSEQ(average_nanoseconds, (d_dia->nnz), units_REAL, units_IndexType);

	return dst_y;
}
#endif

#if DIA
std::vector<CL_REAL> spmv_DIA_param(struct dia_t* d_dia, const std::vector<CL_REAL> d_x, unsigned int workgroup_size, unsigned int max_ndiags_per_wg, unsigned int local_mem_size, unsigned int thread_count)
{
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	unsigned long long ioff_access = 0;
	for (IndexType i = 0; i < d_dia->n; i++)
	{
		for (IndexType j = 0; j < d_dia->ndiags; j++)
		{
			long q = i + d_dia->ioff[j];
			if (q >= 0 && q < d_dia->n)
				ioff_access += 1;
		}
	}
	//d_dia->diags + d_x + dst_y
	unsigned long long units_REAL = 2 * ioff_access + d_dia->n;
	//d_dia->ioff
	unsigned long long units_IndexType = (d_dia->ndiags * (jc::best_fit(d_dia->n, workgroup_size) + workgroup_size - 1) / workgroup_size);
#if !OVERRIDE_THREADS
	//
	//Instruction count
	long double instr_count = 0;
	for (IndexType i = 0; i < d_dia->ndiags; i += max_ndiags_per_wg)
		instr_count += 2 + 1 + ((double)min(d_dia->ndiags - i, max_ndiags_per_wg) / workgroup_size) * 4 + 2 + ((double)min(d_dia->ndiags - i, max_ndiags_per_wg) / workgroup_size) * 4 + 4 + 1 + min(d_dia->ndiags - i, max_ndiags_per_wg) * 4 + 2 + min(d_dia->ndiags - i, max_ndiags_per_wg) * 15 + 3;
	//
#else
	//
	//Instruction count
	long double instr_count = 0;
	for (IndexType i = 0; i < d_dia->ndiags; i += max_ndiags_per_wg)
		instr_count += 3 + 1 + ((double)min(d_dia->ndiags - i, max_ndiags_per_wg) / workgroup_size) * 4 + 2 + ((double)min(d_dia->ndiags - i, max_ndiags_per_wg) / workgroup_size) * 4 + 4 + 1 + min(d_dia->ndiags - i, max_ndiags_per_wg) * 4 + 2 + min(d_dia->ndiags - i, max_ndiags_per_wg) * 15 + 5;
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
						" -DN_MATRIX=" + std::to_string(d_dia->n) +
						" -DSTRIDE_MATRIX=" + std::to_string(d_dia->stride) +
						" -DWORKGROUP_SIZE=" + std::to_string(workgroup_size) +
						" -DUNROLL_SHARED=" + std::to_string(((workgroup_size + max_ndiags_per_wg - 1) / max_ndiags_per_wg) + 1);
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + DIA_KERNEL_FILE, context, device, macro.c_str());
#if !OVERRIDE_THREADS
	cl::Kernel kernel{ program, "spmv_dia" };
#else
	cl::Kernel kernel{ program, "occ_spmv_dia" };
#endif
	//
	printHeaderInfoGPU(d_dia->n, d_dia->nnz, deviceName, macro, instr_count);
	//
	size_t byte_size_d_ioff = d_dia->ndiags * sizeof(cl_int);
	size_t byte_size_d_diags = d_dia->stride * d_dia->ndiags * sizeof(CL_REAL);
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_ioff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ioff };
	cl::Buffer d_diags_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_diags };
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	cl_ulong size;
	device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
	//
	size_t local_byte_size_shia = max(max_ndiags_per_wg * sizeof(cl_int), local_mem_size);
	//
	queue.enqueueWriteBuffer(d_ioff_buffer, CL_TRUE, 0, byte_size_d_ioff, d_dia->ioff);
	queue.enqueueWriteBuffer(d_diags_buffer, CL_TRUE, 0, byte_size_d_diags, d_dia->diags);
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	kernel.setArg(1, d_ioff_buffer);
	kernel.setArg(2, d_diags_buffer);
	kernel.setArg(3, d_x_buffer);
	kernel.setArg(4, dst_y_buffer);
	kernel.setArg(5, cl::Local(local_byte_size_shia));
	//
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	std::cout << "!!! A work-group uses " << local_byte_size_shia << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
	for (int r = 0; r < REPEAT; r++)
	{
		nanoseconds = 0;
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
		for (IndexType i = 0; i < d_dia->ndiags; i += max_ndiags_per_wg)
		{
			kernel.setArg(0, min(d_dia->ndiags - i, max_ndiags_per_wg)); // set ndiag for this iteration
			kernel.setArg(6, i);
			kernel.setArg(7, i * d_dia->stride);
			nanoseconds +=
				jc::run_and_time_kernel(kernel,
					queue,
#if !OVERRIDE_THREADS
					cl::NDRange(jc::best_fit(d_dia->n, workgroup_size)),
#else
					cl::NDRange(jc::best_fit(thread_count, workgroup_size)),
#endif
					cl::NDRange(workgroup_size));
		}
		printRunInfoGPU(r + 1, nanoseconds, (d_dia->nnz), units_REAL, units_IndexType, instr_count);
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoGPU(average_nanoseconds, (d_dia->nnz), units_REAL, units_IndexType, instr_count);

	return dst_y;
}

std::vector<CL_REAL> spmv_DIA(struct dia_t* d_dia, const std::vector<CL_REAL> d_x)
{
	return spmv_DIA_param(d_dia, d_x, WORKGROUP_SIZE, MAX_NDIAG_PER_WG, 0, 0);
}
#endif

#if TRANSPOSED_DIA
std::vector<CL_REAL> spmv_TRANSPOSED_DIA_param(struct dia_t* d_dia, const std::vector<CL_REAL> d_x, unsigned int workgroup_size, unsigned int max_ndiags_per_wg, unsigned int local_mem_size, unsigned int thread_count)
{
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	unsigned long long ioff_access = 0;
	for (IndexType i = 0; i < d_dia->n; i++)
	{
		for (IndexType j = 0; j < d_dia->ndiags; j++)
		{
			long q = i + d_dia->ioff[j];
			if (q >= 0 && q < d_dia->n)
				ioff_access += 1;
		}
	}
	//d_dia->diags + d_x + dst_y
	unsigned long long units_REAL = 2 * ioff_access + d_dia->n;
	//d_dia->ioff
	unsigned long long units_IndexType = (d_dia->ndiags * (jc::best_fit(d_dia->n, workgroup_size) + workgroup_size - 1) / workgroup_size);
#if !OVERRIDE_THREADS
	//
	//Instruction count
	long double instr_count = 0;
	for (IndexType i = 0; i < d_dia->ndiags; i += max_ndiags_per_wg)
		instr_count += 2 + 1 + ((double)min(d_dia->ndiags - i, max_ndiags_per_wg) / workgroup_size) * 4 + 2 + ((double)min(d_dia->ndiags - i, max_ndiags_per_wg) / workgroup_size) * 4 + 4 + 1 + min(d_dia->ndiags - i, max_ndiags_per_wg) * 4 + 2 + min(d_dia->ndiags - i, max_ndiags_per_wg) * 15 + 3;
	//
#else
	//
	//Instruction count
	long double instr_count = 0;
	for (IndexType i = 0; i < d_dia->ndiags; i += max_ndiags_per_wg)
		instr_count += 3 + 1 + ((double)min(d_dia->ndiags - i, max_ndiags_per_wg) / workgroup_size) * 4 + 2 + ((double)min(d_dia->ndiags - i, max_ndiags_per_wg) / workgroup_size) * 4 + 4 + 1 + min(d_dia->ndiags - i, max_ndiags_per_wg) * 4 + 2 + min(d_dia->ndiags - i, max_ndiags_per_wg) * 15 + 5;
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
						" -DN_MATRIX=" + std::to_string(d_dia->n) +
						" -DSTRIDE_MATRIX=" + std::to_string(d_dia->stride) +
						" -DWORKGROUP_SIZE=" + std::to_string(workgroup_size) +
						" -DUNROLL_SHARED=" + std::to_string(((workgroup_size + max_ndiags_per_wg - 1) / max_ndiags_per_wg) + 1);
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + TRANSPOSED_DIA_KERNEL_FILE, context, device, macro.c_str());
#if !OVERRIDE_THREADS
	cl::Kernel kernel{ program, "spmv_transposed_dia" };
#else
	cl::Kernel kernel{ program, "occ_spmv_transposed_dia" };
#endif
	//
	printHeaderInfoGPU(d_dia->n, d_dia->nnz, deviceName, macro, instr_count);
	//
	size_t byte_size_d_ioff = d_dia->ndiags * sizeof(cl_int);
	size_t byte_size_d_diags = d_dia->stride * d_dia->ndiags * sizeof(CL_REAL);
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_ioff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ioff };
	cl::Buffer d_diags_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_diags };
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	cl_ulong size;
	device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
	//
	size_t local_byte_size_shia = max(max_ndiags_per_wg * sizeof(cl_int), local_mem_size);
	//
	queue.enqueueWriteBuffer(d_ioff_buffer, CL_TRUE, 0, byte_size_d_ioff, d_dia->ioff);
	queue.enqueueWriteBuffer(d_diags_buffer, CL_TRUE, 0, byte_size_d_diags, d_dia->diags);
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	kernel.setArg(1, d_ioff_buffer);
	kernel.setArg(2, d_diags_buffer);
	kernel.setArg(3, d_x_buffer);
	kernel.setArg(4, dst_y_buffer);
	kernel.setArg(5, cl::Local(local_byte_size_shia));
	//
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	std::cout << "!!! A work-group uses " << local_byte_size_shia << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
	for (int r = 0; r < REPEAT; r++)
	{
		nanoseconds = 0;
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
		for (IndexType i = 0; i < d_dia->ndiags; i += max_ndiags_per_wg)
		{
			kernel.setArg(0, min(d_dia->ndiags - i, max_ndiags_per_wg)); // set ndiag for this iteration
			kernel.setArg(6, i);
			kernel.setArg(7, i * d_dia->stride);
			nanoseconds +=
				jc::run_and_time_kernel(kernel,
					queue,
#if !OVERRIDE_THREADS
					cl::NDRange(jc::best_fit(d_dia->n, workgroup_size)),
#else
					cl::NDRange(jc::best_fit(thread_count, workgroup_size)),
#endif
					cl::NDRange(workgroup_size));
		}
		printRunInfoGPU(r + 1, nanoseconds, (d_dia->nnz), units_REAL, units_IndexType, instr_count);
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoGPU(average_nanoseconds, (d_dia->nnz), units_REAL, units_IndexType, instr_count);

	return dst_y;
}

std::vector<CL_REAL> spmv_TRANSPOSED_DIA(struct dia_t* d_dia, const std::vector<CL_REAL> d_x)
{
	return spmv_TRANSPOSED_DIA_param(d_dia, d_x, WORKGROUP_SIZE, MAX_NDIAG_PER_WG, 0, 0);
}
#endif

#if HDIA_SEQ
std::vector<REAL> spmv_HDIA_sequential(struct hdia_t* d_hdia, const std::vector<REAL> d_x)
{
	std::vector<REAL> dst_y(d_x.size(), 0);
	//
	printHeaderInfoSEQ(d_hdia->n, d_hdia->nnz);
	//
	unsigned long long ioff_access = 0, total_ndiags = 0;
	IndexType row_hack_id, ndiags, row_hoff;
	for (IndexType i = 0; i < d_hdia->n; i++)
	{
		row_hack_id = i / HDIA_HACKSIZE;
		total_ndiags += ndiags = d_hdia->ndiags[row_hack_id];
		row_hoff = d_hdia->hoff[row_hack_id];

		for (IndexType j = 0; j < ndiags; j++)
		{
			long q = d_hdia->ioff[row_hoff + j] + i;
			if (q >= 0 && q < d_hdia->n)
				ioff_access += 1;
		}
	}
	//d_dia->diags + d_x + dst_y
	unsigned long long units_REAL = 2 * ioff_access + d_hdia->n;
	//d_dia->ioff + d_hdia->memoff + d_hdia->ndiags + d_hdia->hoff
	unsigned long long units_IndexType = total_ndiags + d_hdia->n + d_hdia->n + d_hdia->n;
	//
	unsigned long long nanoseconds = 0, total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		std::fill(dst_y.begin(), dst_y.end(), 0);
		nanoseconds = HDIA_sequential(d_hdia, d_x, dst_y);
		printRunInfoGPUSEQ(r + 1, nanoseconds, (d_hdia->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoGPUSEQ(average_nanoseconds, (d_hdia->nnz), units_REAL, units_IndexType);

	return dst_y;
}
#endif

#if HDIA
std::vector<CL_REAL> spmv_HDIA_param(struct hdia_t* d_hdia, const std::vector<CL_REAL> d_x, unsigned int workgroup_size, unsigned int max_ndiags_per_hack, unsigned int local_mem_size, unsigned int thread_count)
{
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	unsigned long long ioff_access = 0, total_ndiags = 0;
	IndexType row_hack_id, ndiags, row_hoff;
	for (IndexType i = 0; i < d_hdia->n; i++)
	{
		row_hack_id = i / HDIA_HACKSIZE;
		total_ndiags += ndiags = d_hdia->ndiags[row_hack_id];
		row_hoff = d_hdia->hoff[row_hack_id];

		for (IndexType j = 0; j < ndiags; j++)
		{
			long q = d_hdia->ioff[row_hoff + j] + i;
			if (q >= 0 && q < d_hdia->n)
				ioff_access += 1;
		}
	}
	//d_dia->diags + d_x + dst_y
	unsigned long long units_REAL = 2 * ioff_access + d_hdia->n;
	//d_dia->ioff + d_hdia->memoff + d_hdia->ndiags + d_hdia->hoff
	unsigned long long units_IndexType = total_ndiags + d_hdia->n + d_hdia->n + d_hdia->n;
#if !OVERRIDE_THREADS
	//
	//Instruction count
	unsigned int shared_step = min(HDIA_HACKSIZE, workgroup_size);
	long double instr_count = 0;
	for (IndexType i = 0; i < *(d_hdia->ndiags + d_hdia->nhoff - 1) - 1; i += max_ndiags_per_hack)
		instr_count += 26 + 1 + ((double)min(*(d_hdia->ndiags + d_hdia->nhoff - 1) - i, max_ndiags_per_hack) / shared_step) * 4 + 2 + ((double)min(*(d_hdia->ndiags + d_hdia->nhoff - 1) - i, max_ndiags_per_hack) / shared_step) * 6 + 3 + 1 + min(*(d_hdia->ndiags + d_hdia->nhoff - 1) - i, max_ndiags_per_hack) * 4 + 2 + min(*(d_hdia->ndiags + d_hdia->nhoff - 1) - i, max_ndiags_per_hack) * 16 + 3;
	//
#else
	//
	//Instruction count
	unsigned int shared_step = min(HDIA_HACKSIZE, workgroup_size);
	long double instr_count = 0;
	for (IndexType i = 0; i < *(d_hdia->ndiags + d_hdia->nhoff - 1) - 1; i += max_ndiags_per_hack)
		instr_count += 27 + 1 + ((double)min(*(d_hdia->ndiags + d_hdia->nhoff - 1) - i, max_ndiags_per_hack) / shared_step) * 4 + 2 + ((double)min(*(d_hdia->ndiags + d_hdia->nhoff - 1) - i, max_ndiags_per_hack) / shared_step) * 6 + 3 + 1 + min(*(d_hdia->ndiags + d_hdia->nhoff - 1) - i, max_ndiags_per_hack) * 4 + 2 + min(*(d_hdia->ndiags + d_hdia->nhoff - 1) - i, max_ndiags_per_hack) * 16 + 5;
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
						" -DN_MATRIX=" + std::to_string(d_hdia->n) +
						" -DWORKGROUP_SIZE=" + std::to_string(workgroup_size) +
						" -DMAX_NDIAG=" + std::to_string(max_ndiags_per_hack) +
						" -DHACKSIZE=" + std::to_string(HDIA_HACKSIZE) +
						" -DNHOFF=" + std::to_string(d_hdia->nhoff - 1) +
						" -DUNROLL_SHARED=" + std::to_string(((workgroup_size + max_ndiags_per_hack - 1) / max_ndiags_per_hack) + 1);
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + HDIA_KERNEL_FILE, context, device, macro.c_str());
#if !OVERRIDE_THREADS
	cl::Kernel kernel{ program, "spmv_hdia" };
#else
	cl::Kernel kernel{ program, "occ_spmv_hdia" };
#endif
	//
	printHeaderInfoGPU(d_hdia->n, d_hdia->nnz, deviceName, macro, instr_count);
	//
	size_t byte_size_d_ndiags = d_hdia->nhoff * sizeof(cl_uint);
	size_t byte_size_d_ioff = *(d_hdia->hoff + d_hdia->nhoff - 1) * sizeof(cl_int);
	size_t byte_size_d_diags = *(d_hdia->memoff + d_hdia->nhoff - 1) * sizeof(CL_REAL);
	size_t byte_size_d_hoff = d_hdia->nhoff * sizeof(cl_uint);
	size_t byte_size_d_memoff = d_hdia->nhoff * sizeof(cl_uint);
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_ndiags_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ndiags };
	cl::Buffer d_ioff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ioff };
	cl::Buffer d_diags_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_diags };
	cl::Buffer d_hoff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_hoff };
	cl::Buffer d_memoff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_memoff };
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	cl_ulong size;
	device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
	//
	size_t local_byte_size_shioff = max(((workgroup_size + HDIA_HACKSIZE - 1) / HDIA_HACKSIZE) * max_ndiags_per_hack * sizeof(cl_int), local_mem_size);
	//
	queue.enqueueWriteBuffer(d_ndiags_buffer, CL_TRUE, 0, byte_size_d_ndiags, d_hdia->ndiags);
	queue.enqueueWriteBuffer(d_ioff_buffer, CL_TRUE, 0, byte_size_d_ioff, d_hdia->ioff);
	queue.enqueueWriteBuffer(d_diags_buffer, CL_TRUE, 0, byte_size_d_diags, d_hdia->diags);
	queue.enqueueWriteBuffer(d_hoff_buffer, CL_TRUE, 0, byte_size_d_hoff, d_hdia->hoff);
	queue.enqueueWriteBuffer(d_memoff_buffer, CL_TRUE, 0, byte_size_d_hoff, d_hdia->memoff);
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	kernel.setArg(0, d_ndiags_buffer);
	kernel.setArg(1, d_ioff_buffer);
	kernel.setArg(2, d_diags_buffer);
	kernel.setArg(3, d_hoff_buffer);
	kernel.setArg(4, d_memoff_buffer);
	kernel.setArg(5, d_x_buffer);
	kernel.setArg(6, dst_y_buffer);
	kernel.setArg(7, cl::Local(local_byte_size_shioff));
	//
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	std::cout << "!!! A work-group uses " << local_byte_size_shioff << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
	for (int r = 0; r < REPEAT; r++)
	{
		nanoseconds = 0;
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
		for (IndexType i = 0; i < *(d_hdia->ndiags + d_hdia->nhoff - 1) - 1; i += max_ndiags_per_hack)
		{
			kernel.setArg(8, i);
			kernel.setArg(9, i * d_hdia->stride);
			nanoseconds +=
				jc::run_and_time_kernel(kernel,
					queue,
#if !OVERRIDE_THREADS
					cl::NDRange(jc::best_fit(d_hdia->n, workgroup_size)),
#else
					cl::NDRange(jc::best_fit(thread_count, workgroup_size)),
#endif
					cl::NDRange(workgroup_size));
		}
		printRunInfoGPU(r + 1, nanoseconds, (d_hdia->nnz), units_REAL, units_IndexType, instr_count);
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoGPU(average_nanoseconds, (d_hdia->nnz), units_REAL, units_IndexType, instr_count);

	return dst_y;
}

std::vector<CL_REAL> spmv_HDIA(struct hdia_t* d_hdia, const std::vector<CL_REAL> d_x)
{
	return spmv_HDIA_param(d_hdia, d_x, WORKGROUP_SIZE, MAX_NDIAG_PER_HACK, 0, 0);
}
#endif

#if HDIA_OLD
std::vector<CL_REAL> spmv_HDIA_OLD_param(struct hdia_t* d_hdia, const std::vector<CL_REAL> d_x, unsigned int workgroup_size, unsigned int local_mem_size, unsigned int thread_count)
{
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	unsigned long long ioff_access = 0, total_ndiags = 0;
	IndexType row_hack_id, ndiags, row_hoff;
	for (IndexType i = 0; i < d_hdia->n; i++)
	{
		row_hack_id = i / HDIA_HACKSIZE;
		total_ndiags += ndiags = d_hdia->ndiags[row_hack_id];
		row_hoff = d_hdia->hoff[row_hack_id];

		for (IndexType j = 0; j < ndiags; j++)
		{
			long q = d_hdia->ioff[row_hoff + j] + i;
			if (q >= 0 && q < d_hdia->n)
				ioff_access += 1;
		}
	}
	//d_dia->diags + d_x + dst_y
	unsigned long long units_REAL = 2 * ioff_access + d_hdia->n;
	//d_dia->ioff + d_hdia->memoff + d_hdia->ndiags + d_hdia->hoff
	unsigned long long units_IndexType = total_ndiags + d_hdia->n + d_hdia->n + d_hdia->n;
#if !OVERRIDE_THREADS
	//
	//Instruction count
	long double instr_count = 11 + 1 + *(d_hdia->ndiags + d_hdia->nhoff - 1) * 4 + 2 + *(d_hdia->ndiags + d_hdia->nhoff - 1) * 15 + 2;
	//
#else
	//
	//Instruction count
	long double instr_count = 12 + 1 + *(d_hdia->ndiags + d_hdia->nhoff - 1) * 4 + 2 + *(d_hdia->ndiags + d_hdia->nhoff - 1) * 15 + 4;
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
	for (unroll_val = 1; (*(d_hdia->ndiags + d_hdia->nhoff - 1) / 2) >= unroll_val; unroll_val <<= 1);
	//
	//Macro
	std::string macro = getGlobalConstants() +
		" -DOVERRIDE_MEM=" + std::to_string(OVERRIDE_MEM) +
		" -DN_MATRIX=" + std::to_string(d_hdia->n) +
		" -DHACKSIZE=" + std::to_string(HDIA_HACKSIZE) +
		" -DUNROLL=" + std::to_string(unroll_val);
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + HDIA_OLD_KERNEL_FILE, context, device, macro.c_str());
#if !OVERRIDE_THREADS
	cl::Kernel kernel{ program, "spmv_hdia" };
#else
	cl::Kernel kernel{ program, "occ_spmv_hdia" };
#endif
	//
	printHeaderInfoGPU(d_hdia->n, d_hdia->nnz, deviceName, macro, instr_count);
	//
	size_t byte_size_d_ndiags = d_hdia->nhoff * sizeof(cl_uint);
	size_t byte_size_d_ioff = *(d_hdia->hoff + d_hdia->nhoff - 1) * sizeof(cl_int);
	size_t byte_size_d_diags = *(d_hdia->memoff + d_hdia->nhoff - 1) * sizeof(CL_REAL);
	size_t byte_size_d_hoff = d_hdia->nhoff * sizeof(cl_uint);
	size_t byte_size_d_memoff = d_hdia->nhoff * sizeof(cl_uint);
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_ndiags_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ndiags };
	cl::Buffer d_ioff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ioff };
	cl::Buffer d_diags_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_diags };
	cl::Buffer d_hoff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_hoff };
	cl::Buffer d_memoff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_memoff };
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	queue.enqueueWriteBuffer(d_ndiags_buffer, CL_TRUE, 0, byte_size_d_ndiags, d_hdia->ndiags);
	queue.enqueueWriteBuffer(d_ioff_buffer, CL_TRUE, 0, byte_size_d_ioff, d_hdia->ioff);
	queue.enqueueWriteBuffer(d_diags_buffer, CL_TRUE, 0, byte_size_d_diags, d_hdia->diags);
	queue.enqueueWriteBuffer(d_hoff_buffer, CL_TRUE, 0, byte_size_d_hoff, d_hdia->hoff);
	queue.enqueueWriteBuffer(d_memoff_buffer, CL_TRUE, 0, byte_size_d_hoff, d_hdia->memoff);
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	kernel.setArg(0, d_ndiags_buffer);
	kernel.setArg(1, d_ioff_buffer);
	kernel.setArg(2, d_diags_buffer);
	kernel.setArg(3, d_hoff_buffer);
	kernel.setArg(4, d_memoff_buffer);
	kernel.setArg(5, d_x_buffer);
	kernel.setArg(6, dst_y_buffer);
	//
#if OVERRIDE_MEM
	cl_ulong size;
	device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
	//
	size_t local_byte_size_dummy_mem = local_mem_size;
	//
	kernel.setArg(7, cl::Local(local_byte_size_dummy_mem));
	//
	std::cout << "!!! A work-group uses " << local_byte_size_dummy_mem << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
#endif
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
#if !OVERRIDE_THREADS
				cl::NDRange(jc::best_fit(d_hdia->n, workgroup_size)),
#else
				cl::NDRange(jc::best_fit(thread_count, workgroup_size)),
#endif
				cl::NDRange(workgroup_size));
		printRunInfoGPU(r + 1, nanoseconds, (d_hdia->nnz), units_REAL, units_IndexType, instr_count);
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoGPU(average_nanoseconds, (d_hdia->nnz), units_REAL, units_IndexType, instr_count);

	return dst_y;
}

std::vector<CL_REAL> spmv_HDIA_OLD(struct hdia_t* d_hdia, const std::vector<CL_REAL> d_x)
{
	return spmv_HDIA_OLD_param(d_hdia, d_x, WORKGROUP_SIZE, 0, 0);
}
#endif

#endif
#endif