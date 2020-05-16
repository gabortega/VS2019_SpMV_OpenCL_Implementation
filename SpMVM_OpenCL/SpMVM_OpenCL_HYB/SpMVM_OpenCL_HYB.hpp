#include<compiler_config.h>

#if HYB_ELL_SEQ || HYB_ELLG_SEQ || HYB_HLL_SEQ || HYB_ELL || HYB_ELLG || HYB_HLL
#ifndef OPENCL_HYB_H
#define OPENCL_HYB_H

#include<stdio.h>
#include<string>
#include<windows.h>

#include<util_misc.hpp>
#include<CL/cl.h>
#include<JC/util.hpp>
#include<SEQ/ELL.hpp>
#include<SEQ/ELLG.hpp>
#include<SEQ/HLL.hpp>
#include<SEQ/CSR.hpp>

#if PRECISION == 2
#define CL_REAL cl_double
#else
//#elif PRECISION == 1
#define CL_REAL cl_float
//#else
//#define CL_REAL cl_half // TODO?
#endif

#if HYB_ELL_SEQ
std::vector<REAL> spmv_HYB_ELL_sequential(struct hybellg_t* d_hyb, const std::vector<REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n); i++) d_hyb->ellg.jcoeff[i]--;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]--;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]--;
	//
	std::vector<REAL> dst_y(d_x.size(), 0);
	//
	printHeaderInfoSEQ(d_hyb->n, d_hyb->nnz);
	//
	unsigned long long units_REAL = 0, units_IndexType = 0;
	if (d_hyb->ellg.nnz > 0)
	{
		//d_hyb->ellg.a + d_x + dst_y
		units_REAL += 2 * d_hyb->ellg.n * d_hyb->ellg.nell[d_hyb->ellg.n] + d_hyb->ellg.n;
		//d_hyb->ellg.jcoeff
		units_IndexType += d_hyb->ellg.n * d_hyb->ellg.nell[d_hyb->ellg.n];
	}
	if (d_hyb->csr.nnz > 0)
	{
		//d_hyb->csr.a + d_x + dst_y
		units_REAL += d_hyb->csr.nnz + d_hyb->csr.nnz + d_hyb->csr.nnz;
		//d_hyb->csr.ia + d_hyb->csr.ja
		units_IndexType += d_hyb->csr.n + d_hyb->csr.nnz;
	}
	//
	unsigned long long nanoseconds = 0, total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		std::fill(dst_y.begin(), dst_y.end(), 0);
		nanoseconds = 0;
		if (d_hyb->ellg.nnz > 0)
		{
			nanoseconds += ELL_sequential(&(d_hyb->ellg), d_x, dst_y);
		}
		if (d_hyb->csr.nnz > 0)
		{
			nanoseconds += CSR_sequential(&(d_hyb->csr), d_x, dst_y);
		}
		printRunInfoSEQ(r + 1, nanoseconds, (d_hyb->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoSEQ(average_nanoseconds, (d_hyb->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n); i++) d_hyb->ellg.jcoeff[i]++;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]++;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]++;

	return dst_y;
}
#endif

#if HYB_ELL
std::vector<CL_REAL> spmv_HYB_ELL_param(struct hybellg_t* d_hyb, const std::vector<CL_REAL> d_x, unsigned int csr_workgroup_size, unsigned int ellg_workgroup_size, unsigned int csr_local_mem_size, unsigned int ellg_local_mem_size)
{	
	//decrement all values
	for (IndexType i = 0; i < d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n); i++) d_hyb->ellg.jcoeff[i]--;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]--;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]--;
	//
	IndexType i, row_len = 0, coop = 1, repeat = 1, nworkgroups;
	if (d_hyb->csr.nnz > 0)
	{
		for (i = 0; i < d_hyb->csr.n; i++) row_len += d_hyb->csr.ia[i + 1] - d_hyb->csr.ia[i];
		row_len = sqrt(row_len / d_hyb->csr.n);
		for (coop = 1; coop < 32 && row_len >= coop; coop <<= 1);
		nworkgroups = 1 + (d_hyb->csr.n * coop - 1) / (repeat * csr_workgroup_size);
		if (nworkgroups > 1500)
			for (repeat = 1; (1 + (d_hyb->csr.n * coop - 1) / ((repeat + 1) * csr_workgroup_size)) > 1500; repeat++);
		nworkgroups = 1 + (d_hyb->csr.n * coop - 1) / (repeat * csr_workgroup_size);
	}
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	unsigned long long units_REAL = 0, units_IndexType = 0;
	if (d_hyb->ellg.nnz > 0)
	{
		//d_hyb->ellg.a + d_x + dst_y
		units_REAL += 2 * d_hyb->ellg.n * d_hyb->ellg.nell[d_hyb->ellg.n] + d_hyb->ellg.n;
		//d_hyb->ellg.jcoeff
		units_IndexType += d_hyb->ellg.n * d_hyb->ellg.nell[d_hyb->ellg.n];
	}
	if (d_hyb->csr.nnz > 0)
	{
		//d_hyb->csr.a + d_x + dst_y
		units_REAL += d_hyb->csr.nnz + d_hyb->csr.nnz + d_hyb->csr.n;
		//d_hyb->csr.ia + d_hyb->csr.ja
		units_IndexType += d_hyb->csr.n + d_hyb->csr.nnz;
	}
	//
	//Instruction count
	long double instr_count_ell = 0;
	if (d_hyb->ellg.nnz > 0)
	{
		instr_count_ell = 4 + 1 + *(d_hyb->ellg.nell + d_hyb->n) * 4 + 2 + *(d_hyb->ellg.nell + d_hyb->n) * 9 + 2;
		instr_count_ell *= d_hyb->ellg.n;
	}
	//
	//Instruction count
	long double instr_count_csr = 0;
	if (d_hyb->csr.nnz > 0)
	{
		instr_count_csr = 6 + 1 + repeat * 4 + 2 + repeat * (5 + 1 + ((double)row_len / coop) * 12 + 5 + ((double)row_len / coop) * 8 + 2 + 1 + (max(1, log2(coop / 2)) * 4) + 2 + max(1, log2(coop / 2)) * 7 + 9);
		instr_count_csr *= d_hyb->csr.n;
	}
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
	//Macros
	std::string csr_macro = getGlobalConstants() +
							" -DCSR_REPEAT=" + std::to_string(repeat) +
							" -DCSR_COOP=" + std::to_string(coop) +
							" -DUNROLL_SHARED=" + std::to_string(coop / 4) +
							" -DN_MATRIX=" + std::to_string(d_hyb->csr.n) +
							" -DWORKGROUP_SIZE=" + std::to_string(csr_workgroup_size);
	std::string ell_macro = getGlobalConstants() +
							" -DOVERRIDE_MEM=" + std::to_string(OVERRIDE_MEM) +
							" -DNELL=" + std::to_string(*(d_hyb->ellg.nell + d_hyb->ellg.n)) +
							" -DN_MATRIX=" + std::to_string(d_hyb->ellg.n) +
							" -DSTRIDE_MATRIX=" + std::to_string(d_hyb->ellg.stride);
	//
	cl::Program program_csr =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + CSR_KERNEL_FILE, context, device, csr_macro.c_str());
	cl::Kernel kernel_csr{ program_csr, "spmv_csr" };
	//
	cl::Program program_ell =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + ELL_KERNEL_FILE, context, device, ell_macro.c_str());
	cl::Kernel kernel_ell{ program_ell, "spmv_ell" };
	//
	printHeaderInfoGPU_HYB(d_hyb->n, d_hyb->nnz, deviceName, "\nCSR kernel macros: " + csr_macro + "\nELL kernel macros: " + ell_macro, instr_count_csr, instr_count_ell);
	//
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	// ell related
	size_t byte_size_d_jcoeff;
	size_t byte_size_d_a;
	//
	cl::Buffer d_jcoeff_buffer;
	cl::Buffer d_a_buffer;
	//
	if (d_hyb->ellg.nnz > 0)
	{
		byte_size_d_jcoeff = d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n) * sizeof(cl_uint);
		byte_size_d_a = d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n) * sizeof(CL_REAL);
		//
		d_jcoeff_buffer = cl::Buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jcoeff };
		d_a_buffer = cl::Buffer{ context, CL_MEM_READ_ONLY, byte_size_d_a };
		//
		queue.enqueueWriteBuffer(d_jcoeff_buffer, CL_TRUE, 0, byte_size_d_jcoeff, d_hyb->ellg.jcoeff);
		queue.enqueueWriteBuffer(d_a_buffer, CL_TRUE, 0, byte_size_d_a, d_hyb->ellg.a);
		//
		kernel_ell.setArg(0, d_jcoeff_buffer);
		kernel_ell.setArg(1, d_a_buffer);
		kernel_ell.setArg(2, d_x_buffer);
		kernel_ell.setArg(3, dst_y_buffer);
		//
#if OVERRIDE_MEM
		cl_ulong size;
		device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
		//
		size_t local_byte_size_dummy_mem = ellg_local_mem_size;
		//
		kernel_ell.setArg(4, cl::Local(local_byte_size_dummy_mem));
		//
		std::cout << "!!! A work-group uses " << local_byte_size_dummy_mem << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
#endif
	}
	//
	// csr related
	size_t byte_size_d_ia;
	size_t byte_size_d_ja;
	size_t byte_size_d_val;
	//
	cl::Buffer d_ia_buffer;
	cl::Buffer d_ja_buffer;
	cl::Buffer d_val_buffer;
	//
	size_t local_byte_size_shdata;
	//
	if (d_hyb->csr.nnz > 0)
	{
		byte_size_d_ia = (d_hyb->csr.n + 1) * sizeof(cl_uint);
		byte_size_d_ja = d_hyb->csr.nnz * sizeof(cl_uint);
		byte_size_d_val = d_hyb->csr.nnz * sizeof(CL_REAL);
		//
		d_ia_buffer = cl::Buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ia };
		d_ja_buffer = cl::Buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ja };
		d_val_buffer = cl::Buffer{ context, CL_MEM_READ_ONLY, byte_size_d_val };
		//
		cl_ulong size;
		device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
		//
		local_byte_size_shdata = max(csr_workgroup_size * sizeof(CL_REAL), csr_local_mem_size);
		//
		queue.enqueueWriteBuffer(d_ia_buffer, CL_TRUE, 0, byte_size_d_ia, d_hyb->csr.ia);
		queue.enqueueWriteBuffer(d_ja_buffer, CL_TRUE, 0, byte_size_d_ja, d_hyb->csr.ja);
		queue.enqueueWriteBuffer(d_val_buffer, CL_TRUE, 0, byte_size_d_val, d_hyb->csr.a);
		//
		kernel_csr.setArg(0, d_ia_buffer);
		kernel_csr.setArg(1, d_ja_buffer);
		kernel_csr.setArg(2, d_val_buffer);
		kernel_csr.setArg(3, d_x_buffer);
		kernel_csr.setArg(4, dst_y_buffer);
		kernel_csr.setArg(5, cl::Local(local_byte_size_shdata));
		//
		std::cout << "!!! CSR kernel: repeat = " << repeat << ", coop = " << coop << ", nworkgroups = " << nworkgroups << " !!!" << std::endl << std::endl;
		std::cout << "!!! A work-group uses " << local_byte_size_shdata << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
	}
	//
	cl_ulong nanoseconds_ell;
	cl_ulong nanoseconds_csr;
	cl_ulong total_nanoseconds_ell = 0;
	cl_ulong total_nanoseconds_csr = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		nanoseconds_ell = 0;
		nanoseconds_csr = 0;
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
		if (d_hyb->ellg.nnz > 0)
		{
			nanoseconds_ell +=
				jc::run_and_time_kernel(kernel_ell,
					queue,
					cl::NDRange(jc::best_fit(d_hyb->ellg.n, ellg_workgroup_size)),
					cl::NDRange(ellg_workgroup_size));
		}
		if (d_hyb->csr.nnz > 0)
		{
			nanoseconds_csr +=
				jc::run_and_time_kernel(kernel_csr,
					queue,
					cl::NDRange(nworkgroups * csr_workgroup_size),
					cl::NDRange(csr_workgroup_size));
		}
		printRunInfoGPU_HYB(r + 1, nanoseconds_csr, nanoseconds_ell, (d_hyb->csr.nnz), (d_hyb->ellg.nnz), coop, units_REAL, units_IndexType, instr_count_csr, instr_count_ell);
		total_nanoseconds_ell += nanoseconds_ell;
		total_nanoseconds_csr += nanoseconds_csr;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds_ell = total_nanoseconds_ell / (double)REPEAT;
	double average_nanoseconds_csr = total_nanoseconds_csr / (double)REPEAT;
	printAverageRunInfoGPU_HYB(average_nanoseconds_csr, average_nanoseconds_ell, (d_hyb->csr.nnz), (d_hyb->ellg.nnz), coop, units_REAL, units_IndexType, instr_count_csr, instr_count_ell);
	//increment all values
	for (IndexType i = 0; i < d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n); i++) d_hyb->ellg.jcoeff[i]++;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]++;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]++;

	return dst_y;
}

std::vector<CL_REAL> spmv_HYB_ELL(struct hybellg_t* d_hyb, const std::vector<CL_REAL> d_x)
{
	return spmv_HYB_ELL_param(d_hyb, d_x, CSR_WORKGROUP_SIZE, WORKGROUP_SIZE, 0, 0);
}
#endif

#if HYB_ELLG_SEQ
std::vector<REAL> spmv_HYB_ELLG_sequential(struct hybellg_t* d_hyb, const std::vector<REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n); i++) d_hyb->ellg.jcoeff[i]--;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]--;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]--;
	//
	std::vector<REAL> dst_y(d_x.size(), 0);
	//
	printHeaderInfoSEQ(d_hyb->n, d_hyb->nnz);
	//
	unsigned long long units_REAL = 0, units_IndexType = 0;
	if (d_hyb->ellg.nnz > 0)
	{
		unsigned long long total_nell;
		IndexType i;
		for (i = 0, total_nell = 0; i < d_hyb->ellg.n; i++) total_nell += d_hyb->ellg.nell[i];
		//d_hyb->ellg.a + d_x + dst_y
		units_REAL += 2 * total_nell + d_hyb->ellg.n;
		//d_hyb->ellg.jcoeff + d_hyb->ellg.nell
		units_IndexType += total_nell + d_hyb->ellg.n;
	}
	if (d_hyb->csr.nnz > 0)
	{
		//d_hyb->csr.a + d_x + dst_y
		units_REAL += d_hyb->csr.nnz + d_hyb->csr.nnz + d_hyb->csr.nnz;
		//d_hyb->csr.ia + d_hyb->csr.ja
		units_IndexType += d_hyb->csr.n + d_hyb->csr.nnz;
	}
	//
	unsigned long long nanoseconds = 0, total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		std::fill(dst_y.begin(), dst_y.end(), 0);
		nanoseconds = 0;
		if (d_hyb->ellg.nnz > 0)
		{
			nanoseconds += ELLG_sequential(&(d_hyb->ellg), d_x, dst_y);
		}
		if (d_hyb->csr.nnz > 0)
		{
			nanoseconds += CSR_sequential(&(d_hyb->csr), d_x, dst_y);
		}
		printRunInfoSEQ(r + 1, nanoseconds, (d_hyb->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoSEQ(average_nanoseconds, (d_hyb->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n); i++) d_hyb->ellg.jcoeff[i]++;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]++;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]++;

	return dst_y;
}
#endif

#if HYB_ELLG
std::vector<CL_REAL> spmv_HYB_ELLG_param(struct hybellg_t* d_hyb, const std::vector<CL_REAL> d_x, unsigned int csr_workgroup_size, unsigned int ellg_workgroup_size, unsigned int csr_local_mem_size, unsigned int ellg_local_mem_size)
{
	//decrement all values
	for (IndexType i = 0; i < d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n); i++) d_hyb->ellg.jcoeff[i]--;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]--;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]--;
	//
	IndexType i, row_len = 0, coop = 1, repeat = 1, nworkgroups;
	if (d_hyb->csr.nnz > 0)
	{
		for (i = 0; i < d_hyb->csr.n; i++) row_len += d_hyb->csr.ia[i + 1] - d_hyb->csr.ia[i];
		row_len = sqrt(row_len / d_hyb->csr.n);
		for (coop = 1; coop < 32 && row_len >= coop; coop <<= 1);
		nworkgroups = 1 + (d_hyb->csr.n * coop - 1) / (repeat * csr_workgroup_size);
		if (nworkgroups > 1500)
			for (repeat = 1; (1 + (d_hyb->csr.n * coop - 1) / ((repeat + 1) * csr_workgroup_size)) > 1500; repeat++);
		nworkgroups = 1 + (d_hyb->csr.n * coop - 1) / (repeat * csr_workgroup_size);
	}
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	unsigned long long units_REAL = 0, units_IndexType = 0;
	if (d_hyb->ellg.nnz > 0)
	{
		unsigned long long total_nell;
		IndexType i;
		for (i = 0, total_nell = 0; i < d_hyb->ellg.n; i++) total_nell += d_hyb->ellg.nell[i];
		//d_hyb->ellg.a + d_x + dst_y
		units_REAL += 2 * total_nell + d_hyb->ellg.n;
		//d_hyb->ellg.jcoeff + d_hyb->ellg.nell
		units_IndexType += total_nell + d_hyb->ellg.n;
	}
	if (d_hyb->csr.nnz > 0)
	{
		//d_hyb->csr.a + d_x + dst_y
		units_REAL += d_hyb->csr.nnz + d_hyb->csr.nnz + d_hyb->csr.n;
		//d_hyb->csr.ia + d_hyb->csr.ja
		units_IndexType += d_hyb->csr.n + d_hyb->csr.nnz;
	}
	//
	//Instruction count
	long double instr_count_ellg = 0;
	if (d_hyb->ellg.nnz > 0)
	{
		instr_count_ellg = 6 + 1 + *(d_hyb->ellg.nell + d_hyb->ellg.n) * 4 + 2 + *(d_hyb->ellg.nell + d_hyb->ellg.n) * 9 + 2;
		instr_count_ellg *= d_hyb->ellg.n;
	}
	//
	//Instruction count
	long double instr_count_csr = 0;
	if (d_hyb->csr.nnz > 0)
	{
		instr_count_csr = 6 + 1 + repeat * 4 + 2 + repeat * (5 + 1 + ((double)row_len / coop) * 12 + 5 + ((double)row_len / coop) * 8 + 2 + 1 + (max(1, log2(coop / 2)) * 4) + 2 + max(1, log2(coop / 2)) * 7 + 9);
		instr_count_csr *= d_hyb->csr.n;
	}
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
	//Macros
	std::string csr_macro = getGlobalConstants() +
							" -DCSR_REPEAT=" + std::to_string(repeat) +
							" -DCSR_COOP=" + std::to_string(coop) +
							" -DUNROLL_SHARED=" + std::to_string(coop / 4) +
							" -DN_MATRIX=" + std::to_string(d_hyb->csr.n) +
							" -DWORKGROUP_SIZE=" + std::to_string(csr_workgroup_size);
	std::string ellg_macro = getGlobalConstants() +
							" -DOVERRIDE_MEM=" + std::to_string(OVERRIDE_MEM) +
							" -DN_MATRIX=" + std::to_string(d_hyb->ellg.n) +
							" -DSTRIDE_MATRIX=" + std::to_string(d_hyb->ellg.stride);
	//
	cl::Program program_csr =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + CSR_KERNEL_FILE, context, device, csr_macro.c_str());
	cl::Kernel kernel_csr{ program_csr, "spmv_csr" };
	//
	cl::Program program_ellg =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + ELLG_KERNEL_FILE, context, device, ellg_macro.c_str());
	cl::Kernel kernel_ellg{ program_ellg, "spmv_ellg" };
	//
	printHeaderInfoGPU_HYB(d_hyb->n, d_hyb->nnz, deviceName, "\nCSR kernel macros: " + csr_macro + "\nELL kernel macros: " + ellg_macro, instr_count_csr, instr_count_ellg);
	//
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	// ellg related
	size_t byte_size_d_nell;
	size_t byte_size_d_jcoeff;
	size_t byte_size_d_a;
	//
	cl::Buffer d_nell_buffer;
	cl::Buffer d_jcoeff_buffer;
	cl::Buffer d_a_buffer;
	//
	if (d_hyb->ellg.nnz > 0)
	{
		byte_size_d_nell = (d_hyb->ellg.n + 1) * sizeof(cl_uint);
		byte_size_d_jcoeff = d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n) * sizeof(cl_uint);
		byte_size_d_a = d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n) * sizeof(CL_REAL);
		//
		d_nell_buffer = cl::Buffer{ context, CL_MEM_READ_ONLY, byte_size_d_nell };
		d_jcoeff_buffer = cl::Buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jcoeff };
		d_a_buffer = cl::Buffer{ context, CL_MEM_READ_ONLY, byte_size_d_a };
		//
		queue.enqueueWriteBuffer(d_nell_buffer, CL_TRUE, 0, byte_size_d_nell, d_hyb->ellg.nell);
		queue.enqueueWriteBuffer(d_jcoeff_buffer, CL_TRUE, 0, byte_size_d_jcoeff, d_hyb->ellg.jcoeff);
		queue.enqueueWriteBuffer(d_a_buffer, CL_TRUE, 0, byte_size_d_a, d_hyb->ellg.a);
		//
		kernel_ellg.setArg(0, d_nell_buffer);
		kernel_ellg.setArg(1, d_jcoeff_buffer);
		kernel_ellg.setArg(2, d_a_buffer);
		kernel_ellg.setArg(3, d_x_buffer);
		kernel_ellg.setArg(4, dst_y_buffer);
		//
#if OVERRIDE_MEM
		cl_ulong size;
		device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
		//
		size_t local_byte_size_dummy_mem = ellg_local_mem_size;
		//
		kernel_ellg.setArg(5, cl::Local(local_byte_size_dummy_mem));
		//
		std::cout << "!!! A work-group uses " << local_byte_size_dummy_mem << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
#endif
	}
	//
	// csr related
	size_t byte_size_d_ia;
	size_t byte_size_d_ja;
	size_t byte_size_d_val;
	//
	cl::Buffer d_ia_buffer;
	cl::Buffer d_ja_buffer;
	cl::Buffer d_val_buffer;
	//
	size_t local_byte_size_shdata;
	//
	if (d_hyb->csr.nnz > 0)
	{
		byte_size_d_ia = (d_hyb->csr.n + 1) * sizeof(cl_uint);
		byte_size_d_ja = d_hyb->csr.nnz * sizeof(cl_uint);
		byte_size_d_val = d_hyb->csr.nnz * sizeof(CL_REAL);
		//
		d_ia_buffer = cl::Buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ia };
		d_ja_buffer = cl::Buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ja };
		d_val_buffer = cl::Buffer{ context, CL_MEM_READ_ONLY, byte_size_d_val };
		//
		cl_ulong size;
		device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
		//
		local_byte_size_shdata = max(csr_workgroup_size * sizeof(CL_REAL), csr_local_mem_size);
		//
		queue.enqueueWriteBuffer(d_ia_buffer, CL_TRUE, 0, byte_size_d_ia, d_hyb->csr.ia);
		queue.enqueueWriteBuffer(d_ja_buffer, CL_TRUE, 0, byte_size_d_ja, d_hyb->csr.ja);
		queue.enqueueWriteBuffer(d_val_buffer, CL_TRUE, 0, byte_size_d_val, d_hyb->csr.a);
		//
		kernel_csr.setArg(0, d_ia_buffer);
		kernel_csr.setArg(1, d_ja_buffer);
		kernel_csr.setArg(2, d_val_buffer);
		kernel_csr.setArg(3, d_x_buffer);
		kernel_csr.setArg(4, dst_y_buffer);
		kernel_csr.setArg(5, cl::Local(local_byte_size_shdata));
		//
		std::cout << "!!! CSR kernel: repeat = " << repeat << ", coop = " << coop << ", nworkgroups = " << nworkgroups << " !!!" << std::endl << std::endl;
		std::cout << "!!! A work-group uses " << local_byte_size_shdata << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
	}
	//
    cl_ulong nanoseconds_ellg;
    cl_ulong nanoseconds_csr;
    cl_ulong total_nanoseconds_ellg = 0;
    cl_ulong total_nanoseconds_csr = 0;
    //
	for (int r = 0; r < REPEAT; r++)
	{
		nanoseconds_ellg = 0;
		nanoseconds_csr = 0;
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
		if (d_hyb->ellg.nnz > 0)
		{
			nanoseconds_ellg +=
				jc::run_and_time_kernel(kernel_ellg,
					queue,
					cl::NDRange(jc::best_fit(d_hyb->ellg.n, ellg_workgroup_size)),
					cl::NDRange(ellg_workgroup_size));
		}
		if (d_hyb->csr.nnz > 0)
		{
			nanoseconds_csr +=
				jc::run_and_time_kernel(kernel_csr,
					queue,
					cl::NDRange(nworkgroups * csr_workgroup_size),
					cl::NDRange(csr_workgroup_size));
		}
		printRunInfoGPU_HYB(r + 1, nanoseconds_csr, nanoseconds_ellg, (d_hyb->csr.nnz), (d_hyb->ellg.nnz), coop, units_REAL, units_IndexType, instr_count_csr, instr_count_ellg);
		total_nanoseconds_ellg += nanoseconds_ellg;
		total_nanoseconds_csr += nanoseconds_csr;
	}
    queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
    double average_nanoseconds_ellg = total_nanoseconds_ellg / (double)REPEAT;
    double average_nanoseconds_csr = total_nanoseconds_csr / (double)REPEAT;
	printAverageRunInfoGPU_HYB(average_nanoseconds_csr, average_nanoseconds_ellg, (d_hyb->csr.nnz), (d_hyb->ellg.nnz), coop, units_REAL, units_IndexType, instr_count_csr, instr_count_ellg);
	//increment all values
	for (IndexType i = 0; i < d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n); i++) d_hyb->ellg.jcoeff[i]++;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]++;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]++;

    return dst_y;
}

std::vector<CL_REAL> spmv_HYB_ELLG(struct hybellg_t* d_hyb, const std::vector<CL_REAL> d_x)
{
	return spmv_HYB_ELLG_param(d_hyb, d_x, CSR_WORKGROUP_SIZE, WORKGROUP_SIZE, 0, 0);
}
#endif

#if HYB_HLL_SEQ
std::vector<REAL> spmv_HYB_HLL_sequential(struct hybhll_t* d_hyb, const std::vector<REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_hyb->hll.total_mem; i++) d_hyb->hll.jcoeff[i]--;
	for (IndexType i = 0; i < d_hyb->hll.nhoff; i++) d_hyb->hll.hoff[i]--;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]--;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]--;
	//
	std::vector<REAL> dst_y(d_x.size(), 0);
	//
	printHeaderInfoSEQ(d_hyb->n, d_hyb->nnz);
	//
	unsigned long long units_REAL = 0, units_IndexType = 0;
	if (d_hyb->hll.nnz > 0)
	{
		unsigned long long total_nell;
		IndexType i;
		for (i = 0, total_nell = 0; i < d_hyb->hll.nhoff; i++) total_nell += d_hyb->hll.nell[i];
		//d_hyb->hll.a + d_x + dst_y
		units_REAL += 2 * total_nell + d_hyb->hll.n;
		//d_hyb->hll.jcoeff + d_hyb->hll.nell + d_hyb->hll.hoff
		units_IndexType += total_nell + d_hyb->hll.n + d_hyb->hll.n;
	}
	if (d_hyb->csr.nnz > 0)
	{
		//d_hyb->csr.a + d_x + dst_y
		units_REAL += d_hyb->csr.nnz + d_hyb->csr.nnz + d_hyb->csr.nnz;
		//d_hyb->csr.ia + d_hyb->csr.ja
		units_IndexType += d_hyb->csr.n + d_hyb->csr.nnz;
	}
	//
	unsigned long long nanoseconds = 0, total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		std::fill(dst_y.begin(), dst_y.end(), 0);
		nanoseconds = 0;
		if (d_hyb->hll.nnz > 0)
		{
			nanoseconds += HLL_sequential(&(d_hyb->hll), d_x, dst_y);
		}
		if (d_hyb->csr.nnz > 0)
		{
			nanoseconds += CSR_sequential(&(d_hyb->csr), d_x, dst_y);
		}
		printRunInfoSEQ(r + 1, nanoseconds, (d_hyb->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoSEQ(average_nanoseconds, (d_hyb->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < d_hyb->hll.total_mem; i++) d_hyb->hll.jcoeff[i]++;
	for (IndexType i = 0; i < d_hyb->hll.nhoff; i++) d_hyb->hll.hoff[i]++;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]++;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]++;

	return dst_y;
}
#endif

#if HYB_HLL
std::vector<CL_REAL> spmv_HYB_HLL_param(struct hybhll_t* d_hyb, const std::vector<CL_REAL> d_x, unsigned int csr_workgroup_size, unsigned int hll_workgroup_size, unsigned int csr_local_mem_size, unsigned int hll_local_mem_size)
{
	//decrement all values
	for (IndexType i = 0; i < d_hyb->hll.total_mem; i++) d_hyb->hll.jcoeff[i]--;
	for (IndexType i = 0; i < d_hyb->hll.nhoff; i++) d_hyb->hll.hoff[i]--;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]--;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]--;
	//
	IndexType i, row_len = 0, coop = 1, repeat = 1, nworkgroups;
	if (d_hyb->csr.nnz > 0)
	{
		for (i = 0; i < d_hyb->csr.n; i++) row_len += d_hyb->csr.ia[i + 1] - d_hyb->csr.ia[i];
		row_len = sqrt(row_len / d_hyb->csr.n);
		for (coop = 1; coop < 32 && row_len >= coop; coop <<= 1);
		nworkgroups = 1 + (d_hyb->csr.n * coop - 1) / (repeat * csr_workgroup_size);
		if (nworkgroups > 1500)
			for (repeat = 1; (1 + (d_hyb->csr.n * coop - 1) / ((repeat + 1) * csr_workgroup_size)) > 1500; repeat++);
		nworkgroups = 1 + (d_hyb->csr.n * coop - 1) / (repeat * csr_workgroup_size);
	}
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	unsigned long long units_REAL = 0, units_IndexType = 0;
	if (d_hyb->hll.nnz > 0)
	{
		unsigned long long total_nell;
		IndexType i;
		for (i = 0, total_nell = 0; i < d_hyb->hll.nhoff; i++) total_nell += d_hyb->hll.nell[i];
		//d_hyb->hll.a + d_x + dst_y
		units_REAL += 2 * total_nell + d_hyb->hll.n;
		//d_hyb->hll.jcoeff + d_hyb->hll.nell + d_hyb->hll.hoff
		units_IndexType += total_nell + d_hyb->hll.n + d_hyb->hll.n;
	}
	if (d_hyb->csr.nnz > 0)
	{
		//d_hyb->csr.a + d_x + dst_y
		units_REAL += d_hyb->csr.nnz + d_hyb->csr.nnz + d_hyb->csr.n;
		//d_hyb->csr.ia + d_hyb->csr.ja
		units_IndexType += d_hyb->csr.n + d_hyb->csr.nnz;
	}
	//
	//Instruction count
	long double instr_count_hll = 0;
	if (d_hyb->hll.nnz > 0)
	{
		instr_count_hll = 12 + 1 + *(d_hyb->hll.nell + d_hyb->hll.nhoff - 1) * 4 + 2 + *(d_hyb->hll.nell + d_hyb->hll.nhoff - 1) * 10 + 2;
		instr_count_hll *= d_hyb->hll.n;
	}
	//
	//Instruction count
	long double instr_count_csr = 0;
	if (d_hyb->csr.nnz > 0)
	{
		instr_count_csr = 6 + 1 + repeat * 4 + 2 + repeat * (5 + 1 + ((double)row_len / coop) * 12 + 5 + ((double)row_len / coop) * 8 + 2 + 1 + (max(1, log2(coop / 2)) * 4) + 2 + max(1, log2(coop / 2)) * 7 + 9);
		instr_count_csr *= d_hyb->csr.n;
	}
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
	for (unroll_val = 1; (*(d_hyb->hll.nell + d_hyb->hll.nhoff) / 2) >= unroll_val; unroll_val <<= 1);
	//
	//Macros
	std::string csr_macro = getGlobalConstants() +
							" -DCSR_REPEAT=" + std::to_string(repeat) +
							" -DCSR_COOP=" + std::to_string(coop) +
							" -DUNROLL_SHARED=" + std::to_string(coop / 4) +
							" -DN_MATRIX=" + std::to_string(d_hyb->csr.n) +
							" -DWORKGROUP_SIZE=" + std::to_string(csr_workgroup_size);
	std::string hll_macro = getGlobalConstants() +
							" -DOVERRIDE_MEM=" + std::to_string(OVERRIDE_MEM) +
							" -DHACKSIZE=" + std::to_string(HLL_HACKSIZE) +
							" -DN_MATRIX=" + std::to_string(d_hyb->hll.n) +
							" -DUNROLL=" + std::to_string(unroll_val);
	//
	cl::Program program_csr =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + CSR_KERNEL_FILE, context, device, csr_macro.c_str());
	cl::Kernel kernel_csr{ program_csr, "spmv_csr" };
	//
	cl::Program program_hll =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + HLL_KERNEL_FILE, context, device, hll_macro.c_str());
	cl::Kernel kernel_hll{ program_hll, "spmv_hll" };
	//
	printHeaderInfoGPU_HYB(d_hyb->n, d_hyb->nnz, deviceName, "\nCSR kernel macros: " + csr_macro + "\nELL kernel macros: " + hll_macro, instr_count_csr, instr_count_hll);
	//
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	// hll related
	size_t byte_size_d_nell;
	size_t byte_size_d_jcoeff;
	size_t byte_size_d_hoff;
	size_t byte_size_d_a;
	//
	cl::Buffer d_nell_buffer;
	cl::Buffer d_jcoeff_buffer;
	cl::Buffer d_hoff_buffer;
	cl::Buffer d_a_buffer;
	//
	if (d_hyb->hll.nnz > 0)
	{
		byte_size_d_nell = d_hyb->hll.nhoff * sizeof(cl_uint);
		byte_size_d_jcoeff = d_hyb->hll.total_mem * sizeof(cl_uint);
		byte_size_d_hoff = d_hyb->hll.nhoff * sizeof(cl_uint);
		byte_size_d_a = d_hyb->hll.total_mem * sizeof(CL_REAL);
		//
		d_nell_buffer = cl::Buffer{ context, CL_MEM_READ_ONLY, byte_size_d_nell };
		d_jcoeff_buffer = cl::Buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jcoeff };
		d_hoff_buffer = cl::Buffer{ context, CL_MEM_READ_ONLY, byte_size_d_hoff };
		d_a_buffer = cl::Buffer{ context, CL_MEM_READ_ONLY, byte_size_d_a };
		//
		queue.enqueueWriteBuffer(d_nell_buffer, CL_TRUE, 0, byte_size_d_nell, d_hyb->hll.nell);
		queue.enqueueWriteBuffer(d_jcoeff_buffer, CL_TRUE, 0, byte_size_d_jcoeff, d_hyb->hll.jcoeff);
		queue.enqueueWriteBuffer(d_hoff_buffer, CL_TRUE, 0, byte_size_d_hoff, d_hyb->hll.hoff);
		queue.enqueueWriteBuffer(d_a_buffer, CL_TRUE, 0, byte_size_d_a, d_hyb->hll.a);
		//
		kernel_hll.setArg(0, d_nell_buffer);
		kernel_hll.setArg(1, d_jcoeff_buffer);
		kernel_hll.setArg(2, d_hoff_buffer);
		kernel_hll.setArg(3, d_a_buffer);
		kernel_hll.setArg(4, d_x_buffer);
		kernel_hll.setArg(5, dst_y_buffer);
		//
#if OVERRIDE_MEM
		cl_ulong size;
		device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
		//
		size_t local_byte_size_dummy_mem = hll_local_mem_size;
		//
		kernel_hll.setArg(6, cl::Local(local_byte_size_dummy_mem));
		//
		std::cout << "!!! A work-group uses " << local_byte_size_dummy_mem << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
#endif
	}
	//
	// csr related
	size_t byte_size_d_ia;
	size_t byte_size_d_ja;
	size_t byte_size_d_val;
	//
	cl::Buffer d_ia_buffer;
	cl::Buffer d_ja_buffer;
	cl::Buffer d_val_buffer;
	//
	size_t local_byte_size_shdata;
	//
	if (d_hyb->csr.nnz > 0)
	{
		byte_size_d_ia = (d_hyb->csr.n + 1) * sizeof(cl_uint);
		byte_size_d_ja = d_hyb->csr.nnz * sizeof(cl_uint);
		byte_size_d_val = d_hyb->csr.nnz * sizeof(CL_REAL);
		//
		d_ia_buffer = cl::Buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ia };
		d_ja_buffer = cl::Buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ja };
		d_val_buffer = cl::Buffer{ context, CL_MEM_READ_ONLY, byte_size_d_val };
		//
		cl_ulong size;
		device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
		//
		local_byte_size_shdata = max(csr_workgroup_size * sizeof(CL_REAL), csr_local_mem_size);
		//
		queue.enqueueWriteBuffer(d_ia_buffer, CL_TRUE, 0, byte_size_d_ia, d_hyb->csr.ia);
		queue.enqueueWriteBuffer(d_ja_buffer, CL_TRUE, 0, byte_size_d_ja, d_hyb->csr.ja);
		queue.enqueueWriteBuffer(d_val_buffer, CL_TRUE, 0, byte_size_d_val, d_hyb->csr.a);
		//
		kernel_csr.setArg(0, d_ia_buffer);
		kernel_csr.setArg(1, d_ja_buffer);
		kernel_csr.setArg(2, d_val_buffer);
		kernel_csr.setArg(3, d_x_buffer);
		kernel_csr.setArg(4, dst_y_buffer);
		kernel_csr.setArg(5, cl::Local(local_byte_size_shdata));
		//
		std::cout << "!!! CSR kernel: repeat = " << repeat << ", coop = " << coop << ", nworkgroups = " << nworkgroups << " !!!" << std::endl << std::endl;
		std::cout << "!!! A work-group uses " << local_byte_size_shdata << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
	}
	//
	cl_ulong nanoseconds_hll;
	cl_ulong nanoseconds_csr;
	cl_ulong total_nanoseconds_hll = 0;
	cl_ulong total_nanoseconds_csr = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		nanoseconds_hll = 0;
		nanoseconds_csr = 0;
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
		if (d_hyb->hll.nnz > 0)
		{
			nanoseconds_hll +=
				jc::run_and_time_kernel(kernel_hll,
					queue,
					cl::NDRange(jc::best_fit(d_hyb->hll.n, hll_workgroup_size)),
					cl::NDRange(hll_workgroup_size));
		}
		if (d_hyb->csr.nnz > 0)
		{
			nanoseconds_csr +=
				jc::run_and_time_kernel(kernel_csr,
					queue,
					cl::NDRange(nworkgroups * csr_workgroup_size),
					cl::NDRange(csr_workgroup_size));
		}
		printRunInfoGPU_HYB(r + 1, nanoseconds_csr, nanoseconds_hll, (d_hyb->csr.nnz), (d_hyb->hll.nnz), coop, units_REAL, units_IndexType, instr_count_csr, instr_count_hll);
		total_nanoseconds_hll += nanoseconds_hll;
		total_nanoseconds_csr += nanoseconds_csr;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds_hll = total_nanoseconds_hll / (double)REPEAT;
	double average_nanoseconds_csr = total_nanoseconds_csr / (double)REPEAT;
	printAverageRunInfoGPU_HYB(average_nanoseconds_csr, average_nanoseconds_hll, (d_hyb->csr.nnz), (d_hyb->hll.nnz), coop, units_REAL, units_IndexType, instr_count_csr, instr_count_hll);
	//increment all values
	for (IndexType i = 0; i < d_hyb->hll.total_mem; i++) d_hyb->hll.jcoeff[i]++;
	for (IndexType i = 0; i < d_hyb->hll.nhoff; i++) d_hyb->hll.hoff[i]++;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]++;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]++;

	return dst_y;
}

std::vector<CL_REAL> spmv_HYB_HLL(struct hybhll_t* d_hyb, const std::vector<CL_REAL> d_x)
{
	return spmv_HYB_HLL_param(d_hyb, d_x, CSR_WORKGROUP_SIZE, WORKGROUP_SIZE, 0, 0);
}
#endif

#endif
#endif