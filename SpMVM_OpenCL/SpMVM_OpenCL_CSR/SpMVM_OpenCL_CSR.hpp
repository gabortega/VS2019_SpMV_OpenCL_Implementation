#include<compiler_config.h>

#if CSR_SEQ || CSR
#ifndef OPENCL_CSR_H
#define OPENCL_CSR_H

#include<windows.h>

#include<util_misc.hpp>
#include<CL/cl.h>
#include<JC/util.hpp>
#include<SEQ/CSR.hpp>

#if PRECISION == 2
#define CL_REAL cl_double
#else
//#elif PRECISION == 1
#define CL_REAL cl_float
//#else
//#define CL_REAL cl_half // TODO?
#endif

#if CSR_SEQ
std::vector<REAL> spmv_CSR_sequential(struct csr_t* d_csr, const std::vector<REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_csr->n + 1; i++) d_csr->ia[i]--;
	for (IndexType i = 0; i < d_csr->nnz; i++) d_csr->ja[i]--;
	//
	std::vector<REAL> dst_y(d_x.size(), 0);
	//
	printHeaderInfoSEQ(d_csr->n, d_csr->nnz);
	//
	//d_csr->a + d_x + dst_y
	unsigned long long units_REAL = d_csr->nnz + d_csr->nnz + d_csr->nnz;
	//d_csr->ia + d_csr->ja
	unsigned long long units_IndexType = d_csr->n + d_csr->nnz;
	//
	unsigned long long nanoseconds = 0, total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		std::fill(dst_y.begin(), dst_y.end(), 0);
		nanoseconds = CSR_sequential(d_csr, d_x, dst_y);
		printRunInfoSEQ(r + 1, nanoseconds, (d_csr->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoSEQ(average_nanoseconds, (d_csr->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < d_csr->n + 1; i++) d_csr->ia[i]++;
	for (IndexType i = 0; i < d_csr->nnz; i++) d_csr->ja[i]++;

	return dst_y;
}
#endif

#if CSR
std::vector<CL_REAL> spmv_CSR_param(struct csr_t* d_csr, const std::vector<CL_REAL> d_x, unsigned int workgroup_size, unsigned int local_mem_size, unsigned int thread_count)
{
	//decrement all values
	for (IndexType i = 0; i < d_csr->n + 1; i++) d_csr->ia[i]--;
	for (IndexType i = 0; i < d_csr->nnz; i++) d_csr->ja[i]--;
	//
	IndexType i, row_len = 0, coop, repeat = 1, nworkgroups, row_len_sqrt = 0;
	for (i = 0; i < d_csr->n; i++) row_len += d_csr->ia[i + 1] - d_csr->ia[i];
	row_len /= d_csr->n;
	row_len_sqrt = sqrt(row_len);
	for (coop = 1; coop < 32 && row_len_sqrt >= coop; coop <<= 1);
#if !OVERRIDE_THREADS
	nworkgroups = 1 + (d_csr->n * coop - 1) / (repeat * workgroup_size);
	if (nworkgroups > CSR_WORKGROUP_COUNT_THRESHOLD)
		for (repeat = 1; (1 + (d_csr->n * coop - 1) / ((repeat + 1) * workgroup_size)) > CSR_WORKGROUP_COUNT_THRESHOLD; repeat++);
	nworkgroups = 1 + (d_csr->n * coop - 1) / (repeat * workgroup_size);
#else
	nworkgroups = 1 + (thread_count * coop - 1) / (repeat * workgroup_size);
	if (nworkgroups > (thread_count / workgroup_size))
		for (repeat = 1; (1 + (thread_count * coop - 1) / ((repeat + 1) * workgroup_size)) > (thread_count / workgroup_size); repeat++);
	nworkgroups = 1 + (thread_count * coop - 1) / (repeat * workgroup_size);
#endif
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//d_csr->a + d_x + dst_y
	unsigned long long units_REAL = d_csr->nnz + d_csr->nnz + d_csr->n;
	//d_csr->ia + d_csr->ja
	unsigned long long units_IndexType = d_csr->n + d_csr->nnz;
#if !OVERRIDE_THREADS
	//
	//Instruction count
	long double instr_count = 8 + 1 + repeat * 4 + 2 + repeat * (5 + 1 + ((double)row_len / coop) * 12 + 5 + ((double)row_len / coop) * 8 + 2 + 1 + (max(1, log2(coop / 2)) * 4) + 2 + max(1, log2(coop / 2)) * 7 + 9);
	//
#else
	//
	//Instruction count
	long double instr_count = 14 + 1 + repeat * 4 + 2 + repeat * (5 + 1 + ((double)row_len / coop) * 12 + 5 + ((double)row_len / coop) * 8 + 2 + 1 + (max(1, log2(coop / 2)) * 4) + 2 + max(1, log2(coop / 2)) * 7 + 14);
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
	std::string macro = "-DPRECISION=" + std::to_string(PRECISION) +
						" -DCSR_REPEAT=" + std::to_string(repeat) +
						" -DCSR_COOP=" + std::to_string(coop) +
						" -DUNROLL_SHARED=" + std::to_string(coop/4) +
						" -DN_MATRIX=" + std::to_string(d_csr->n) +
						" -DWORKGROUP_SIZE=" + std::to_string(workgroup_size);
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + CSR_KERNEL_FILE, context, device, macro.c_str());
#if !OVERRIDE_THREADS
	cl::Kernel kernel{ program, "spmv_csr" };
#else
	cl::Kernel kernel{ program, "occ_spmv_csr" };
#endif
	//
	printHeaderInfoGPU(d_csr->n, d_csr->nnz, deviceName, macro, instr_count);
	//
	size_t byte_size_d_ia = (d_csr->n + 1) * sizeof(cl_uint);
	size_t byte_size_d_ja = d_csr->nnz * sizeof(cl_uint);
	size_t byte_size_d_a = d_csr->nnz * sizeof(CL_REAL);
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_ia_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ia };
	cl::Buffer d_ja_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ja };
	cl::Buffer d_a_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_a };
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	cl_ulong size;
	device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
	//
	size_t local_byte_size_shdata = max(workgroup_size * sizeof(CL_REAL), local_mem_size);
	//
	queue.enqueueWriteBuffer(d_ia_buffer, CL_TRUE, 0, byte_size_d_ia, d_csr->ia);
	queue.enqueueWriteBuffer(d_ja_buffer, CL_TRUE, 0, byte_size_d_ja, d_csr->ja);
	queue.enqueueWriteBuffer(d_a_buffer, CL_TRUE, 0, byte_size_d_a, d_csr->a);
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	kernel.setArg(0, d_ia_buffer);
	kernel.setArg(1, d_ja_buffer);
	kernel.setArg(2, d_a_buffer);
	kernel.setArg(3, d_x_buffer);
	kernel.setArg(4, dst_y_buffer);
	kernel.setArg(5, cl::Local(local_byte_size_shdata));
	//
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	std::cout << "!!! CSR kernel: repeat = " << repeat << ", coop = " << coop << ", nworkgroups = " << nworkgroups << " !!!" << std::endl << std::endl;
	std::cout << "!!! A work-group uses " << local_byte_size_shdata << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
	for (int r = 0; r < REPEAT; r++)
	{
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
		nanoseconds =
			jc::run_and_time_kernel(kernel,
				queue,
				cl::NDRange(nworkgroups * workgroup_size),
				cl::NDRange(workgroup_size));
		printRunInfoGPU_CSR(r + 1, nanoseconds, (d_csr->nnz), coop, units_REAL, units_IndexType, instr_count);
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfoGPU_CSR(average_nanoseconds, (d_csr->nnz), coop, units_REAL, units_IndexType, instr_count);
	//increment all values
	for (IndexType i = 0; i < d_csr->n + 1; i++) d_csr->ia[i]++;
	for (IndexType i = 0; i < d_csr->nnz; i++) d_csr->ja[i]++;

	return dst_y;
}

std::vector<CL_REAL> spmv_CSR(struct csr_t* d_csr, const std::vector<CL_REAL> d_x)
{
	return spmv_CSR_param(d_csr, d_x, CSR_WORKGROUP_SIZE, 0, 0);
}
#endif

#endif
#endif