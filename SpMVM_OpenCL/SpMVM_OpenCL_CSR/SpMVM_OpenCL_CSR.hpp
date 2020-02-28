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
		printRunInfo(r + 1, nanoseconds, (d_csr->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_csr->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < d_csr->n + 1; i++) d_csr->ia[i]++;
	for (IndexType i = 0; i < d_csr->nnz; i++) d_csr->ja[i]++;

	return dst_y;
}
#endif

#if CSR
std::vector<CL_REAL> spmv_CSR(struct csr_t* d_csr, const std::vector<CL_REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_csr->n + 1; i++) d_csr->ia[i]--;
	for (IndexType i = 0; i < d_csr->nnz; i++) d_csr->ja[i]--;
	//
	IndexType i, row_len = 0, coop, repeat = 1, nworkgroups;
	for (i = 0; i < d_csr->n; i++) row_len += d_csr->ia[i + 1] - d_csr->ia[i];
	row_len = sqrt(row_len/d_csr->n);
	for (coop = 1; coop < 32 && row_len >= coop; coop <<= 1);
	nworkgroups = 1 + (d_csr->n * coop - 1) / (repeat * CSR_WORKGROUP_SIZE);
	if (nworkgroups > 1500)
		for (repeat = 1; (1 + (d_csr->n * coop - 1) / ((repeat + 1) * CSR_WORKGROUP_SIZE)) > 1500; repeat++);
	nworkgroups = 1 + (d_csr->n * coop - 1) / (repeat * CSR_WORKGROUP_SIZE);
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//d_csr->a + d_x + dst_y
	unsigned long long units_REAL = d_csr->nnz + d_csr->nnz + d_csr->n;
	//d_csr->ia + d_csr->ja
	unsigned long long units_IndexType = d_csr->n + d_csr->nnz;
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
						" -DCSR_REPEAT=" + std::to_string(repeat) +
						" -DCSR_COOP=" + std::to_string(coop) +
						" -DUNROLL_SHARED=" + std::to_string(coop/4) +
						" -DN_MATRIX=" + std::to_string(d_csr->n);
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + CSR_KERNEL_FILE, context, device, macro.c_str());
	cl::Kernel kernel{ program, "spmv_csr" };
	//
	printHeaderInfoGPU(d_csr->n, d_csr->nnz, deviceName, macro);
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
	size_t local_byte_size_shdata = CSR_WORKGROUP_SIZE * sizeof(CL_REAL);
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
				cl::NDRange(nworkgroups * CSR_WORKGROUP_SIZE),
				cl::NDRange(CSR_WORKGROUP_SIZE));
		printRunInfo(r + 1, nanoseconds, (d_csr->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_csr->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < d_csr->n + 1; i++) d_csr->ia[i]++;
	for (IndexType i = 0; i < d_csr->nnz; i++) d_csr->ja[i]++;

	return dst_y;
}
#endif

#endif
#endif