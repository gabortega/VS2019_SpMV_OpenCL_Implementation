#include<compiler_config.h>

#if HYB_ELL_SEQ || HYB_ELLG_SEQ || HYB_HLL_SEQ || HYB_ELL || HYB_ELLG || HYB_HLL || HYB_HLL_LOCAL

#include<stdio.h>
#include<string>
#include<windows.h>

#include<util_misc.hpp>
#include<CL/cl.h>
#include<JC/util.hpp>
#include<IO/mmio.h>
#include<IO/convert_input.h>
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
	unsigned long nanoseconds = 0, total_nanoseconds = 0;
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
		printRunInfo(r + 1, nanoseconds, (d_hyb->nnz));
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_hyb->nnz));
	//increment all values
	for (IndexType i = 0; i < d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n); i++) d_hyb->ellg.jcoeff[i]++;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]++;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]++;

	return dst_y;
}
#endif

#if HYB_ELL
std::vector<CL_REAL> spmv_HYB_ELL(struct hybellg_t* d_hyb, const std::vector<CL_REAL> d_x)
{	
	//decrement all values
	for (IndexType i = 0; i < d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n); i++) d_hyb->ellg.jcoeff[i]--;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]--;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]--;
	//
	IndexType i, row_len = 0, coop, repeat = 1, nworkgroups;
	if (d_hyb->csr.nnz > 0)
	{
		for (i = 0; i < d_hyb->csr.n; i++) row_len += d_hyb->csr.ia[i + 1] - d_hyb->csr.ia[i];
		row_len = sqrt(row_len / d_hyb->csr.n);
		for (coop = 1; coop < 32 && row_len >= coop; coop <<= 1);
		nworkgroups = 1 + (d_hyb->csr.n * coop - 1) / (repeat * CSR_WORKGROUP_SIZE);
		if (nworkgroups > 1500)
			for (repeat = 1; (1 + (d_hyb->csr.n * coop - 1) / ((repeat + 1) * CSR_WORKGROUP_SIZE)) > 1500; repeat++);
		nworkgroups = 1 + (d_hyb->csr.n * coop - 1) / (repeat * CSR_WORKGROUP_SIZE);
	}
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	//Macros
	std::string csr_macro = "-DPRECISION=" + std::to_string(PRECISION) +
							" -DCSR_REPEAT=" + std::to_string(repeat) +
							" -DCSR_COOP=" + std::to_string(coop) +
							" -DN_MATRIX=" + std::to_string(d_hyb->csr.n);
	std::string ell_macro = "-DPRECISION=" + std::to_string(PRECISION) + 
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
		local_byte_size_shdata = CSR_WORKGROUP_SIZE * sizeof(CL_REAL);
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
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		nanoseconds = 0;
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_d_x, dst_y.data());
		if (d_hyb->ellg.nnz > 0)
		{
			nanoseconds +=
				jc::run_and_time_kernel(kernel_ell,
					queue,
					cl::NDRange(jc::best_fit(d_hyb->ellg.n, WORKGROUP_SIZE)),
					cl::NDRange(WORKGROUP_SIZE));
		}
		if (d_hyb->csr.nnz > 0)
		{
			nanoseconds +=
				jc::run_and_time_kernel(kernel_csr,
					queue,
					cl::NDRange(1500 * CSR_WORKGROUP_SIZE),
					cl::NDRange(CSR_WORKGROUP_SIZE));
		}
		printRunInfo(r + 1, nanoseconds, (d_hyb->nnz));
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_hyb->nnz));
	//increment all values
	for (IndexType i = 0; i < d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n); i++) d_hyb->ellg.jcoeff[i]++;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]++;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]++;

	return dst_y;
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
	unsigned long nanoseconds = 0, total_nanoseconds = 0;
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
		printRunInfo(r + 1, nanoseconds, (d_hyb->nnz));
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_hyb->nnz));
	//increment all values
	for (IndexType i = 0; i < d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n); i++) d_hyb->ellg.jcoeff[i]++;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]++;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]++;

	return dst_y;
}
#endif

#if HYB_ELLG
std::vector<CL_REAL> spmv_HYB_ELLG(struct hybellg_t* d_hyb, const std::vector<CL_REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n); i++) d_hyb->ellg.jcoeff[i]--;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]--;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]--;
	//
	IndexType i, row_len = 0, coop, repeat = 1, nworkgroups;
	if (d_hyb->csr.nnz > 0)
	{
		for (i = 0; i < d_hyb->csr.n; i++) row_len += d_hyb->csr.ia[i + 1] - d_hyb->csr.ia[i];
		row_len = sqrt(row_len / d_hyb->csr.n);
		for (coop = 1; coop < 32 && row_len >= coop; coop <<= 1);
		nworkgroups = 1 + (d_hyb->csr.n * coop - 1) / (repeat * CSR_WORKGROUP_SIZE);
		if (nworkgroups > 1500)
			for (repeat = 1; (1 + (d_hyb->csr.n * coop - 1) / ((repeat + 1) * CSR_WORKGROUP_SIZE)) > 1500; repeat++);
		nworkgroups = 1 + (d_hyb->csr.n * coop - 1) / (repeat * CSR_WORKGROUP_SIZE);
	}
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	//Macros
	std::string csr_macro = "-DPRECISION=" + std::to_string(PRECISION) +
							" -DCSR_REPEAT=" + std::to_string(repeat) +
							" -DCSR_COOP=" + std::to_string(coop) +
							" -DN_MATRIX=" + std::to_string(d_hyb->csr.n);
	std::string ellg_macro = "-DPRECISION=" + std::to_string(PRECISION) +
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
		local_byte_size_shdata = CSR_WORKGROUP_SIZE * sizeof(CL_REAL);
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
    cl_ulong nanoseconds;
    cl_ulong total_nanoseconds = 0;
    //
	for (int r = 0; r < REPEAT; r++)
	{
		nanoseconds = 0;
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_d_x, dst_y.data());
		if (d_hyb->ellg.nnz > 0)
		{
			nanoseconds +=
				jc::run_and_time_kernel(kernel_ellg,
					queue,
					cl::NDRange(jc::best_fit(d_hyb->ellg.n, WORKGROUP_SIZE)),
					cl::NDRange(WORKGROUP_SIZE));
		}
		if (d_hyb->csr.nnz > 0)
		{
			nanoseconds +=
				jc::run_and_time_kernel(kernel_csr,
					queue,
					cl::NDRange(1500 * CSR_WORKGROUP_SIZE),
					cl::NDRange(CSR_WORKGROUP_SIZE));
		}
		printRunInfo(r + 1, nanoseconds, (d_hyb->nnz));
		total_nanoseconds += nanoseconds;
	}
    queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
    double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_hyb->nnz));
	//increment all values
	for (IndexType i = 0; i < d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n); i++) d_hyb->ellg.jcoeff[i]++;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]++;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]++;

    return dst_y;
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
	unsigned long nanoseconds = 0, total_nanoseconds = 0;
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
		printRunInfo(r + 1, nanoseconds, (d_hyb->nnz));
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_hyb->nnz));
	//increment all values
	for (IndexType i = 0; i < d_hyb->hll.total_mem; i++) d_hyb->hll.jcoeff[i]++;
	for (IndexType i = 0; i < d_hyb->hll.nhoff; i++) d_hyb->hll.hoff[i]++;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]++;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]++;

	return dst_y;
}
#endif

#if HYB_HLL
std::vector<CL_REAL> spmv_HYB_HLL(struct hybhll_t* d_hyb, const std::vector<CL_REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_hyb->hll.total_mem; i++) d_hyb->hll.jcoeff[i]--;
	for (IndexType i = 0; i < d_hyb->hll.nhoff; i++) d_hyb->hll.hoff[i]--;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]--;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]--;
	//
	IndexType i, row_len = 0, coop, repeat = 1, nworkgroups;
	if (d_hyb->csr.nnz > 0)
	{
		for (i = 0; i < d_hyb->csr.n; i++) row_len += d_hyb->csr.ia[i + 1] - d_hyb->csr.ia[i];
		row_len = sqrt(row_len / d_hyb->csr.n);
		for (coop = 1; coop < 32 && row_len >= coop; coop <<= 1);
		nworkgroups = 1 + (d_hyb->csr.n * coop - 1) / (repeat * CSR_WORKGROUP_SIZE);
		if (nworkgroups > 1500)
			for (repeat = 1; (1 + (d_hyb->csr.n * coop - 1) / ((repeat + 1) * CSR_WORKGROUP_SIZE)) > 1500; repeat++);
		nworkgroups = 1 + (d_hyb->csr.n * coop - 1) / (repeat * CSR_WORKGROUP_SIZE);
	}
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	//Macros
	std::string csr_macro = "-DPRECISION=" + std::to_string(PRECISION) +
							" -DCSR_REPEAT=" + std::to_string(repeat) +
							" -DCSR_COOP=" + std::to_string(coop) +
							" -DN_MATRIX=" + std::to_string(d_hyb->csr.n);
	std::string hll_macro = "-DPRECISION=" + std::to_string(PRECISION) +
							" -DHACKSIZE=" + std::to_string(HLL_HACKSIZE) +
							" -DN_MATRIX=" + std::to_string(d_hyb->hll.n);
	//
	cl::Program program_csr =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + CSR_KERNEL_FILE, context, device, csr_macro.c_str());
	cl::Kernel kernel_csr{ program_csr, "spmv_csr" };
	//
	cl::Program program_hll =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + HLL_KERNEL_FILE, context, device, hll_macro.c_str());
	cl::Kernel kernel_hll{ program_hll, "spmv_hll" };
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
		local_byte_size_shdata = CSR_WORKGROUP_SIZE * sizeof(CL_REAL);
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
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		nanoseconds = 0;
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_d_x, dst_y.data());
		if (d_hyb->hll.nnz > 0)
		{
			nanoseconds +=
				jc::run_and_time_kernel(kernel_hll,
					queue,
					cl::NDRange(jc::best_fit(d_hyb->hll.n, WORKGROUP_SIZE)),
					cl::NDRange(WORKGROUP_SIZE));
		}
		if (d_hyb->csr.nnz > 0)
		{
			nanoseconds +=
				jc::run_and_time_kernel(kernel_csr,
					queue,
					cl::NDRange(1500 * CSR_WORKGROUP_SIZE),
					cl::NDRange(CSR_WORKGROUP_SIZE));
		}
		printRunInfo(r + 1, nanoseconds, (d_hyb->nnz));
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_hyb->nnz));
	//increment all values
	for (IndexType i = 0; i < d_hyb->hll.total_mem; i++) d_hyb->hll.jcoeff[i]++;
	for (IndexType i = 0; i < d_hyb->hll.nhoff; i++) d_hyb->hll.hoff[i]++;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]++;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]++;

	return dst_y;
}
#endif

#if HYB_HLL_LOCAL
std::vector<CL_REAL> spmv_HYB_HLL_LOCAL(struct hybhll_t* d_hyb, const std::vector<CL_REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_hyb->hll.total_mem; i++) d_hyb->hll.jcoeff[i]--;
	for (IndexType i = 0; i < d_hyb->hll.nhoff; i++) d_hyb->hll.hoff[i]--;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]--;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]--;
	//
	IndexType i, row_len = 0, coop, repeat = 1, nworkgroups;
	if (d_hyb->csr.nnz > 0)
	{
		for (i = 0; i < d_hyb->csr.n; i++) row_len += d_hyb->csr.ia[i + 1] - d_hyb->csr.ia[i];
		row_len = sqrt(row_len / d_hyb->csr.n);
		for (coop = 1; coop < 32 && row_len >= coop; coop <<= 1);
		nworkgroups = 1 + (d_hyb->csr.n * coop - 1) / (repeat * CSR_WORKGROUP_SIZE);
		if (nworkgroups > 1500)
			for (repeat = 1; (1 + (d_hyb->csr.n * coop - 1) / ((repeat + 1) * CSR_WORKGROUP_SIZE)) > 1500; repeat++);
		nworkgroups = 1 + (d_hyb->csr.n * coop - 1) / (repeat * CSR_WORKGROUP_SIZE);
	}
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	//Macros
	std::string csr_macro = "-DPRECISION=" + std::to_string(PRECISION) +
							" -DCSR_REPEAT=" + std::to_string(repeat) +
							" -DCSR_COOP=" + std::to_string(coop) +
							" -DN_MATRIX=" + std::to_string(d_hyb->csr.n);
	std::string hll_macro = "-DPRECISION=" + std::to_string(PRECISION) +
							" -DHACKSIZE=" + std::to_string(HLL_HACKSIZE) +
							" -DN_MATRIX=" + std::to_string(d_hyb->hll.n);
	//
	cl::Program program_csr =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + CSR_KERNEL_FILE, context, device, csr_macro.c_str());
	cl::Kernel kernel_csr{ program_csr, "spmv_csr" };
	//
	cl::Program program_hll_local =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + HLL_LOCAL_KERNEL_FILE, context, device, hll_macro.c_str());
	cl::Kernel kernel_hll_local{ program_hll_local, "spmv_hll_local" };
	//
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	// hll_local related
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
		cl_ulong size;
		device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
		//
		size_t local_byte_size_shhoff = WORKGROUP_SIZE / HLL_HACKSIZE * sizeof(cl_uint);
		//
		kernel_hll_local.setArg(0, d_nell_buffer);
		kernel_hll_local.setArg(1, d_jcoeff_buffer);
		kernel_hll_local.setArg(2, d_hoff_buffer);
		kernel_hll_local.setArg(3, d_a_buffer);
		kernel_hll_local.setArg(4, d_x_buffer);
		kernel_hll_local.setArg(5, dst_y_buffer);
		kernel_hll_local.setArg(6, cl::Local(local_byte_size_shhoff));
		//
		std::cout << "!!! HLL_LOCAL: A work-group uses " << local_byte_size_shhoff << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
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
		local_byte_size_shdata = CSR_WORKGROUP_SIZE * sizeof(CL_REAL);
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
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		nanoseconds = 0;
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_d_x, dst_y.data());
		if (d_hyb->hll.nnz > 0)
		{
			nanoseconds +=
				jc::run_and_time_kernel(kernel_hll_local,
					queue,
					cl::NDRange(jc::best_fit(d_hyb->hll.n, WORKGROUP_SIZE)),
					cl::NDRange(WORKGROUP_SIZE));
		}
		if (d_hyb->csr.nnz > 0)
		{
			nanoseconds +=
				jc::run_and_time_kernel(kernel_csr,
					queue,
					cl::NDRange(1500 * CSR_WORKGROUP_SIZE),
					cl::NDRange(CSR_WORKGROUP_SIZE));
		}
		printRunInfo(r + 1, nanoseconds, (d_hyb->nnz));
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_hyb->nnz));

	//increment all values
	for (IndexType i = 0; i < d_hyb->hll.total_mem; i++) d_hyb->hll.jcoeff[i]++;
	for (IndexType i = 0; i < d_hyb->hll.nhoff; i++) d_hyb->hll.hoff[i]++;
	for (IndexType i = 0; i < d_hyb->csr.n + 1; i++) d_hyb->csr.ia[i]++;
	for (IndexType i = 0; i < d_hyb->csr.nnz; i++) d_hyb->csr.ja[i]++;

	return dst_y;
}
#endif

int main(void)
{
	// Error checking
#if HYB_HLL_LOCAL
	if ((WORKGROUP_SIZE / HLL_HACKSIZE) * HLL_HACKSIZE != WORKGROUP_SIZE)
	{
		std::cout << "!!! ERROR: WORKGROUP_SIZE MUST BE A MULTIPLE OF HLL_HACKSIZE !!!" << std::endl;
		system("PAUSE");
		exit(1);
	}
#endif

	FILE* f;
	struct coo_t coo;
	struct csr_t csr;
#if HYB_ELL_SEQ || HYB_ELL || HYB_ELLG_SEQ || HYB_ELLG
	struct hybellg_t hyb_ellg;
#endif
#if HYB_HLL_SEQ || HYB_HLL || HYB_HLL_LOCAL
	struct hybhll_t hybhll_t;
#endif

	std::string input_filename = (INPUT_FOLDER + (std::string)"/" + INPUT_FILE);
	
	if (createOutputDirectory(OUTPUT_FOLDER, HYB_OUTPUT_FOLDER))
		exit(1);
	std::string output_file = (OUTPUT_FOLDER + (std::string)"/" + HYB_OUTPUT_FOLDER + (std::string)"/" + OUTPUT_FILENAME + getTimeOfRun() + OUTPUT_FILEFORMAT);

	std::cout << "!!! OUTPUT IS BEING WRITTEN TO "<< output_file <<" !!!" << std::endl;
	std::cout << "!!! PROGRAM WILL EXIT AUTOMATICALLY AFTER PROCESSING; PRESS CTRL-C TO ABORT !!!" << std::endl;
	system("PAUSE");
	freopen_s(&f, output_file.c_str(), "w", stdout);

	std::cout << "-- LOADING INPUT FILE " << input_filename << " --" << std::endl;
	MM_To_COO(input_filename.c_str(), &coo, COO_LOG);
	std::cout << "-- INPUT FILE LOADED --" << std::endl << std::endl;
	std::cout << "-- PRE-PROCESSING INPUT --" << std::endl;
	IndexType n = coo.n;
	COO_To_CSR(&coo, &csr, CSR_LOG);
	FreeCOO(&coo);
#if HYB_ELL_SEQ || HYB_ELL || HYB_ELLG_SEQ || HYB_ELLG
	CSR_To_HYBELLG(&csr, &hyb_ellg, HYB_ELLG_LOG);
#endif
#if HYB_HLL_SEQ || HYB_HLL || HYB_HLL_LOCAL
	CSR_To_HYBHLL(&csr, &hybhll_t, HYB_HLL_LOG);
#endif
	FreeCSR(&csr);
	std::cout << "-- DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;

	std::vector<CL_REAL> x = std::vector<CL_REAL>();
	for (IndexType i = 0; i < n; i++)
		x.push_back(i);

#if HYB_ELL_SEQ
	std::cout << std::endl << "-- STARTING HYB_ELL SEQUENTIAL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y1 = spmv_HYB_ELL_sequential(&hyb_ellg, x);
	std::cout << std::endl << "-- FINISHED HYB_ELL SEQUENTIAL OPERATION --" << std::endl << std::endl;
	if (HYB_ELL_SEQ_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y1.size(); i++)
			std::cout << y1[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HYB_ELL
	std::cout << std::endl << "-- STARTING HYB_ELL KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y2 = spmv_HYB_ELL(&hyb_ellg, x);
	std::cout << std::endl << "-- FINISHED HYB_ELL KERNEL OPERATION --" << std::endl << std::endl;
	if (HYB_ELL_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y2.size(); i++)
			std::cout << y2[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HYB_ELLG_SEQ
	std::cout << std::endl << "-- STARTING HYB_ELL-G SEQUENTIAL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y3 = spmv_HYB_ELLG_sequential(&hyb_ellg, x);
	std::cout << std::endl << "-- FINISHED HYB_ELL-G SEQUENTIAL OPERATION --" << std::endl << std::endl;
	if (HYB_ELLG_SEQ_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y3.size(); i++)
			std::cout << y3[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HYB_ELLG
	std::cout << std::endl << "-- STARTING HYB_ELL-G KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y4 = spmv_HYB_ELLG(&hyb_ellg, x);
	std::cout << std::endl << "-- FINISHED HYB_ELL-G KERNEL OPERATION --" << std::endl << std::endl;
	if (HYB_ELLG_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y4.size(); i++)
			std::cout << y4[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HYB_HLL_SEQ
	std::cout << std::endl << "-- STARTING HYB_HLL SEQUENTIAL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y5 = spmv_HYB_HLL_sequential(&hybhll_t, x);
	std::cout << std::endl << "-- FINISHED HYB_HLL SEQUENTIAL OPERATION --" << std::endl << std::endl;
	if (HYB_HLL_SEQ_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y5.size(); i++)
			std::cout << y5[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HYB_HLL
	std::cout << std::endl << "-- STARTING HYB_HLL KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y6 = spmv_HYB_HLL(&hybhll_t, x);
	std::cout << std::endl << "-- FINISHED HYB_HLL KERNEL OPERATION --" << std::endl << std::endl;
	if (HYB_HLL_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y6.size(); i++)
			std::cout << y6[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HYB_HLL_LOCAL
	std::cout << std::endl << "-- STARTING HYB_HLL_LOCAL KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y7 = spmv_HYB_HLL_LOCAL(&hybhll_t, x);
	std::cout << std::endl << "-- FINISHED HYB_HLL_LOCAL KERNEL OPERATION --" << std::endl << std::endl;
	if (HYB_HLL_LOCAL_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y7.size(); i++)
			std::cout << y7[i] << " ";
		std::cout << std::endl;
	}
#endif

	x.clear();
#if HYB_ELL_SEQ || HYB_ELL || HYB_ELLG_SEQ || HYB_ELLG
	FreeHYBELLG(&hyb_ellg);
#if HYB_ELL_SEQ
	y1.clear();
#endif
#if HYB_ELL
	y2.clear();
#endif
#if HYB_ELLG_SEQ
	y3.clear();
#endif
#if HYB_ELLG
	y4.clear();
#endif
#endif
#if HYB_HLL_SEQ || HYB_HLL || HYB_HLL_LOCAL
	FreeHYBHLL(&hybhll_t);
#if HYB_HLL
	y5.clear();
#endif
#if HYB_HLL
	y6.clear();
#endif
#if HYB_HLL_LOCAL
	y7.clear();
#endif
#endif
#if DEBUG
	system("PAUSE"); // for debugging
#endif

	return 0;
}

#endif