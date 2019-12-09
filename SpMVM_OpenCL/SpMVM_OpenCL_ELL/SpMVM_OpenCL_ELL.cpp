#include<compiler_config.h>

#if ELL_SEQ || ELL || ELLG_SEQ || ELLG || HLL_SEQ || HLL || HLL_LOCAL

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
	unsigned long nanoseconds = 0, total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		std::fill(dst_y.begin(), dst_y.end(), 0);
		nanoseconds = ELL_sequential(d_ell, d_x, dst_y);
		printRunInfo(r + 1, nanoseconds, (d_ell->nnz));
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_ell->nnz));
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
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
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
	std::cout << "Kernel macros: " << macro << std::endl << std::endl;
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
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_d_x, dst_y.data());
		nanoseconds =
			jc::run_and_time_kernel(kernel,
				queue,
				cl::NDRange(jc::best_fit(d_ell->n, WORKGROUP_SIZE)),
				cl::NDRange(WORKGROUP_SIZE));
		std::cout << "Run: " << r + 1 << " | Time elapsed: " << nanoseconds << " ns | Effective throughput: " << 2 * (d_ell->nnz) / (nanoseconds * 1e-9) / 1e9 << "GFLOP/s\n";
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	std::cout << std::endl << "Average time: " << average_nanoseconds << " ns | Average effective throughput: " << 2 * (d_ell->nnz) / (average_nanoseconds * 1e-9) / 1e9 << "GFLOP/s\n";
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
	unsigned long nanoseconds = 0, total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		std::fill(dst_y.begin(), dst_y.end(), 0);
		nanoseconds = ELLG_sequential(d_ellg, d_x, dst_y);
		printRunInfo(r + 1, nanoseconds, (d_ellg->nnz));
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_ellg->nnz));
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
    //
    cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
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
	std::cout << "Kernel macros: " << macro << std::endl << std::endl;
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
        queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_d_x, dst_y.data());
        nanoseconds =
            jc::run_and_time_kernel(kernel,
                queue,
				cl::NDRange(jc::best_fit(d_ellg->n, WORKGROUP_SIZE)),
				cl::NDRange(WORKGROUP_SIZE));
		std::cout << "Run: " << r + 1 << " | Time elapsed: " << nanoseconds << " ns | Effective throughput: " << 2 * (d_ellg->nnz) / (nanoseconds * 1e-9) / 1e9 << "GFLOP/s\n";
        total_nanoseconds += nanoseconds;
    }
    queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
    double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	std::cout << std::endl << "Average time: " << average_nanoseconds << " ns | Average effective throughput: " << 2 * (d_ellg->nnz) / (average_nanoseconds * 1e-9) / 1e9 << "GFLOP/s\n";
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
	unsigned long nanoseconds = 0, total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		std::fill(dst_y.begin(), dst_y.end(), 0);
		nanoseconds = HLL_sequential(d_hll, d_x, dst_y);
		printRunInfo(r + 1, nanoseconds, (d_hll->nnz));
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_hll->nnz));
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
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
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
	std::cout << "Kernel macros: " << macro << std::endl << std::endl;
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
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_d_x, dst_y.data());
		nanoseconds =
			jc::run_and_time_kernel(kernel,
				queue,
				cl::NDRange(jc::best_fit(d_hll->n, WORKGROUP_SIZE)),
				cl::NDRange(WORKGROUP_SIZE));
		std::cout << "Run: " << r + 1 << " | Time elapsed: " << nanoseconds << " ns | Effective throughput: " << 2 * (d_hll->nnz) / (nanoseconds * 1e-9) / 1e9 << "GFLOP/s\n";
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	std::cout << std::endl << "Average time: " << average_nanoseconds << " ns | Average effective throughput: " << 2 * (d_hll->nnz) / (average_nanoseconds * 1e-9) / 1e9 << "GFLOP/s\n";
	//increment all values
	for (IndexType i = 0; i < d_hll->total_mem; i++) d_hll->jcoeff[i]++;
	for (IndexType i = 0; i < d_hll->nhoff; i++) d_hll->hoff[i]++;

	return dst_y;
}
#endif

#if HLL_LOCAL
std::vector<CL_REAL> spmv_HLL_LOCAL(struct hll_t* d_hll, const std::vector<CL_REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_hll->total_mem; i++) d_hll->jcoeff[i]--;
	for (IndexType i = 0; i < d_hll->nhoff; i++) d_hll->hoff[i]--;
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
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
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + HLL_LOCAL_KERNEL_FILE, context, device, macro.c_str());
	cl::Kernel kernel{ program, "spmv_hll_local" };
	//
	std::cout << "Kernel macros: " << macro << std::endl << std::endl;
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
	cl_ulong size;
	device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
	//
	size_t local_byte_size_shhoff = WORKGROUP_SIZE / HLL_HACKSIZE * sizeof(cl_uint);
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
	kernel.setArg(6, cl::Local(local_byte_size_shhoff));
	//
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	std::cout << "!!! A work-group uses " << local_byte_size_shhoff << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
	for (int r = 0; r < REPEAT; r++)
	{
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_d_x, dst_y.data());
		nanoseconds =
			jc::run_and_time_kernel(kernel,
				queue,
				cl::NDRange(jc::best_fit(d_hll->n, WORKGROUP_SIZE)),
				cl::NDRange(WORKGROUP_SIZE));
		std::cout << "Run: " << r + 1 << " | Time elapsed: " << nanoseconds << " ns | Effective throughput: " << 2 * (d_hll->nnz) / (nanoseconds * 1e-9) / 1e9 << "GFLOP/s\n";
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	std::cout << std::endl << "Average time: " << average_nanoseconds << " ns | Average effective throughput: " << 2 * (d_hll->nnz) / (average_nanoseconds * 1e-9) / 1e9 << "GFLOP/s\n";
	//increment all values
	for (IndexType i = 0; i < d_hll->total_mem; i++) d_hll->jcoeff[i]++;
	for (IndexType i = 0; i < d_hll->nhoff; i++) d_hll->hoff[i]++;

	return dst_y;
}
#endif

int main(void)
{
	// Error checking
#if HLL_LOCAL
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
#if ELL_SEQ || ELL || ELLG
	struct ellg_t ellg;
#endif
#if HLL_SEQ || HLL || HLL_LOCAL
	struct hll_t hll;
#endif

	std::string input_filename = (INPUT_FOLDER + (std::string)"/" + INPUT_FILE);
	
	if (createOutputDirectory(OUTPUT_FOLDER, ELL_OUTPUT_FOLDER))
		exit(1);
	std::string output_file = (OUTPUT_FOLDER + (std::string)"/" + ELL_OUTPUT_FOLDER + (std::string)"/" + OUTPUT_FILENAME + getTimeOfRun() + OUTPUT_FILEFORMAT);

	std::cout << "!!! OUTPUT IS BEING WRITTEN TO "<< output_file <<" !!!" << std::endl;
	std::cout << "!!! PROGRAM WILL EXIT AUTOMATICALLY AFTER PROCESSING; PRESS CTRL-C TO ABORT !!!" << std::endl;
	system("PAUSE");
	freopen_s(&f, output_file.c_str(), "w", stdout);

	std::cout << "-- LOADING INPUT FILE " << input_filename << " --" << std::endl;
	MM_To_COO(input_filename.c_str(), &coo, COO_LOG);
	IndexType n = coo.n;
	std::cout << "-- INPUT FILE LOADED --" << std::endl << std::endl;
	std::cout << "-- PRE-PROCESSING INPUT --" << std::endl;
	COO_To_CSR(&coo, &csr, CSR_LOG);
#if ELL_SEQ || ELL || ELLG
	if(!CSR_To_ELLG(&csr, &ellg, ELLG_LOG))
		std::cout << "ELL-G HAS BEEN TRUNCATED" << std::endl;
#endif
#if HLL_SEQ || HLL || HLL_LOCAL
	if (!CSR_To_HLL(&csr, &hll, HLL_LOG))
		std::cout << "HLL HAS BEEN TRUNCATED" << std::endl;
#endif
	FreeCOO(&coo);
	FreeCSR(&csr);
	std::cout << "-- DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;

	std::vector<CL_REAL> x = std::vector<CL_REAL>();
	for (IndexType i = 0; i < n; i++)
		x.push_back(i);

#if ELL_SEQ
	std::cout << std::endl << "-- STARTING ELL SEQUENTIAL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y1 = spmv_ELL_sequential(&ellg, x);
	std::cout << std::endl << "-- FINISHED ELL SEQUENTIAL OPERATION --" << std::endl << std::endl;
	if (ELL_SEQ_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y1.size(); i++)
			std::cout << y1[i] << " ";
		std::cout << std::endl;
	}
#endif
#if ELL
	std::cout << std::endl << "-- STARTING ELL KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y2 = spmv_ELL(&ellg, x);
	std::cout << std::endl << "-- FINISHED ELL KERNEL OPERATION --" << std::endl << std::endl;
	if (ELL_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y2.size(); i++)
			std::cout << y2[i] << " ";
		std::cout << std::endl;
	}
#endif
#if ELLG_SEQ
	std::cout << std::endl << "-- STARTING ELL-G SEQUENTIAL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y3 = spmv_ELLG_sequential(&ellg, x);
	std::cout << std::endl << "-- FINISHED ELL-G SEQUENTIAL OPERATION --" << std::endl << std::endl;
	if (ELLG_SEQ_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y3.size(); i++)
			std::cout << y3[i] << " ";
		std::cout << std::endl;
	}
#endif
#if ELLG
	std::cout << std::endl << "-- STARTING ELL-G KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y4 = spmv_ELLG(&ellg, x);
	std::cout << std::endl << "-- FINISHED ELL-G KERNEL OPERATION --" << std::endl << std::endl;
	if (ELLG_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y4.size(); i++)
			std::cout << y4[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HLL_SEQ
	std::cout << std::endl << "-- STARTING HLL SEQUENTIAL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y5 = spmv_HLL_sequential(&hll, x);
	std::cout << std::endl << "-- FINISHED HLL SEQUENTIAL OPERATION --" << std::endl << std::endl;
	if (HLL_SEQ_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y5.size(); i++)
			std::cout << y5[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HLL
	std::cout << std::endl << "-- STARTING HLL KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y6 = spmv_HLL(&hll, x);
	std::cout << std::endl << "-- FINISHED HLL KERNEL OPERATION --" << std::endl << std::endl;
	if (HLL_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y6.size(); i++)
			std::cout << y6[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HLL_LOCAL
	std::cout << std::endl << "-- STARTING HLL_LOCAL KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y7 = spmv_HLL_LOCAL(&hll, x);
	std::cout << std::endl << "-- FINISHED HLL_LOCAL KERNEL OPERATION --" << std::endl << std::endl;
	if (HLL_LOCAL_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y7.size(); i++)
			std::cout << y7[i] << " ";
		std::cout << std::endl;
	}
#endif

	x.clear();
#if ELL_SEQ || ELL || ELLG_SEQ || ELLG
	FreeELLG(&ellg);
#if ELL_SEQ
	y1.clear();
#endif
#if ELL
	y2.clear();
#endif
#if ELLG_SEQ
	y3.clear();
#endif
#if ELLG
	y4.clear();
#endif
#endif
#if HLL_SEQ || HLL || HLL_LOCAL
	FreeHLL(&hll);
#if HLL_SEQ
	y5.clear();
#endif
#if HLL
	y6.clear();
#endif
#if HLL_LOCAL
	y7.clear();
#endif
#endif
#if DEBUG
	system("PAUSE"); // for debugging
#endif

	return 0;
}

#endif