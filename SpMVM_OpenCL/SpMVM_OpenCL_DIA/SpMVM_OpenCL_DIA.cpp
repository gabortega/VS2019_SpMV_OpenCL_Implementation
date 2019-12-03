#include<compiler_config.h>

#if DIA || HDIA || HDIA_LOCAL

#include<stdio.h>
#include<string>
#include<windows.h>

#include<util_misc.hpp>
#include<CL/cl.h>
#include<JC/util.hpp>
#include<IO/mmio.h>
#include<IO/convert_input.h>

#if PRECISION == 2
#define CL_REAL cl_double
#else
//#elif PRECISION == 1
#define CL_REAL cl_float
//#else
//#define CL_REAL cl_half // TODO?
#endif

#if DIA
std::vector<CL_REAL> spmv_DIA(const struct dia_t* d_dia, const std::vector<CL_REAL> d_x)
{
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + DIA_KERNEL_FILE, context, device);
#if PRECISION == 2
	cl::Kernel kernel{ program, "spmv_dia_d" };
#else
	cl::Kernel kernel{ program, "spmv_dia_s" };
#endif
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
	size_t local_byte_size_shia = MAX_NDIAG_PER_WG * sizeof(cl_uint);
	//
	queue.enqueueWriteBuffer(d_ioff_buffer, CL_TRUE, 0, byte_size_d_ioff, d_dia->ioff);
	queue.enqueueWriteBuffer(d_diags_buffer, CL_TRUE, 0, byte_size_d_diags, d_dia->diags);
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	kernel.setArg(0, d_dia->n);
	kernel.setArg(2, d_dia->stride);
	kernel.setArg(3, d_ioff_buffer);
	kernel.setArg(4, d_diags_buffer);
	kernel.setArg(5, d_x_buffer);
	kernel.setArg(6, dst_y_buffer);
	kernel.setArg(7, cl::Local(local_byte_size_shia));
	//
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	std::cout << "!!! A work-group uses " << local_byte_size_shia << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
	for (int r = 0; r < REPEAT; r++)
	{
		nanoseconds = 0;
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_d_x, dst_y.data());
		for (IndexType i = 0; i < d_dia->ndiags; i += MAX_NDIAG_PER_WG)
		{
			kernel.setArg(1, min(d_dia->ndiags - i, MAX_NDIAG_PER_WG)); // set njad for this iteration
			kernel.setArg(8, i);
			kernel.setArg(9, i * d_dia->stride);
			nanoseconds +=
				jc::run_and_time_kernel(kernel,
					queue,
					cl::NDRange(min(MAX_THREADS, jc::best_fit(d_dia->n, WORKGROUP_SIZE))),
					cl::NDRange(WORKGROUP_SIZE));
		}
		std::cout << "Run: " << r + 1 << " | Time elapsed: " << nanoseconds << " ns | Effective throughput: " << 2 * (d_dia->nnz) / (nanoseconds * 1e-9) / 1e9 << "GFLOP/s\n";
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	std::cout << std::endl << "Average time: " << average_nanoseconds << " ns | Average effective throughput: " << 2 * (d_dia->nnz) / (average_nanoseconds * 1e-9) / 1e9 << "GFLOP/s\n";

	return dst_y;
}
#endif

#if HDIA
std::vector<CL_REAL> spmv_HDIA(const struct hdia_t* d_hdia, const std::vector<CL_REAL> d_x)
{
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + HDIA_KERNEL_FILE, context, device);
#if PRECISION == 2
	cl::Kernel kernel{ program, "spmv_hdia_d" };
#else
	cl::Kernel kernel{ program, "spmv_hdia_s" };
#endif
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
	kernel.setArg(0, d_hdia->n);
	kernel.setArg(1, d_hdia->nhoff);
	kernel.setArg(2, HDIA_HACKSIZE);
	kernel.setArg(3, d_ndiags_buffer);
	kernel.setArg(4, d_ioff_buffer);
	kernel.setArg(5, d_diags_buffer);
	kernel.setArg(6, d_hoff_buffer);
	kernel.setArg(7, d_memoff_buffer);
	kernel.setArg(8, d_x_buffer);
	kernel.setArg(9, dst_y_buffer);
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
				cl::NDRange(min(MAX_THREADS, jc::best_fit(d_hdia->n, WORKGROUP_SIZE))),
				cl::NDRange(WORKGROUP_SIZE));
		std::cout << "Run: " << r + 1 << " | Time elapsed: " << nanoseconds << " ns | Effective throughput: " << 2 * (d_hdia->nnz) / (nanoseconds * 1e-9) / 1e9 << "GFLOP/s\n";
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	std::cout << std::endl << "Average time: " << average_nanoseconds << " ns | Average effective throughput: " << 2 * (d_hdia->nnz) / (average_nanoseconds * 1e-9) / 1e9 << "GFLOP/s\n";

	return dst_y;
}
#endif

#if HDIA_LOCAL
std::vector<CL_REAL> spmv_HDIA_LOCAL(const struct hdia_t* d_hdia, const std::vector<CL_REAL> d_x)
{
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + HDIA_LOCAL_KERNEL_FILE, context, device);
#if PRECISION == 2
	cl::Kernel kernel{ program, "spmv_hdia_local_d" };
#else
	cl::Kernel kernel{ program, "spmv_hdia_local_s" };
#endif
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
	size_t local_byte_size_shhoff = WORKGROUP_SIZE / HDIA_HACKSIZE * 3 * sizeof(cl_uint);
	//
	queue.enqueueWriteBuffer(d_ndiags_buffer, CL_TRUE, 0, byte_size_d_ndiags, d_hdia->ndiags);
	queue.enqueueWriteBuffer(d_ioff_buffer, CL_TRUE, 0, byte_size_d_ioff, d_hdia->ioff);
	queue.enqueueWriteBuffer(d_diags_buffer, CL_TRUE, 0, byte_size_d_diags, d_hdia->diags);
	queue.enqueueWriteBuffer(d_hoff_buffer, CL_TRUE, 0, byte_size_d_hoff, d_hdia->hoff);
	queue.enqueueWriteBuffer(d_memoff_buffer, CL_TRUE, 0, byte_size_d_hoff, d_hdia->memoff);
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	kernel.setArg(0, d_hdia->n);
	kernel.setArg(1, d_hdia->nhoff);
	kernel.setArg(2, HDIA_HACKSIZE);
	kernel.setArg(3, d_ndiags_buffer);
	kernel.setArg(4, d_ioff_buffer);
	kernel.setArg(5, d_diags_buffer);
	kernel.setArg(6, d_hoff_buffer);
	kernel.setArg(7, d_memoff_buffer);
	kernel.setArg(8, d_x_buffer);
	kernel.setArg(9, dst_y_buffer);
	kernel.setArg(10, cl::Local(local_byte_size_shhoff));
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
				cl::NDRange(min(MAX_THREADS, jc::best_fit(d_hdia->n, WORKGROUP_SIZE))),
				cl::NDRange(WORKGROUP_SIZE));
		std::cout << "Run: " << r + 1 << " | Time elapsed: " << nanoseconds << " ns | Effective throughput: " << 2 * (d_hdia->nnz) / (nanoseconds * 1e-9) / 1e9 << "GFLOP/s\n";
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	std::cout << std::endl << "Average time: " << average_nanoseconds << " ns | Average effective throughput: " << 2 * (d_hdia->nnz) / (average_nanoseconds * 1e-9) / 1e9 << "GFLOP/s\n";

	return dst_y;
}
#endif

int main(void)
{
	// Error checking
	if (WORKGROUP_SIZE > MAX_NDIAG_PER_WG) 
	{
		std::cout << "!!! ERROR: WORKGROUP_SIZE CANNOT BE GREATER THAN MAX_NDIAG_PER_WG !!!" << std::endl;
		system("PAUSE");
		exit(1);
	}
#if HDIA_LOCAL
	if ((WORKGROUP_SIZE / HDIA_HACKSIZE) * HDIA_HACKSIZE != WORKGROUP_SIZE)
	{
		std::cout << "!!! ERROR: WORKGROUP_SIZE MUST BE A MULTIPLE OF HDIA_HACKSIZE !!!" << std::endl;
		system("PAUSE");
		exit(1);
	}
#endif

	FILE* f;
	struct coo_t coo;
	struct csr_t csr;
#if DIA
	struct dia_t dia;
#endif
#if HDIA || HDIA_LOCAL
	struct hdia_t hdia;
#endif

	std::string input_filename = (INPUT_FOLDER + (std::string)"/" + INPUT_FILE);

	if (createOutputDirectory(OUTPUT_FOLDER, DIA_OUTPUT_FOLDER))
		exit(1);
	std::string output_file = (OUTPUT_FOLDER + (std::string)"/" + DIA_OUTPUT_FOLDER + (std::string)"/" + OUTPUT_FILENAME + getTimeOfRun() + OUTPUT_FILEFORMAT);

	std::cout << "!!! OUTPUT IS BEING WRITTEN TO " << output_file << " !!!" << std::endl;
	std::cout << "!!! PROGRAM WILL EXIT AUTOMATICALLY AFTER PROCESSING; PRESS CTRL-C TO ABORT !!!" << std::endl;
	system("PAUSE");
	freopen_s(&f, output_file.c_str(), "w", stdout);

	std::cout << "-- LOADING INPUT FILE " << input_filename << " --" << std::endl;
	MM_To_COO(input_filename.c_str(), &coo, COO_LOG);
	IndexType n = coo.n;
	std::cout << "-- INPUT FILE LOADED --" << std::endl << std::endl;
	std::cout << "-- PRE-PROCESSING INPUT --" << std::endl;
	COO_To_CSR(&coo, &csr, CSR_LOG);
#if DIA
	if (!CSR_To_DIA(&csr, &dia, DIA_LOG))
		std::cout << "DIA IS INCOMPLETE" << std::endl;
#endif
#if HDIA || HDIA_LOCAL
	if(!CSR_To_HDIA(&csr, &hdia, HDIA_LOG))
		std::cout << "HDIA IS INCOMPLETE" << std::endl;
#endif
	FreeCOO(&coo);
	FreeCSR(&csr);
	std::cout << "-- DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;

	std::vector<CL_REAL> x = std::vector<CL_REAL>();
	for (IndexType i = 0; i < n; i++)
		x.push_back(i);

#if DIA
	std::cout << std::endl << "-- STARTING DIA KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y1 = spmv_DIA(&dia, x);
	std::cout << std::endl << "-- FINISHED DIA KERNEL OPERATION --" << std::endl << std::endl;
	if (DIA_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y1.size(); i++)
			std::cout << y1[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HDIA
	std::cout << std::endl << "-- STARTING HDIA KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y2 = spmv_HDIA(&hdia, x);
	std::cout << std::endl << "-- FINISHED HDIA KERNEL OPERATION --" << std::endl << std::endl;
	if (HDIA_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y2.size(); i++)
			std::cout << y2[i] << " ";
	}
	std::cout << std::endl;
#endif
#if HDIA_LOCAL
	std::cout << std::endl << "-- STARTING HDIA_LOCAL KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y3 = spmv_HDIA_LOCAL(&hdia, x);
	std::cout << std::endl << "-- FINISHED HDIA_LOCAL KERNEL OPERATION --" << std::endl << std::endl;
	if (HDIA_LOCAL_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y3.size(); i++)
			std::cout << y3[i] << " ";
		std::cout << std::endl;
	}
#endif

	x.clear();
#if DIA
	FreeDIA(&dia);
	y1.clear();
#endif
#if HDIA || HDIA_LOCAL
	FreeHDIA(&hdia);
#if HDIA
	y2.clear();
#endif
#if HDIA_LOCAL
	y3.clear();
#endif
#endif
#if DEBUG
	system("PAUSE"); // for debugging
#endif

	return 0;
}

#endif