#include<compiler_config.h>

#if CSR

#include<stdio.h>
#include<string>
#include<math.h> 
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

std::vector<CL_REAL> spmv_CSR(const struct csr_t* d_csr, const std::vector<CL_REAL> d_x)
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
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + CSR_KERNEL_FILE, context, device);
#if PRECISION == 2
	cl::Kernel kernel{ program, "spmv_csr_d" };
#else
	cl::Kernel kernel{ program, "spmv_csr_s" };
#endif
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
	kernel.setArg(0, d_csr->n);
	kernel.setArg(1, repeat);
	kernel.setArg(2, coop);
	kernel.setArg(3, d_ia_buffer);
	kernel.setArg(4, d_ja_buffer);
	kernel.setArg(5, d_a_buffer);
	kernel.setArg(6, d_x_buffer);
	kernel.setArg(7, dst_y_buffer);
	kernel.setArg(8, cl::Local(local_byte_size_shdata));
	//
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	std::cout << "!!! CSR kernel: repeat = " << repeat << ", coop = " << coop << ", nworkgroups = " << nworkgroups << " !!!" << std::endl << std::endl;
	std::cout << "!!! A work-group uses " << local_byte_size_shdata << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
	for (int r = 0; r < REPEAT; r++)
	{
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_d_x, dst_y.data());
		nanoseconds =
			jc::run_and_time_kernel(kernel,
				queue,
				cl::NDRange(1500 * CSR_WORKGROUP_SIZE),
				cl::NDRange(CSR_WORKGROUP_SIZE));
		std::cout << "Run: " << r + 1 << " | Time elapsed: " << nanoseconds << " ns | Effective throughput: " << 2 * (d_csr->nnz) / (nanoseconds * 1e-9) / 1e9 << "GFLOP/s\n";
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	std::cout << std::endl << "Average time: " << average_nanoseconds << " ns | Average effective throughput: " << 2 * (d_csr->nnz) / (average_nanoseconds * 1e-9) / 1e9 << "GFLOP/s\n";
	//increment all values
	for (IndexType i = 0; i < d_csr->n + 1; i++) d_csr->ia[i]++;
	for (IndexType i = 0; i < d_csr->nnz; i++) d_csr->ja[i]++;

	return dst_y;
}

int main(void)
{
	// Error checking
	// TODO?

	FILE* f;
	struct coo_t coo;
	struct csr_t csr;
	struct jad_t jad;

	std::string input_filename = (INPUT_FOLDER + (std::string)"/" + INPUT_FILE);

	if (createOutputDirectory(OUTPUT_FOLDER, CSR_OUTPUT_FOLDER))
		exit(1);
	std::string output_file = (OUTPUT_FOLDER + (std::string)"/" + CSR_OUTPUT_FOLDER + (std::string)"/" + OUTPUT_FILENAME + getTimeOfRun() + OUTPUT_FILEFORMAT);

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
	FreeCOO(&coo);
	std::cout << "-- DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;

	std::vector<CL_REAL> x = std::vector<CL_REAL>();
	for (IndexType i = 0; i < n; i++)
		x.push_back(i);

	std::cout << std::endl << "-- STARTING CSR KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y = spmv_CSR(&csr, x);
	std::cout << std::endl << "-- FINISHED CSR KERNEL OPERATION --" << std::endl << std::endl;
	if (CSR_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y.size(); i++)
			std::cout << y[i] << " ";
		std::cout << std::endl;
	}

	
	x.clear(); 
	FreeCSR(&csr);
	y.clear();
#if DEBUG
	system("PAUSE"); // for debugging
#endif

	return 0;
}

#endif