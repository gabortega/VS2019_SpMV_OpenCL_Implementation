#include<compiler_config.h>

#if JAD_SEQ || JAD

#include<stdio.h>
#include<string>
#include<windows.h>

#include<util_misc.hpp>
#include<CL/cl.h>
#include<JC/util.hpp>
#include<IO/mmio.h>
#include<IO/convert_input.h>
#include<SEQ/JAD.hpp>

#if PRECISION == 2
#define CL_REAL cl_double
#else
//#elif PRECISION == 1
#define CL_REAL cl_float
//#else
//#define CL_REAL cl_half // TODO?
#endif

#if JAD_SEQ
std::vector<REAL> spmv_JAD_sequential(struct jad_t* d_jad, const std::vector<REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < (d_jad->njad[d_jad->n] + 1); i++) d_jad->ia[i]--;
	for (IndexType i = 0; i < d_jad->total; i++) d_jad->ja[i]--;
	//
	std::vector<REAL> dst_y(d_x.size(), 0);
	//jad->a + d_x + dst_y
	unsigned long long units_REAL = d_jad->nnz + d_jad->nnz + d_jad->nnz;
	//jad->ia + jad->ja + jad->njad + jad->perm
	unsigned long long units_IndexType = d_jad->njad[d_jad->n] + d_jad->nnz + 2 * d_jad->njad[d_jad->n] * d_jad->n + d_jad->nnz;
	//
	unsigned long nanoseconds = 0, total_nanoseconds = 0;
	//
	for (int r = 0; r < REPEAT; r++)
	{
		std::fill(dst_y.begin(), dst_y.end(), 0);
		nanoseconds = JAD_sequential(d_jad, d_x, dst_y);
		printRunInfo(r + 1, nanoseconds, (d_jad->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_jad->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < (d_jad->njad[d_jad->n] + 1); i++) d_jad->ia[i]++;
	for (IndexType i = 0; i < d_jad->total; i++) d_jad->ja[i]++;

	return dst_y;
}
#endif

#if JAD
std::vector<CL_REAL> spmv_JAD(struct jad_t* d_jad, const std::vector<CL_REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < (d_jad->njad[d_jad->n] + 1); i++) d_jad->ia[i]--;
	for (IndexType i = 0; i < d_jad->total; i++) d_jad->ja[i]--;
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	unsigned long long for_iters = (d_jad->njad[d_jad->n] + MAX_NJAD_PER_WG - 1) / MAX_NJAD_PER_WG;
	//jad->a + d_x + dst_y
	unsigned long long units_REAL = d_jad->nnz + d_jad->nnz + d_jad->n * for_iters;
	//jad->ia + jad->ja + jad->njad + jad->perm
	unsigned long long units_IndexType = ((d_jad->njad[d_jad->n] + 1) * (jc::best_fit(d_jad->n, WORKGROUP_SIZE) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) + d_jad->nnz + 3 * d_jad->n * for_iters;
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	//Macro
	std::string macro = "-DPRECISION=" + std::to_string(PRECISION) +
						" -DN_MATRIX=" + std::to_string(d_jad->n) +
						" -DUNROLL_SHARED=" + std::to_string(((WORKGROUP_SIZE + MAX_NJAD_PER_WG - 1)/ MAX_NJAD_PER_WG) + 1) +
						" -DWORKGROUP_SIZE=" + std::to_string(WORKGROUP_SIZE);
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + JAD_KERNEL_FILE, context, device, macro.c_str());
	cl::Kernel kernel{ program, "spmv_jad" };
	//
	std::cout << "Kernel macros: " << macro << std::endl << std::endl;
	//
	size_t byte_size_d_njad = d_jad->n * sizeof(cl_uint);
	size_t byte_size_d_ia = (d_jad->njad[d_jad->n] + 1) * sizeof(cl_uint);
	size_t byte_size_d_ja = d_jad->total * sizeof(cl_uint);
	size_t byte_size_d_a = d_jad->total * sizeof(CL_REAL);
	size_t byte_size_d_perm = d_jad->n * sizeof(cl_uint);
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_njad_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_njad };
	cl::Buffer d_ia_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ia };
	cl::Buffer d_ja_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ja };
	cl::Buffer d_a_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_a };
	cl::Buffer d_perm_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_perm };
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	cl_ulong size;
	device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
	//
	size_t local_byte_size_shia = MAX_NJAD_PER_WG * sizeof(cl_uint);
	//
	queue.enqueueWriteBuffer(d_njad_buffer, CL_TRUE, 0, byte_size_d_njad, d_jad->njad);
	queue.enqueueWriteBuffer(d_ia_buffer, CL_TRUE, 0, byte_size_d_ia, d_jad->ia);
	queue.enqueueWriteBuffer(d_ja_buffer, CL_TRUE, 0, byte_size_d_ja, d_jad->ja);
	queue.enqueueWriteBuffer(d_a_buffer, CL_TRUE, 0, byte_size_d_a, d_jad->a);
	queue.enqueueWriteBuffer(d_perm_buffer, CL_TRUE, 0, byte_size_d_perm, d_jad->perm);
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	kernel.setArg(1, d_njad_buffer);
	kernel.setArg(2, d_ia_buffer);
	kernel.setArg(3, d_ja_buffer);
	kernel.setArg(4, d_a_buffer);
	kernel.setArg(5, d_perm_buffer);
	kernel.setArg(6, d_x_buffer);
	kernel.setArg(7, dst_y_buffer);
	kernel.setArg(8, cl::Local(local_byte_size_shia));
	//
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	std::cout << "!!! A work-group uses " << local_byte_size_shia << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
	for (int r = 0; r < REPEAT; r++)
	{
		nanoseconds = 0;
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_d_x, dst_y.data());
		for (IndexType i = 0; i < *(d_jad->njad + d_jad->n); i += MAX_NJAD_PER_WG)
		{
			kernel.setArg(0, min(*(d_jad->njad + d_jad->n) - i, MAX_NJAD_PER_WG)); // set njad for this iteration
			kernel.setArg(9, i);
			nanoseconds +=
				jc::run_and_time_kernel(kernel,
					queue,
					cl::NDRange(jc::best_fit(d_jad->n, WORKGROUP_SIZE)),
					cl::NDRange(WORKGROUP_SIZE));
		}
		printRunInfo(r + 1, nanoseconds, (d_jad->nnz), units_REAL, units_IndexType);
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	printAverageRunInfo(average_nanoseconds, (d_jad->nnz), units_REAL, units_IndexType);
	//increment all values
	for (IndexType i = 0; i < (d_jad->njad[d_jad->n] + 1); i++) d_jad->ia[i]++;
	for (IndexType i = 0; i < d_jad->total; i++) d_jad->ja[i]++;

	return dst_y;
}
#endif

int main(void)
{
	// Error checking
	if (WORKGROUP_SIZE > MAX_NJAD_PER_WG) 
	{
		std::cout << "!!! ERROR: WORKGROUP_SIZE CANNOT BE GREATER THAN MAX_NJAD_PER_WG !!!" << std::endl;
		system("PAUSE");
		exit(1);
	}

	FILE* f;
	struct coo_t coo;
	struct csr_t csr;
	struct jad_t jad;

	std::string input_filename = (INPUT_FOLDER + (std::string)"/" + INPUT_FILE);

	if (createOutputDirectory(OUTPUT_FOLDER, JAD_OUTPUT_FOLDER))
		exit(1);
	std::string output_file = (OUTPUT_FOLDER + (std::string)"/" + JAD_OUTPUT_FOLDER + (std::string)"/" + OUTPUT_FILENAME + getTimeOfRun() + OUTPUT_FILEFORMAT);

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
	CSR_To_JAD(&csr, &jad, JAD_LOG);
	FreeCOO(&coo);
	FreeCSR(&csr);
	std::cout << "-- DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;

	std::vector<CL_REAL> x = std::vector<CL_REAL>();
	for (IndexType i = 0; i < n; i++)
		x.push_back(i);

#if JAD_SEQ
	std::cout << std::endl << "-- STARTING JAD SEQUENTIAL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y1 = spmv_JAD_sequential(&jad, x);
	std::cout << std::endl << "-- FINISHED JAD SEQUENTIAL OPERATION --" << std::endl << std::endl;
	if (JAD_SEQ_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y1.size(); i++)
			std::cout << y1[i] << " ";
		std::cout << std::endl;
	}
#endif
#if JAD
	std::cout << std::endl << "-- STARTING JAD KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y2 = spmv_JAD(&jad, x);
	std::cout << std::endl << "-- FINISHED JAD KERNEL OPERATION --" << std::endl << std::endl;
	if (JAD_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y2.size(); i++)
			std::cout << y2[i] << " ";
		std::cout << std::endl;
	}
#endif

	x.clear();
	FreeJAD(&jad);
#if JAD_SEQ
	y1.clear();
#endif
#if JAD
	y2.clear();
#endif
#if DEBUG
	system("PAUSE"); // for debugging
#endif

	return 0;
}

#endif