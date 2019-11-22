#include<compiler_config.h>

#if COO

#include<stdio.h>
#include<string>
#include<windows.h>

#include<util_time.hpp>
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

#define DIVIDE_INTO(x,y) ((x + y - 1)/y)

// Implementation based off the spmv_coo_flat_device.cu.h from sc2009_spmv by: Nathan Bell & Michael Garland
// URL: https://code.google.com/archive/p/cusp-library/downloads
//
std::vector<CL_REAL> spmv_COO_serial(const struct coo_t* d_coo, const std::vector<CL_REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_coo->nnz; i++) d_coo->ir[i]--, d_coo->jc[i]--;
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + COO_KERNEL_FILE, context, device);
#if PRECISION == 2
	cl::Kernel kernel{ program, "spmv_coo_serial_d" };
#else
	cl::Kernel kernel{ program, "spmv_coo_serial_s" };
#endif
	//
	size_t byte_size_d_ir = d_coo->nnz * sizeof(cl_uint);
	size_t byte_size_d_jc = d_coo->nnz * sizeof(cl_uint);
	size_t byte_size_d_val = d_coo->nnz * sizeof(CL_REAL);
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_ir_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ir };
	cl::Buffer d_jc_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jc };
	cl::Buffer d_val_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_val };
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	queue.enqueueWriteBuffer(d_ir_buffer, CL_TRUE, 0, byte_size_d_ir, d_coo->ir);
	queue.enqueueWriteBuffer(d_jc_buffer, CL_TRUE, 0, byte_size_d_jc, d_coo->jc);
	queue.enqueueWriteBuffer(d_val_buffer, CL_TRUE, 0, byte_size_d_val, d_coo->val);
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	kernel.setArg(0, d_coo->nnz);
	kernel.setArg(1, 0);
	kernel.setArg(2, d_ir_buffer);
	kernel.setArg(3, d_jc_buffer);
	kernel.setArg(4, d_val_buffer);
	kernel.setArg(5, d_x_buffer);
	kernel.setArg(6, dst_y_buffer);
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
				cl::NDRange(1),
				cl::NDRange(1));
		std::cout << "Run: " << r + 1 << " | Time elapsed: " << nanoseconds << " ns\n";
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	std::cout << std::endl << "Average time: " << average_nanoseconds << " ns\n";
	//increment all values
	for (IndexType i = 0; i < d_coo->nnz; i++) d_coo->ir[i]++, d_coo->jc[i]++;

	return dst_y;
}

std::vector<CL_REAL> spmv_COO_flat(const struct coo_t* d_coo, const std::vector<CL_REAL> d_x)
{
	// decrement all values
	for (IndexType i = 0; i < d_coo->nnz; i++) d_coo->ir[i]--, d_coo->jc[i]--;
	//
	// determine specific values
	unsigned IndexType max_workgroups = MAX_THREADS / (2 * WORKGROUP_SIZE);
	//
	unsigned IndexType nunits = d_coo->nnz / WARP_SIZE;
	unsigned IndexType nwarps = min(nunits, WARPS_PER_WORKGROUP * max_workgroups);
	unsigned IndexType nworkgroups = DIVIDE_INTO(nwarps, WARPS_PER_WORKGROUP);
	unsigned IndexType niters = DIVIDE_INTO(nunits, nwarps);
	//
	unsigned IndexType interval_size = WARP_SIZE * niters;
	//
	IndexType tail = nunits * WARP_SIZE; // do the last few nonzeros separately (fewer than WARP_SIZE elements)
	//
	unsigned IndexType active_warps = (interval_size == 0) ? 0 : DIVIDE_INTO(tail, interval_size);
	//
	// allocate vectors
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	std::vector<cl_uint> temp_rows(active_warps, 0);
	std::vector<CL_REAL> temp_vals(active_warps, 0);
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	cl::Program program =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + COO_KERNEL_FILE, context, device);
#if PRECISION == 2
	cl::Kernel kernel_flat{ program, "spmv_coo_flat_d" };
	cl::Kernel kernel_serial{ program, "spmv_coo_serial_d" };
	cl::Kernel kernel_reduce_update{ program, "spmv_coo_reduce_update_d" };
#else
	cl::Kernel kernel_flat{ program, "spmv_coo_flat_s" };
	cl::Kernel kernel_serial{ program, "spmv_coo_serial_s" };
	cl::Kernel kernel_reduce_update{ program, "spmv_coo_reduce_update_s" };
#endif

	//
	size_t byte_size_d_ir = d_coo->nnz * sizeof(cl_uint);
	size_t byte_size_d_jc = d_coo->nnz * sizeof(cl_uint);
	size_t byte_size_d_val = d_coo->nnz * sizeof(CL_REAL);
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	size_t byte_size_temp_rows = temp_rows.size() * sizeof(CL_REAL);
	size_t byte_size_temp_vals = temp_vals.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_ir_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ir };
	cl::Buffer d_jc_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jc };
	cl::Buffer d_val_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_val };
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	cl::Buffer temp_rows_buffer{ context, CL_MEM_READ_WRITE, byte_size_temp_rows };
	cl::Buffer temp_vals_buffer{ context, CL_MEM_READ_WRITE, byte_size_temp_vals };
	//
	cl_ulong size;
	device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
	//
	// Step 1
	size_t local_byte_size_shrows_1 = WORKGROUP_SIZE * sizeof(cl_uint);
	size_t local_byte_size_shvals_1 = WORKGROUP_SIZE * sizeof(CL_REAL);
	// Step 2
	size_t local_byte_size_shrows_2 = (__WORKGROUP_SIZE + 1) * sizeof(cl_uint);
	size_t local_byte_size_shvals_2 = (__WORKGROUP_SIZE + 1) * sizeof(CL_REAL);
	//
	queue.enqueueWriteBuffer(d_ir_buffer, CL_TRUE, 0, byte_size_d_ir, d_coo->ir);
	queue.enqueueWriteBuffer(d_jc_buffer, CL_TRUE, 0, byte_size_d_jc, d_coo->jc);
	queue.enqueueWriteBuffer(d_val_buffer, CL_TRUE, 0, byte_size_d_val, d_coo->val);
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	queue.enqueueWriteBuffer(temp_rows_buffer, CL_TRUE, 0, byte_size_temp_rows, temp_rows.data());
	queue.enqueueWriteBuffer(temp_vals_buffer, CL_TRUE, 0, byte_size_temp_vals, temp_vals.data());
	//
	kernel_flat.setArg(0, tail);
	kernel_flat.setArg(1, interval_size);
	kernel_flat.setArg(2, d_ir_buffer);
	kernel_flat.setArg(3, d_jc_buffer);
	kernel_flat.setArg(4, d_val_buffer);
	kernel_flat.setArg(5, d_x_buffer);
	kernel_flat.setArg(6, dst_y_buffer);
	kernel_flat.setArg(7, temp_rows_buffer);
	kernel_flat.setArg(8, temp_vals_buffer);
	kernel_flat.setArg(9, cl::Local(local_byte_size_shrows_1));
	kernel_flat.setArg(10, cl::Local(local_byte_size_shvals_1));
	//
	kernel_serial.setArg(0, d_coo->nnz - tail);
	kernel_serial.setArg(1, tail);
	kernel_serial.setArg(2, d_ir_buffer);
	kernel_serial.setArg(3, d_jc_buffer);
	kernel_serial.setArg(4, d_val_buffer);
	kernel_serial.setArg(5, d_x_buffer);
	kernel_serial.setArg(6, dst_y_buffer);
	//
	kernel_reduce_update.setArg(0, active_warps);
	kernel_reduce_update.setArg(1, temp_rows_buffer);
	kernel_reduce_update.setArg(2, temp_vals_buffer);
	kernel_reduce_update.setArg(3, dst_y_buffer);
	kernel_reduce_update.setArg(4, cl::Local(local_byte_size_shrows_2));
	kernel_reduce_update.setArg(5, cl::Local(local_byte_size_shvals_2));
	//
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	std::cout << "!!! kernel_flat: A work-group uses " << local_byte_size_shrows_1 + local_byte_size_shvals_1 << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
	std::cout << "!!! kernel_reduce_update: A work-group uses " << local_byte_size_shrows_2 + local_byte_size_shvals_2 << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
	for (int r = 0; r < REPEAT; r++)
	{
		queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_d_x, dst_y.data());
		queue.enqueueWriteBuffer(temp_rows_buffer, CL_TRUE, 0, byte_size_temp_rows, temp_rows.data());
		queue.enqueueWriteBuffer(temp_vals_buffer, CL_TRUE, 0, byte_size_temp_vals, temp_vals.data());
		nanoseconds =
			jc::run_and_time_kernel(kernel_flat,
				queue,
				cl::NDRange(nworkgroups * WORKGROUP_SIZE),
				cl::NDRange(WORKGROUP_SIZE));
		if (d_coo->nnz - tail > 0)
			nanoseconds +=
				jc::run_and_time_kernel(kernel_serial,
					queue,
					cl::NDRange(1),
					cl::NDRange(1));
		nanoseconds +=
			jc::run_and_time_kernel(kernel_reduce_update,
				queue,
				cl::NDRange(__WORKGROUP_SIZE),
				cl::NDRange(__WORKGROUP_SIZE));
		std::cout << "Run: " << r + 1 << " | Time elapsed: " << nanoseconds << " ns\n";
		total_nanoseconds += nanoseconds;
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	std::cout << std::endl << "Average time: " << average_nanoseconds << " ns\n";
	// increment all values
	for (IndexType i = 0; i < d_coo->nnz; i++) d_coo->ir[i]++, d_coo->jc[i]++;

	return dst_y;
}

int main(void)
{
	// Error checking
	// TODO?

	FILE* f;
	struct coo_t coo;

	std::string input_filename = (INPUT_FOLDER + (std::string)"/" + INPUT_FILE);

	std::string output_file = (OUTPUT_FOLDER + (std::string)"/" + COO_OUTPUT_FOLDER + (std::string)"/" + OUTPUT_FILENAME + getTimeOfRun() + OUTPUT_FILEFORMAT);

	std::cout << "!!! OUTPUT IS BEING WRITTEN TO " << output_file << " !!!" << std::endl;
	std::cout << "!!! PROGRAM WILL EXIT AUTOMATICALLY AFTER PROCESSING; PRESS CTRL-C TO ABORT !!!" << std::endl;
	system("PAUSE");
	freopen_s(&f, output_file.c_str(), "w", stdout);

	std::cout << "-- LOADING INPUT FILE " << input_filename << " --" << std::endl;
	MM_To_COO(input_filename.c_str(), &coo, COO_LOG);
	std::cout << "-- INPUT FILE LOADED --" << std::endl << std::endl;

	std::vector<CL_REAL> x = std::vector<CL_REAL>();
	for (IndexType i = 0; i < coo.n; i++)
		x.push_back(i);

	std::cout << std::endl << "-- STARTING COO KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y;
	if (coo.nnz < WARP_SIZE)
		y = spmv_COO_serial(&coo, x);
	else
		y = spmv_COO_serial(&coo, x);
	std::cout << std::endl << "-- FINISHED COO KERNEL OPERATION --" << std::endl << std::endl;
	if (COO_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y.size(); i++)
			std::cout << y[i] << " ";
		std::cout << std::endl;
	}

	x.clear();
	FreeCOO(&coo);
	y.clear();
#if DEBUG
	system("PAUSE"); // for debugging
#endif

	return 0;
}

#endif