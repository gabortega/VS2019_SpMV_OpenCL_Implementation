#include<compiler_config.h>

#if HYB_ELL || HYB_ELLG || HYB_HLL || HYB_HLL_LOCAL

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

#define DIVIDE_INTO(x,y) ((x + y - 1)/y)

#if HYB_ELL
std::vector<CL_REAL> spmv_HYB_ELL(const struct hybellg_t* d_hyb, const std::vector<CL_REAL> d_x)
{	
	//decrement all values
	for (IndexType i = 0; i < d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n); i++) d_hyb->ellg.jcoeff[i]--;
	for (IndexType i = 0; i < d_hyb->coo.nnz; i++) d_hyb->coo.ir[i]--, d_hyb->coo.jc[i]--;
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	cl::Program program_ell =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + ELL_KERNEL_FILE, context, device);
	cl::Program program_coo =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + COO_KERNEL_FILE, context, device);
#if PRECISION == 2
	cl::Kernel kernel_ell{ program_ell, "spmv_ell_d" };
	cl::Kernel kernel_flat{ program_coo, "spmv_coo_flat_d" };
	cl::Kernel kernel_serial{ program_coo, "spmv_coo_serial_d" };
	cl::Kernel kernel_reduce_update{ program_coo, "spmv_coo_reduce_update_d" };
#else
	cl::Kernel kernel_ell{ program_ell, "spmv_ell_s" };
	cl::Kernel kernel_flat{ program_coo, "spmv_coo_flat_s" };
	cl::Kernel kernel_serial{ program_coo, "spmv_coo_serial_s" };
	cl::Kernel kernel_reduce_update{ program_coo, "spmv_coo_reduce_update_s" };
#endif
	//
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	if (d_hyb->ellg.nnz > 0)
	{
		size_t byte_size_d_jcoeff = d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n) * sizeof(cl_uint);
		size_t byte_size_d_a = d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n) * sizeof(CL_REAL);
		//
		cl::Buffer d_jcoeff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jcoeff };
		cl::Buffer d_a_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_a };
		//
		queue.enqueueWriteBuffer(d_jcoeff_buffer, CL_TRUE, 0, byte_size_d_jcoeff, d_hyb->ellg.jcoeff);
		queue.enqueueWriteBuffer(d_a_buffer, CL_TRUE, 0, byte_size_d_a, d_hyb->ellg.a);
		//
		kernel_ell.setArg(0, d_hyb->ellg.n);
		kernel_ell.setArg(1, *(d_hyb->ellg.nell + d_hyb->ellg.n));
		kernel_ell.setArg(2, d_hyb->ellg.stride);
		kernel_ell.setArg(3, d_jcoeff_buffer);
		kernel_ell.setArg(4, d_a_buffer);
		kernel_ell.setArg(5, d_x_buffer);
		kernel_ell.setArg(6, dst_y_buffer);
	}
	//
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	if (true) // if (d_hyb->coo.nnz < WARP_SIZE)
	{
		size_t byte_size_d_ir = d_hyb->coo.nnz * sizeof(cl_uint);
		size_t byte_size_d_jc = d_hyb->coo.nnz * sizeof(cl_uint);
		size_t byte_size_d_val = d_hyb->coo.nnz * sizeof(CL_REAL);
		//
		cl::Buffer d_ir_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ir };
		cl::Buffer d_jc_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jc };
		cl::Buffer d_val_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_val };
		//
		queue.enqueueWriteBuffer(d_ir_buffer, CL_TRUE, 0, byte_size_d_ir, d_hyb->coo.ir);
		queue.enqueueWriteBuffer(d_jc_buffer, CL_TRUE, 0, byte_size_d_jc, d_hyb->coo.jc);
		queue.enqueueWriteBuffer(d_val_buffer, CL_TRUE, 0, byte_size_d_val, d_hyb->coo.val);
		//
		kernel_serial.setArg(0, d_hyb->coo.nnz);
		kernel_serial.setArg(1, 0);
		kernel_serial.setArg(2, d_ir_buffer);
		kernel_serial.setArg(3, d_jc_buffer);
		kernel_serial.setArg(4, d_val_buffer);
		kernel_serial.setArg(5, d_x_buffer);
		kernel_serial.setArg(6, dst_y_buffer);
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
						cl::NDRange(min(MAX_THREADS, jc::best_fit(d_hyb->ellg.n, WORKGROUP_SIZE))),
						cl::NDRange(WORKGROUP_SIZE));
			}
			nanoseconds +=
				jc::run_and_time_kernel(kernel_serial,
					queue,
					cl::NDRange(1),
					cl::NDRange(1));
			std::cout << "Run: " << r + 1 << " | Time elapsed: " << nanoseconds << " ns\n";
			total_nanoseconds += nanoseconds;
		}
	}
	else
	{
		// determine specific values
		unsigned IndexType max_workgroups = MAX_THREADS / (2 * WORKGROUP_SIZE);
		//
		unsigned IndexType nunits = d_hyb->coo.nnz / WARP_SIZE;
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
		std::vector<cl_uint> temp_rows(active_warps, 0);
		std::vector<CL_REAL> temp_vals(active_warps, 0);
		//
		size_t byte_size_d_ir = d_hyb->coo.nnz * sizeof(cl_uint);
		size_t byte_size_d_jc = d_hyb->coo.nnz * sizeof(cl_uint);
		size_t byte_size_d_val = d_hyb->coo.nnz * sizeof(CL_REAL);
		size_t byte_size_temp_rows = temp_rows.size() * sizeof(CL_REAL);
		size_t byte_size_temp_vals = temp_vals.size() * sizeof(CL_REAL);
		//
		cl::Buffer d_ir_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ir };
		cl::Buffer d_jc_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jc };
		cl::Buffer d_val_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_val };
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
		queue.enqueueWriteBuffer(d_ir_buffer, CL_TRUE, 0, byte_size_d_ir, d_hyb->coo.ir);
		queue.enqueueWriteBuffer(d_jc_buffer, CL_TRUE, 0, byte_size_d_jc, d_hyb->coo.jc);
		queue.enqueueWriteBuffer(d_val_buffer, CL_TRUE, 0, byte_size_d_val, d_hyb->coo.val);
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
		kernel_serial.setArg(0, d_hyb->coo.nnz - tail);
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
		std::cout << "!!! COO: kernel_flat: A work-group uses " << local_byte_size_shrows_1 + local_byte_size_shvals_1 << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
		std::cout << "!!! COO: kernel_reduce_update: A work-group uses " << local_byte_size_shrows_2 + local_byte_size_shvals_2 << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
		for (int r = 0; r < REPEAT; r++)
		{
			nanoseconds = 0;
			queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_d_x, dst_y.data());
			queue.enqueueWriteBuffer(temp_rows_buffer, CL_TRUE, 0, byte_size_temp_rows, temp_rows.data());
			queue.enqueueWriteBuffer(temp_vals_buffer, CL_TRUE, 0, byte_size_temp_vals, temp_vals.data());
			if (d_hyb->ellg.nnz > 0)
			{
				nanoseconds +=
					jc::run_and_time_kernel(kernel_ell,
						queue,
						cl::NDRange(min(MAX_THREADS, jc::best_fit(d_hyb->ellg.n, WORKGROUP_SIZE))),
						cl::NDRange(WORKGROUP_SIZE));
			}
			nanoseconds +=
				jc::run_and_time_kernel(kernel_flat,
					queue,
					cl::NDRange(nworkgroups * WORKGROUP_SIZE),
					cl::NDRange(WORKGROUP_SIZE));
			if (d_hyb->coo.nnz - tail > 0)
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
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	std::cout << std::endl << "Average time: " << average_nanoseconds << " ns\n";
	//increment all values
	for (IndexType i = 0; i < d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n); i++) d_hyb->ellg.jcoeff[i]++;
	for (IndexType i = 0; i < d_hyb->coo.nnz; i++) d_hyb->coo.ir[i]++, d_hyb->coo.jc[i]++;

	return dst_y;
}
#endif

#if HYB_ELLG
std::vector<CL_REAL> spmv_HYB_ELLG(const struct hybellg_t* d_hyb, const std::vector<CL_REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n); i++) d_hyb->ellg.jcoeff[i]--;
	for (IndexType i = 0; i < d_hyb->coo.nnz; i++) d_hyb->coo.ir[i]--, d_hyb->coo.jc[i]--;
	//
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	cl::Program program_ellg =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + ELLG_KERNEL_FILE, context, device);
	cl::Program program_coo =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + COO_KERNEL_FILE, context, device);
#if PRECISION == 2
	cl::Kernel kernel_ellg{ program_ellg, "spmv_ellg_d" };
	cl::Kernel kernel_flat{ program_coo, "spmv_coo_flat_d" };
	cl::Kernel kernel_serial{ program_coo, "spmv_coo_serial_d" };
	cl::Kernel kernel_reduce_update{ program_coo, "spmv_coo_reduce_update_d" };
#else
	cl::Kernel kernel_ellg{ program_ellg, "spmv_ellg_s" };
	cl::Kernel kernel_flat{ program_coo, "spmv_coo_flat_s" };
	cl::Kernel kernel_serial{ program_coo, "spmv_coo_serial_s" };
	cl::Kernel kernel_reduce_update{ program_coo, "spmv_coo_reduce_update_s" };
#endif
	//
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	if (d_hyb->ellg.nnz > 0)
	{
		size_t byte_size_d_nell = (d_hyb->ellg.n + 1) * sizeof(cl_uint);
		size_t byte_size_d_jcoeff = d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n) * sizeof(cl_uint);
		size_t byte_size_d_a = d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n) * sizeof(CL_REAL);
		//
		cl::Buffer d_nell_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_nell };
		cl::Buffer d_jcoeff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jcoeff };
		cl::Buffer d_a_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_a };
		//
		queue.enqueueWriteBuffer(d_nell_buffer, CL_TRUE, 0, byte_size_d_nell, d_hyb->ellg.nell);
		queue.enqueueWriteBuffer(d_jcoeff_buffer, CL_TRUE, 0, byte_size_d_jcoeff, d_hyb->ellg.jcoeff);
		queue.enqueueWriteBuffer(d_a_buffer, CL_TRUE, 0, byte_size_d_a, d_hyb->ellg.a);
		//
		kernel_ellg.setArg(0, d_hyb->ellg.n);
		kernel_ellg.setArg(1, d_hyb->ellg.stride);
		kernel_ellg.setArg(2, d_nell_buffer);
		kernel_ellg.setArg(3, d_jcoeff_buffer);
		kernel_ellg.setArg(4, d_a_buffer);
		kernel_ellg.setArg(5, d_x_buffer);
		kernel_ellg.setArg(6, dst_y_buffer);
	}
    //
    cl_ulong nanoseconds;
    cl_ulong total_nanoseconds = 0;
    //
	if (true) // if (d_hyb->coo.nnz < WARP_SIZE)
	{
		size_t byte_size_d_ir = d_hyb->coo.nnz * sizeof(cl_uint);
		size_t byte_size_d_jc = d_hyb->coo.nnz * sizeof(cl_uint);
		size_t byte_size_d_val = d_hyb->coo.nnz * sizeof(CL_REAL);
		//
		cl::Buffer d_ir_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ir };
		cl::Buffer d_jc_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jc };
		cl::Buffer d_val_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_val };
		//
		queue.enqueueWriteBuffer(d_ir_buffer, CL_TRUE, 0, byte_size_d_ir, d_hyb->coo.ir);
		queue.enqueueWriteBuffer(d_jc_buffer, CL_TRUE, 0, byte_size_d_jc, d_hyb->coo.jc);
		queue.enqueueWriteBuffer(d_val_buffer, CL_TRUE, 0, byte_size_d_val, d_hyb->coo.val);
		//
		kernel_serial.setArg(0, d_hyb->coo.nnz);
		kernel_serial.setArg(1, 0);
		kernel_serial.setArg(2, d_ir_buffer);
		kernel_serial.setArg(3, d_jc_buffer);
		kernel_serial.setArg(4, d_val_buffer);
		kernel_serial.setArg(5, d_x_buffer);
		kernel_serial.setArg(6, dst_y_buffer);
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
						cl::NDRange(min(MAX_THREADS, jc::best_fit(d_hyb->ellg.n, WORKGROUP_SIZE))),
						cl::NDRange(WORKGROUP_SIZE));
			}
			nanoseconds +=
				jc::run_and_time_kernel(kernel_serial,
					queue,
					cl::NDRange(1),
					cl::NDRange(1));
			std::cout << "Run: " << r + 1 << " | Time elapsed: " << nanoseconds << " ns\n";
			total_nanoseconds += nanoseconds;
		}
	}
	else
	{
		// determine specific values
		unsigned IndexType max_workgroups = MAX_THREADS / (2 * WORKGROUP_SIZE);
		//
		unsigned IndexType nunits = d_hyb->coo.nnz / WARP_SIZE;
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
		std::vector<cl_uint> temp_rows(active_warps, 0);
		std::vector<CL_REAL> temp_vals(active_warps, 0);
		//
		size_t byte_size_d_ir = d_hyb->coo.nnz * sizeof(cl_uint);
		size_t byte_size_d_jc = d_hyb->coo.nnz * sizeof(cl_uint);
		size_t byte_size_d_val = d_hyb->coo.nnz * sizeof(CL_REAL);
		size_t byte_size_temp_rows = temp_rows.size() * sizeof(CL_REAL);
		size_t byte_size_temp_vals = temp_vals.size() * sizeof(CL_REAL);
		//
		cl::Buffer d_ir_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ir };
		cl::Buffer d_jc_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jc };
		cl::Buffer d_val_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_val };
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
		queue.enqueueWriteBuffer(d_ir_buffer, CL_TRUE, 0, byte_size_d_ir, d_hyb->coo.ir);
		queue.enqueueWriteBuffer(d_jc_buffer, CL_TRUE, 0, byte_size_d_jc, d_hyb->coo.jc);
		queue.enqueueWriteBuffer(d_val_buffer, CL_TRUE, 0, byte_size_d_val, d_hyb->coo.val);
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
		kernel_serial.setArg(0, d_hyb->coo.nnz - tail);
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
		std::cout << "!!! COO: kernel_flat: A work-group uses " << local_byte_size_shrows_1 + local_byte_size_shvals_1 << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
		std::cout << "!!! COO: kernel_reduce_update: A work-group uses " << local_byte_size_shrows_2 + local_byte_size_shvals_2 << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
		for (int r = 0; r < REPEAT; r++)
		{
			nanoseconds = 0;
			queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_d_x, dst_y.data());
			queue.enqueueWriteBuffer(temp_rows_buffer, CL_TRUE, 0, byte_size_temp_rows, temp_rows.data());
			queue.enqueueWriteBuffer(temp_vals_buffer, CL_TRUE, 0, byte_size_temp_vals, temp_vals.data());
			if (d_hyb->ellg.nnz > 0)
			{
				nanoseconds +=
					jc::run_and_time_kernel(kernel_ellg,
						queue,
						cl::NDRange(min(MAX_THREADS, jc::best_fit(d_hyb->ellg.n, WORKGROUP_SIZE))),
						cl::NDRange(WORKGROUP_SIZE));
			}
			nanoseconds +=
				jc::run_and_time_kernel(kernel_flat,
					queue,
					cl::NDRange(nworkgroups * WORKGROUP_SIZE),
					cl::NDRange(WORKGROUP_SIZE));
			if (d_hyb->coo.nnz - tail > 0)
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
	}
    queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
    double average_nanoseconds = total_nanoseconds / (double)REPEAT;
    std::cout << std::endl << "Average time: " << average_nanoseconds << " ns\n";
	//increment all values
	for (IndexType i = 0; i < d_hyb->ellg.stride * *(d_hyb->ellg.nell + d_hyb->ellg.n); i++) d_hyb->ellg.jcoeff[i]++;
	for (IndexType i = 0; i < d_hyb->coo.nnz; i++) d_hyb->coo.ir[i]++, d_hyb->coo.jc[i]++;

    return dst_y;
}
#endif

#if HYB_HLL
std::vector<CL_REAL> spmv_HYB_HLL(const struct hybhll_t* d_hyb, const std::vector<CL_REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_hyb->hll.total_mem; i++) d_hyb->hll.jcoeff[i]--;
	for (IndexType i = 0; i < d_hyb->hll.nhoff; i++) d_hyb->hll.hoff[i]--;
	for (IndexType i = 0; i < d_hyb->coo.nnz; i++) d_hyb->coo.ir[i]--, d_hyb->coo.jc[i]--;
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	cl::Program program_hll =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + HLL_KERNEL_FILE, context, device);
	cl::Program program_coo =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + COO_KERNEL_FILE, context, device);
#if PRECISION == 2
	cl::Kernel kernel_hll{ program_hll, "spmv_hll_d" };
	cl::Kernel kernel_flat{ program_coo, "spmv_coo_flat_d" };
	cl::Kernel kernel_serial{ program_coo, "spmv_coo_serial_d" };
	cl::Kernel kernel_reduce_update{ program_coo, "spmv_coo_reduce_update_d" };
#else
	cl::Kernel kernel_hll{ program_hll, "spmv_hll_s" };
	cl::Kernel kernel_flat{ program_coo, "spmv_coo_flat_s" };
	cl::Kernel kernel_serial{ program_coo, "spmv_coo_serial_s" };
	cl::Kernel kernel_reduce_update{ program_coo, "spmv_coo_reduce_update_s" };
#endif*
	//
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	if (d_hyb->hll.nnz > 0)
	{
		size_t byte_size_d_nell = d_hyb->hll.nhoff * sizeof(cl_uint);
		size_t byte_size_d_jcoeff = d_hyb->hll.total_mem * sizeof(cl_uint);
		size_t byte_size_d_hoff = d_hyb->hll.nhoff * sizeof(cl_uint);
		size_t byte_size_d_a = d_hyb->hll.total_mem * sizeof(CL_REAL);
		//
		cl::Buffer d_nell_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_nell };
		cl::Buffer d_jcoeff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jcoeff };
		cl::Buffer d_hoff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_hoff };
		cl::Buffer d_a_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_a };
		//
		queue.enqueueWriteBuffer(d_nell_buffer, CL_TRUE, 0, byte_size_d_nell, d_hyb->hll.nell);
		queue.enqueueWriteBuffer(d_jcoeff_buffer, CL_TRUE, 0, byte_size_d_jcoeff, d_hyb->hll.jcoeff);
		queue.enqueueWriteBuffer(d_hoff_buffer, CL_TRUE, 0, byte_size_d_hoff, d_hyb->hll.hoff);
		queue.enqueueWriteBuffer(d_a_buffer, CL_TRUE, 0, byte_size_d_a, d_hyb->hll.a);
		//
		kernel_hll.setArg(0, d_hyb->hll.n);
		kernel_hll.setArg(1, HLL_HACKSIZE);
		kernel_hll.setArg(2, d_nell_buffer);
		kernel_hll.setArg(3, d_jcoeff_buffer);
		kernel_hll.setArg(4, d_hoff_buffer);
		kernel_hll.setArg(5, d_a_buffer);
		kernel_hll.setArg(6, d_x_buffer);
		kernel_hll.setArg(7, dst_y_buffer);
	}
	//
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	if (true) // if (d_hyb->coo.nnz < WARP_SIZE)
	{
		size_t byte_size_d_ir = d_hyb->coo.nnz * sizeof(cl_uint);
		size_t byte_size_d_jc = d_hyb->coo.nnz * sizeof(cl_uint);
		size_t byte_size_d_val = d_hyb->coo.nnz * sizeof(CL_REAL);
		//
		cl::Buffer d_ir_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ir };
		cl::Buffer d_jc_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jc };
		cl::Buffer d_val_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_val };
		//
		queue.enqueueWriteBuffer(d_ir_buffer, CL_TRUE, 0, byte_size_d_ir, d_hyb->coo.ir);
		queue.enqueueWriteBuffer(d_jc_buffer, CL_TRUE, 0, byte_size_d_jc, d_hyb->coo.jc);
		queue.enqueueWriteBuffer(d_val_buffer, CL_TRUE, 0, byte_size_d_val, d_hyb->coo.val);
		//
		kernel_serial.setArg(0, d_hyb->coo.nnz);
		kernel_serial.setArg(1, 0);
		kernel_serial.setArg(2, d_ir_buffer);
		kernel_serial.setArg(3, d_jc_buffer);
		kernel_serial.setArg(4, d_val_buffer);
		kernel_serial.setArg(5, d_x_buffer);
		kernel_serial.setArg(6, dst_y_buffer);
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
						cl::NDRange(min(MAX_THREADS, jc::best_fit(d_hyb->hll.n, WORKGROUP_SIZE))),
						cl::NDRange(WORKGROUP_SIZE));
			}
			nanoseconds +=
				jc::run_and_time_kernel(kernel_serial,
					queue,
					cl::NDRange(1),
					cl::NDRange(1));
			std::cout << "Run: " << r + 1 << " | Time elapsed: " << nanoseconds << " ns\n";
			total_nanoseconds += nanoseconds;
		}
	}
	else
	{
		// determine specific values
		unsigned IndexType max_workgroups = MAX_THREADS / (2 * WORKGROUP_SIZE);
		//
		unsigned IndexType nunits = d_hyb->coo.nnz / WARP_SIZE;
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
		std::vector<cl_uint> temp_rows(active_warps, 0);
		std::vector<CL_REAL> temp_vals(active_warps, 0);
		//
		size_t byte_size_d_ir = d_hyb->coo.nnz * sizeof(cl_uint);
		size_t byte_size_d_jc = d_hyb->coo.nnz * sizeof(cl_uint);
		size_t byte_size_d_val = d_hyb->coo.nnz * sizeof(CL_REAL);
		size_t byte_size_temp_rows = temp_rows.size() * sizeof(CL_REAL);
		size_t byte_size_temp_vals = temp_vals.size() * sizeof(CL_REAL);
		//
		cl::Buffer d_ir_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ir };
		cl::Buffer d_jc_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jc };
		cl::Buffer d_val_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_val };
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
		queue.enqueueWriteBuffer(d_ir_buffer, CL_TRUE, 0, byte_size_d_ir, d_hyb->coo.ir);
		queue.enqueueWriteBuffer(d_jc_buffer, CL_TRUE, 0, byte_size_d_jc, d_hyb->coo.jc);
		queue.enqueueWriteBuffer(d_val_buffer, CL_TRUE, 0, byte_size_d_val, d_hyb->coo.val);
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
		kernel_serial.setArg(0, d_hyb->coo.nnz - tail);
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
		std::cout << "!!! COO: kernel_flat: A work-group uses " << local_byte_size_shrows_1 + local_byte_size_shvals_1 << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
		std::cout << "!!! COO: kernel_reduce_update: A work-group uses " << local_byte_size_shrows_2 + local_byte_size_shvals_2 << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
		for (int r = 0; r < REPEAT; r++)
		{
			nanoseconds = 0;
			queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_d_x, dst_y.data());
			queue.enqueueWriteBuffer(temp_rows_buffer, CL_TRUE, 0, byte_size_temp_rows, temp_rows.data());
			queue.enqueueWriteBuffer(temp_vals_buffer, CL_TRUE, 0, byte_size_temp_vals, temp_vals.data());
			if (d_hyb->hll.nnz > 0)
			{
				nanoseconds +=
					jc::run_and_time_kernel(kernel_hll,
						queue,
						cl::NDRange(min(MAX_THREADS, jc::best_fit(d_hyb->hll.n, WORKGROUP_SIZE))),
						cl::NDRange(WORKGROUP_SIZE));
			}
			nanoseconds +=
				jc::run_and_time_kernel(kernel_flat,
					queue,
					cl::NDRange(nworkgroups * WORKGROUP_SIZE),
					cl::NDRange(WORKGROUP_SIZE));
			if (d_hyb->coo.nnz - tail > 0)
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
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	std::cout << std::endl << "Average time: " << average_nanoseconds << " ns\n";
	//increment all values
	for (IndexType i = 0; i < d_hyb->hll.total_mem; i++) d_hyb->hll.jcoeff[i]++;
	for (IndexType i = 0; i < d_hyb->hll.nhoff; i++) d_hyb->hll.hoff[i]++;
	for (IndexType i = 0; i < d_hyb->coo.nnz; i++) d_hyb->coo.ir[i]++, d_hyb->coo.jc[i]++;

	return dst_y;
}
#endif

#if HYB_HLL_LOCAL
std::vector<CL_REAL> spmv_HYB_HLL_LOCAL(const struct hybhll_t* d_hyb, const std::vector<CL_REAL> d_x)
{
	//decrement all values
	for (IndexType i = 0; i < d_hyb->hll.total_mem; i++) d_hyb->hll.jcoeff[i]--;
	for (IndexType i = 0; i < d_hyb->hll.nhoff; i++) d_hyb->hll.hoff[i]--;
	for (IndexType i = 0; i < d_hyb->coo.nnz; i++) d_hyb->coo.ir[i]--, d_hyb->coo.jc[i]--;
	std::vector<CL_REAL> dst_y(d_x.size(), 0);
	//
	cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
	cl::Context context{ device };
	cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };
	//
	cl::Program program_hll_local =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + HLL_LOCAL_KERNEL_FILE, context, device);
	cl::Program program_coo =
		jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + COO_KERNEL_FILE, context, device);
#if PRECISION == 2
	cl::Kernel kernel_hll_local{ program_hll_local, "spmv_hll_local_d" };
	cl::Kernel kernel_flat{ program_coo, "spmv_coo_flat_d" };
	cl::Kernel kernel_serial{ program_coo, "spmv_coo_serial_d" };
	cl::Kernel kernel_reduce_update{ program_coo, "spmv_coo_reduce_update_d" };
#else
	cl::Kernel kernel_hll_local{ program_hll_local, "spmv_hll_local_s" };
	cl::Kernel kernel_flat{ program_coo, "spmv_coo_flat_s" };
	cl::Kernel kernel_serial{ program_coo, "spmv_coo_serial_s" };
	cl::Kernel kernel_reduce_update{ program_coo, "spmv_coo_reduce_update_s" };
#endif
	//
	size_t byte_size_d_x = d_x.size() * sizeof(CL_REAL);
	size_t byte_size_dst_y = dst_y.size() * sizeof(CL_REAL);
	//
	cl::Buffer d_x_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_x };
	cl::Buffer dst_y_buffer{ context, CL_MEM_WRITE_ONLY, byte_size_dst_y };
	//
	queue.enqueueWriteBuffer(d_x_buffer, CL_TRUE, 0, byte_size_d_x, d_x.data());
	//
	if (d_hyb->hll.nnz > 0)
	{
		size_t byte_size_d_nell = d_hyb->hll.nhoff * sizeof(cl_uint);
		size_t byte_size_d_jcoeff = d_hyb->hll.total_mem * sizeof(cl_uint);
		size_t byte_size_d_hoff = d_hyb->hll.nhoff * sizeof(cl_uint);
		size_t byte_size_d_a = d_hyb->hll.total_mem * sizeof(CL_REAL);
		//
		cl::Buffer d_nell_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_nell };
		cl::Buffer d_jcoeff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jcoeff };
		cl::Buffer d_hoff_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_hoff };
		cl::Buffer d_a_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_a };
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
		kernel_hll_local.setArg(0, d_hyb->hll.n);
		kernel_hll_local.setArg(1, HLL_HACKSIZE);
		kernel_hll_local.setArg(2, d_nell_buffer);
		kernel_hll_local.setArg(3, d_jcoeff_buffer);
		kernel_hll_local.setArg(4, d_hoff_buffer);
		kernel_hll_local.setArg(5, d_a_buffer);
		kernel_hll_local.setArg(6, d_x_buffer);
		kernel_hll_local.setArg(7, dst_y_buffer);
		kernel_hll_local.setArg(8, cl::Local(local_byte_size_shhoff));
		//
		std::cout << "!!! HLL_LOCAL: A work-group uses " << local_byte_size_shhoff << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
	}
	//
	cl_ulong nanoseconds;
	cl_ulong total_nanoseconds = 0;
	//
	if (true) // if (d_hyb->coo.nnz < WARP_SIZE)
	{
		size_t byte_size_d_ir = d_hyb->coo.nnz * sizeof(cl_uint);
		size_t byte_size_d_jc = d_hyb->coo.nnz * sizeof(cl_uint);
		size_t byte_size_d_val = d_hyb->coo.nnz * sizeof(CL_REAL);
		//
		cl::Buffer d_ir_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ir };
		cl::Buffer d_jc_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jc };
		cl::Buffer d_val_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_val };
		//
		queue.enqueueWriteBuffer(d_ir_buffer, CL_TRUE, 0, byte_size_d_ir, d_hyb->coo.ir);
		queue.enqueueWriteBuffer(d_jc_buffer, CL_TRUE, 0, byte_size_d_jc, d_hyb->coo.jc);
		queue.enqueueWriteBuffer(d_val_buffer, CL_TRUE, 0, byte_size_d_val, d_hyb->coo.val);
		//
		kernel_serial.setArg(0, d_hyb->coo.nnz);
		kernel_serial.setArg(1, 0);
		kernel_serial.setArg(2, d_ir_buffer);
		kernel_serial.setArg(3, d_jc_buffer);
		kernel_serial.setArg(4, d_val_buffer);
		kernel_serial.setArg(5, d_x_buffer);
		kernel_serial.setArg(6, dst_y_buffer);
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
						cl::NDRange(min(MAX_THREADS, jc::best_fit(d_hyb->hll.n, WORKGROUP_SIZE))),
						cl::NDRange(WORKGROUP_SIZE));
			}
			nanoseconds +=
				jc::run_and_time_kernel(kernel_serial,
					queue,
					cl::NDRange(1),
					cl::NDRange(1));
			std::cout << "Run: " << r + 1 << " | Time elapsed: " << nanoseconds << " ns\n";
			total_nanoseconds += nanoseconds;
		}
	}
	else
	{
		// determine specific values
		unsigned IndexType max_workgroups = MAX_THREADS / (2 * WORKGROUP_SIZE);
		//
		unsigned IndexType nunits = d_hyb->coo.nnz / WARP_SIZE;
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
		std::vector<cl_uint> temp_rows(active_warps, 0);
		std::vector<CL_REAL> temp_vals(active_warps, 0);
		//
		size_t byte_size_d_ir = d_hyb->coo.nnz * sizeof(cl_uint);
		size_t byte_size_d_jc = d_hyb->coo.nnz * sizeof(cl_uint);
		size_t byte_size_d_val = d_hyb->coo.nnz * sizeof(CL_REAL);
		size_t byte_size_temp_rows = temp_rows.size() * sizeof(CL_REAL);
		size_t byte_size_temp_vals = temp_vals.size() * sizeof(CL_REAL);
		//
		cl::Buffer d_ir_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_ir };
		cl::Buffer d_jc_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_jc };
		cl::Buffer d_val_buffer{ context, CL_MEM_READ_ONLY, byte_size_d_val };
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
		queue.enqueueWriteBuffer(d_ir_buffer, CL_TRUE, 0, byte_size_d_ir, d_hyb->coo.ir);
		queue.enqueueWriteBuffer(d_jc_buffer, CL_TRUE, 0, byte_size_d_jc, d_hyb->coo.jc);
		queue.enqueueWriteBuffer(d_val_buffer, CL_TRUE, 0, byte_size_d_val, d_hyb->coo.val);
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
		kernel_serial.setArg(0, d_hyb->coo.nnz - tail);
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
		std::cout << "!!! COO: kernel_flat: A work-group uses " << local_byte_size_shrows_1 + local_byte_size_shvals_1 << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
		std::cout << "!!! COO: kernel_reduce_update: A work-group uses " << local_byte_size_shrows_2 + local_byte_size_shvals_2 << " bytes of the max local memory size of " << size << " bytes per Compute Unit !!!" << std::endl << std::endl;
		for (int r = 0; r < REPEAT; r++)
		{
			nanoseconds = 0;
			queue.enqueueWriteBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_d_x, dst_y.data());
			queue.enqueueWriteBuffer(temp_rows_buffer, CL_TRUE, 0, byte_size_temp_rows, temp_rows.data());
			queue.enqueueWriteBuffer(temp_vals_buffer, CL_TRUE, 0, byte_size_temp_vals, temp_vals.data());
			if (d_hyb->hll.nnz > 0)
			{
				nanoseconds +=
					jc::run_and_time_kernel(kernel_hll_local,
						queue,
						cl::NDRange(min(MAX_THREADS, jc::best_fit(d_hyb->hll.n, WORKGROUP_SIZE))),
						cl::NDRange(WORKGROUP_SIZE));
			}
			nanoseconds +=
				jc::run_and_time_kernel(kernel_flat,
					queue,
					cl::NDRange(nworkgroups * WORKGROUP_SIZE),
					cl::NDRange(WORKGROUP_SIZE));
			if (d_hyb->coo.nnz - tail > 0)
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
	}
	queue.enqueueReadBuffer(dst_y_buffer, CL_TRUE, 0, byte_size_dst_y, dst_y.data());
	double average_nanoseconds = total_nanoseconds / (double)REPEAT;
	std::cout << std::endl << "Average time: " << average_nanoseconds << " ns\n";
	//increment all values
	for (IndexType i = 0; i < d_hyb->hll.total_mem; i++) d_hyb->hll.jcoeff[i]++;
	for (IndexType i = 0; i < d_hyb->hll.nhoff; i++) d_hyb->hll.hoff[i]++;
	for (IndexType i = 0; i < d_hyb->coo.nnz; i++) d_hyb->coo.ir[i]++, d_hyb->coo.jc[i]++;

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
#if HYB_ELL || HYB_ELLG
	struct hybellg_t hyb_ellg;
#endif
#if HYB_HLL || HYB_HLL_LOCAL
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
	IndexType n = coo.n;
	std::cout << "-- INPUT FILE LOADED --" << std::endl << std::endl;
	std::cout << "-- PRE-PROCESSING INPUT --" << std::endl;
#if HYB_ELL || HYB_ELLG
	COO_To_HYBELLG(&coo, &hyb_ellg, HYB_ELLG_LOG);
#endif
#if HYB_HLL || HYB_HLL_LOCAL
	COO_To_HYBHLL(&coo, &hybhll_t, HYB_HLL_LOG);
#endif
	FreeCOO(&coo);
	std::cout << "-- DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;

	std::vector<CL_REAL> x = std::vector<CL_REAL>();
	for (IndexType i = 0; i < n; i++)
		x.push_back(i);

#if HYB_ELL
	std::cout << std::endl << "-- STARTING HYB_ELL KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y1 = spmv_HYB_ELL(&hyb_ellg, x);
	std::cout << std::endl << "-- FINISHED HYB_ELL KERNEL OPERATION --" << std::endl << std::endl;
	if (HYB_ELL_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y1.size(); i++)
			std::cout << y1[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HYB_ELLG
	std::cout << std::endl << "-- STARTING HYB_ELL-G KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y2 = spmv_HYB_ELLG(&hyb_ellg, x);
	std::cout << std::endl << "-- FINISHED HYB_ELL-G KERNEL OPERATION --" << std::endl << std::endl;
	if (HYB_ELLG_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y2.size(); i++)
			std::cout << y2[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HYB_HLL
	std::cout << std::endl << "-- STARTING HYB_HLL KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y3 = spmv_HYB_HLL(&hybhll_t, x);
	std::cout << std::endl << "-- FINISHED HYB_HLL KERNEL OPERATION --" << std::endl << std::endl;
	if (HYB_HLL_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y3.size(); i++)
			std::cout << y3[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HYB_HLL_LOCAL
	std::cout << std::endl << "-- STARTING HYB_HLL_LOCAL KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y4 = spmv_HYB_HLL_LOCAL(&hybhll_t, x);
	std::cout << std::endl << "-- FINISHED HYB_HLL_LOCAL KERNEL OPERATION --" << std::endl << std::endl;
	if (HYB_HLL_LOCAL_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y4.size(); i++)
			std::cout << y4[i] << " ";
		std::cout << std::endl;
	}
#endif

	x.clear();
#if HYB_ELL || HYB_ELLG
	FreeHYBELLG(&hyb_ellg);
#if HYB_ELL
	y1.clear();
#endif
#if HYB_ELLG
	y2.clear();
#endif
#endif
#if HYB_HLL || HYB_HLL_LOCAL
	FreeHYBHLL(&hybhll_t);
#if HYB_HLL
	y3.clear();
#endif
#if HYB_HLL_LOCAL
	y4.clear();
#endif
#endif
#if DEBUG
	system("PAUSE"); // for debugging
#endif

	return 0;
}

#endif