#define WARP_SIZE 32

// Implementation based off the spmv_coo_flat_device.cu.h from sc2009_spmv by: Nathan Bell & Michael Garland
// URL: https://code.google.com/archive/p/cusp-library/downloads
//
// Does not work properly, except for the serial kernel !!!
//
/*-------------------------------- Single-precision----------------------------------*/
float segreduce_warp_s(int thread_lane, int row, float val, __local int* rows, __local float* vals)
{
	int local_index = get_local_id(0);
	rows[local_index] = row;
	vals[local_index] = val;

	if (thread_lane >= 1 && row == rows[local_index - 1]) { vals[local_index] = val = val + vals[local_index - 1]; }
	if (thread_lane >= 2 && row == rows[local_index - 2]) { vals[local_index] = val = val + vals[local_index - 2]; }
	if (thread_lane >= 4 && row == rows[local_index - 4]) { vals[local_index] = val = val + vals[local_index - 4]; }
	if (thread_lane >= 8 && row == rows[local_index - 8]) { vals[local_index] = val = val + vals[local_index - 8]; }
	if (thread_lane >= 16 && row == rows[local_index - 16]) { vals[local_index] = val = val + vals[local_index - 16]; }

	return val;
}

void segreduce_block_s(__local int* idx, __local float* val)
{
	float left = 0;
	int local_index = get_local_id(0);

	if (local_index >= 1 && idx[local_index] == idx[local_index - 1]) { left = val[local_index - 1]; } barrier(CLK_LOCAL_MEM_FENCE); val[local_index] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
	if (local_index >= 2 && idx[local_index] == idx[local_index - 2]) { left = val[local_index - 2]; } barrier(CLK_LOCAL_MEM_FENCE); val[local_index] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
	if (local_index >= 4 && idx[local_index] == idx[local_index - 4]) { left = val[local_index - 4]; } barrier(CLK_LOCAL_MEM_FENCE); val[local_index] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
	if (local_index >= 8 && idx[local_index] == idx[local_index - 8]) { left = val[local_index - 8]; } barrier(CLK_LOCAL_MEM_FENCE); val[local_index] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
	if (local_index >= 16 && idx[local_index] == idx[local_index - 16]) { left = val[local_index - 16]; } barrier(CLK_LOCAL_MEM_FENCE); val[local_index] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
	if (local_index >= 32 && idx[local_index] == idx[local_index - 32]) { left = val[local_index - 32]; } barrier(CLK_LOCAL_MEM_FENCE); val[local_index] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
	if (local_index >= 64 && idx[local_index] == idx[local_index - 64]) { left = val[local_index - 64]; } barrier(CLK_LOCAL_MEM_FENCE); val[local_index] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
	if (local_index >= 128 && idx[local_index] == idx[local_index - 128]) { left = val[local_index - 128]; } barrier(CLK_LOCAL_MEM_FENCE); val[local_index] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
	if (local_index >= 256 && idx[local_index] == idx[local_index - 256]) { left = val[local_index - 256]; } barrier(CLK_LOCAL_MEM_FENCE); val[local_index] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void spmv_coo_flat_s(int nnz, int interval_size,
	__constant unsigned int* d_ir,
	__constant unsigned int* d_jc,
	__constant float* d_val,
	__constant float* d_x,
	__global float* dst_y,
	__global int* temp_rows,
	__global float* temp_vals,
	__local unsigned int* rows,
	__local float* vals)
{
	int thread_id = get_global_id(0);                 // global thread index
	int thread_local_id = get_local_id(0);
	int thread_lane = thread_local_id & (WARP_SIZE - 1);                           // thread index within the warp
	int warp_id = thread_id / WARP_SIZE;                               // global warp index

	int interval_begin = warp_id * interval_size;                            // warp's offset into d_ir,d_jc,d_val
	int interval_end = min(interval_begin + interval_size, nnz);  // end of warps's work

	if (interval_begin >= interval_end)                                                   // warp has no work to do 
		return;

	if (thread_lane == 31) {
		// initialize the carry in values
		rows[thread_local_id] = d_ir[interval_begin];
		vals[thread_local_id] = 0;
	}

	for (int n = interval_begin + thread_lane; n < interval_end; n += WARP_SIZE)
	{
		unsigned int row = d_ir[n];                                         // row index (i)
		float val = d_val[n] * d_x[d_jc[n]];							// A(i,j) * d_x(j)

		barrier(CLK_GLOBAL_MEM_FENCE);

		if (thread_lane == 0)
		{
			if (row == rows[thread_local_id + 31])
				val += vals[thread_local_id + 31];                        // row continues
			else
				dst_y[rows[thread_local_id + 31]] += vals[thread_local_id + 31];  // row terminated
		}

		val = segreduce_warp_s(thread_lane, row, val, rows, vals);      // segmented reduction in shared memory
		
		barrier(CLK_LOCAL_MEM_FENCE);

		if (thread_lane < 31 && row != rows[thread_local_id + 1])
			dst_y[row] += val;                                            // row terminated
	}

	if (thread_lane == 31)
	{
		// write the carry out values
		temp_rows[warp_id] = rows[thread_local_id];
		temp_vals[warp_id] = vals[thread_local_id];
	}
}

__kernel void spmv_coo_reduce_update_s(int num_warps,
	__global int* temp_rows,
	__global float* temp_vals,
	__global float* dst_y,
	__local int* rows,
	__local float* vals)
{
	int thread_local_id = get_local_id(0);
	int end = num_warps - (num_warps & (get_local_size(0) - 1));

	if (thread_local_id == 0) {
		rows[get_local_size(0)] = -1;
		vals[get_local_size(0)] = 0.0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int i = thread_local_id;

	while (i < end) {
		// do full blocks
		rows[thread_local_id] = temp_rows[i];
		vals[thread_local_id] = temp_vals[i];

		barrier(CLK_LOCAL_MEM_FENCE);

		segreduce_block_s(rows, vals);

		if (rows[thread_local_id] != rows[thread_local_id + 1])
			dst_y[rows[thread_local_id]] += vals[thread_local_id];

		barrier(CLK_LOCAL_MEM_FENCE);

		i += get_local_size(0);
	}

	if (end < num_warps) {
		if (i < num_warps) {
			rows[thread_local_id] = temp_rows[i];
			vals[thread_local_id] = temp_vals[i];
		}
		else {
			rows[thread_local_id] = -1;
			vals[thread_local_id] = 0.0;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		segreduce_block_s(rows, vals);

		if (i < num_warps)
			if (rows[thread_local_id] != rows[thread_local_id + 1])
				dst_y[rows[thread_local_id]] += vals[thread_local_id];
	}
}

__kernel void spmv_coo_serial_s(int nnz, int offset,
	__constant unsigned int* d_ir,
	__constant unsigned int* d_jc,
	__constant float* d_val,
	__constant float* d_x,
	__global float* dst_y)
{
	for (int n = 0; n < nnz; n++) 
	{
		dst_y[d_ir[n + offset]] += d_val[n + offset] * d_x[d_jc[n + offset]];
	}
}

/*-------------------------------- Double-precision----------------------------------*/
double segreduce_warp_d(int thread_lane, int row, double val, __local int* rows, __local double* vals)
{
	int local_index = get_local_id(0);
	rows[local_index] = row;
	vals[local_index] = val;

	if (thread_lane >= 1 && row == rows[local_index - 1]) { vals[local_index] = val = val + vals[local_index - 1]; }
	if (thread_lane >= 2 && row == rows[local_index - 2]) { vals[local_index] = val = val + vals[local_index - 2]; }
	if (thread_lane >= 4 && row == rows[local_index - 4]) { vals[local_index] = val = val + vals[local_index - 4]; }
	if (thread_lane >= 8 && row == rows[local_index - 8]) { vals[local_index] = val = val + vals[local_index - 8]; }
	if (thread_lane >= 16 && row == rows[local_index - 16]) { vals[local_index] = val = val + vals[local_index - 16]; }

	return val;
}

void segreduce_block_d(__local int* idx, __local double* val)
{
	double left = 0;
	int local_index = get_local_id(0);

	if (local_index >= 1 && idx[local_index] == idx[local_index - 1]) { left = val[local_index - 1]; } barrier(CLK_LOCAL_MEM_FENCE); val[local_index] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
	if (local_index >= 2 && idx[local_index] == idx[local_index - 2]) { left = val[local_index - 2]; } barrier(CLK_LOCAL_MEM_FENCE); val[local_index] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
	if (local_index >= 4 && idx[local_index] == idx[local_index - 4]) { left = val[local_index - 4]; } barrier(CLK_LOCAL_MEM_FENCE); val[local_index] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
	if (local_index >= 8 && idx[local_index] == idx[local_index - 8]) { left = val[local_index - 8]; } barrier(CLK_LOCAL_MEM_FENCE); val[local_index] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
	if (local_index >= 16 && idx[local_index] == idx[local_index - 16]) { left = val[local_index - 16]; } barrier(CLK_LOCAL_MEM_FENCE); val[local_index] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
	if (local_index >= 32 && idx[local_index] == idx[local_index - 32]) { left = val[local_index - 32]; } barrier(CLK_LOCAL_MEM_FENCE); val[local_index] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
	if (local_index >= 64 && idx[local_index] == idx[local_index - 64]) { left = val[local_index - 64]; } barrier(CLK_LOCAL_MEM_FENCE); val[local_index] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
	if (local_index >= 128 && idx[local_index] == idx[local_index - 128]) { left = val[local_index - 128]; } barrier(CLK_LOCAL_MEM_FENCE); val[local_index] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
	if (local_index >= 256 && idx[local_index] == idx[local_index - 256]) { left = val[local_index - 256]; } barrier(CLK_LOCAL_MEM_FENCE); val[local_index] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void spmv_coo_flat_d(int nnz, int interval_size,
	__constant unsigned int* d_ir,
	__constant unsigned int* d_jc,
	__constant double* d_val,
	__constant double* d_x,
	__global double* dst_y,
	__global int* temp_rows,
	__global double* temp_vals,
	__local unsigned int* rows,
	__local double* vals)
{
	int thread_id = get_global_id(0);                 // global thread index
	int thread_local_id = get_local_id(0);
	int thread_lane = thread_local_id & (WARP_SIZE - 1);                           // thread index within the warp
	int warp_id = thread_id / WARP_SIZE;                               // global warp index

	int interval_begin = warp_id * interval_size;                            // warp's offset into d_ir,d_jc,d_val
	int interval_end = min(interval_begin + interval_size, nnz);  // end of warps's work

	if (interval_begin >= interval_end)                                                   // warp has no work to do 
		return;

	if (thread_lane == 31) {
		// initialize the carry in values
		rows[thread_local_id] = d_ir[interval_begin];
		vals[thread_local_id] = 0;
	}

	for (int n = interval_begin + thread_lane; n < interval_end; n += WARP_SIZE)
	{
		unsigned int row = d_ir[n];                                         // row index (i)
		double val = d_val[n] * d_x[d_jc[n]];                            // A(i,j) * d_x(j)

		barrier(CLK_GLOBAL_MEM_FENCE);

		if (thread_lane == 0)
		{
			if (row == rows[thread_local_id + 31])
				val += vals[thread_local_id + 31];                        // row continues
			else
				dst_y[rows[thread_local_id + 31]] += vals[thread_local_id + 31];  // row terminated
		}

		val = segreduce_warp_d(thread_lane, row, val, rows, vals);      // segmented reduction in shared memory
		
		barrier(CLK_LOCAL_MEM_FENCE);

		if (thread_lane < 31 && row != rows[thread_local_id + 1])
			dst_y[row] += val;                                            // row terminated
	}

	if (thread_lane == 31)
	{
		// write the carry out values
		temp_rows[warp_id] = rows[thread_local_id];
		temp_vals[warp_id] = vals[thread_local_id];
	}
}

__kernel void spmv_coo_reduce_update_d(int num_warps,
	__global int* temp_rows,
	__global double* temp_vals,
	__global double* dst_y,
	__local int* rows,
	__local double* vals)
{
	int thread_local_id = get_local_id(0);
	int end = num_warps - (num_warps & (get_local_size(0) - 1));

	if (thread_local_id == 0) {
		rows[get_local_size(0)] = -1;
		vals[get_local_size(0)] = 0.0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int i = thread_local_id;

	while (i < end) {
		// do full blocks
		rows[thread_local_id] = temp_rows[i];
		vals[thread_local_id] = temp_vals[i];

		barrier(CLK_LOCAL_MEM_FENCE);

		segreduce_block_d(rows, vals);

		if (rows[thread_local_id] != rows[thread_local_id + 1])
			dst_y[rows[thread_local_id]] += vals[thread_local_id];

		barrier(CLK_LOCAL_MEM_FENCE);

		i += get_local_size(0);
	}

	if (end < num_warps) {
		if (i < num_warps) {
			rows[thread_local_id] = temp_rows[i];
			vals[thread_local_id] = temp_vals[i];
		}
		else {
			rows[thread_local_id] = -1;
			vals[thread_local_id] = 0.0;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		segreduce_block_d(rows, vals);

		if (i < num_warps)
			if (rows[thread_local_id] != rows[thread_local_id + 1])
				dst_y[rows[thread_local_id]] += vals[thread_local_id];
	}
}

__kernel void spmv_coo_serial_d(int nnz, int offset,
	__constant unsigned int* d_ir,
	__constant unsigned int* d_jc,
	__constant double* d_val,
	__constant double* d_x,
	__global double* dst_y)
{
	for (int n = 0; n < nnz; n++)
	{
		dst_y[d_ir[n + offset]] += d_val[n + offset] * d_x[d_jc[n + offset]];
	}
}