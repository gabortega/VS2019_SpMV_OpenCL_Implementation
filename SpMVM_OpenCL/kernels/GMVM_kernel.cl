// Converted from CUDA code provided in the first answer of stackoverflow question
// URL: https://stackoverflow.com/questions/26417475/matrix-vector-multiplication-in-cuda-benchmarking-performance
//
#if PRECISION == 2
/*-------------------------------- Double-precision----------------------------------*/
__kernel void spmv_gmvm(
	__global double* d_val,
	__global double* d_x,
	__global double* dst_y,
	__local double* sharedx)
{
	__private unsigned int global_id = get_global_id(0);
	__private unsigned int local_id = get_local_id(0);
	__private unsigned int i, j, q;

	__private double r = 0.0;

	q = local_id;
#pragma unroll(1)
	for (i = 0; i < N_WORKGROUPS; q = ((++i) * WORKGROUP_SIZE + local_id))
	{
		if (q < N_MATRIX)
			sharedx[local_id] = d_x[q];
		else
			sharedx[local_id] = 0.0f;
		barrier(CLK_LOCAL_MEM_FENCE);

		q = global_id + (WORKGROUP_SIZE * i) * N_MATRIX;
#if WORKGROUP_SIZE >= 32
#pragma unroll(32)
#else
#pragma unroll
#endif
		for (j = 0; j < WORKGROUP_SIZE; q = global_id + ((++j) + WORKGROUP_SIZE * i) * N_MATRIX)
		{
			if (q < NN_MATRIX)
				r += d_val[q] * sharedx[j];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (global_id < N_MATRIX)
		dst_y[global_id] = r;
}

#else
/*-------------------------------- Single-precision----------------------------------*/
__kernel void spmv_gmvm(
	__global float* d_val,
	__global float* d_x,
	__global float* dst_y,
	__local float* sharedx)
{
	__private unsigned int global_id = get_global_id(0);
	__private unsigned int local_id = get_local_id(0);
	__private unsigned int i, j, q;

	__private float r = 0.0;

	q = local_id;
#pragma unroll(1)
	for (i = 0; i < N_WORKGROUPS; q = ((++i) * WORKGROUP_SIZE + local_id))
	{
		if (q < N_MATRIX)
			sharedx[local_id] = d_x[q];
		else
			sharedx[local_id] = 0.0f;
		barrier(CLK_LOCAL_MEM_FENCE);

		q = global_id + (WORKGROUP_SIZE * i) * N_MATRIX;
#if WORKGROUP_SIZE >= 32
#pragma unroll(32)
#else
#pragma unroll
#endif
		for (j = 0; j < WORKGROUP_SIZE; q = global_id + ((++j) + WORKGROUP_SIZE * i) * N_MATRIX)
		{
			if (q < NN_MATRIX)
				r += d_val[q] * sharedx[j];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (global_id < N_MATRIX)
		dst_y[global_id] = r;
}
#endif
