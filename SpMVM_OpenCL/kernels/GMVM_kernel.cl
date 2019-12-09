// Converted from CUDA code provided in the first answer of stackoverflow question
// URL: https://stackoverflow.com/questions/26417475/matrix-vector-multiplication-in-cuda-benchmarking-performance
//
#if PRECISION == 2
/*-------------------------------- Double-precision----------------------------------*/
__kernel void spmv_gmvm(
	__global double* d_val,
	__constant double* d_x,
	__global double* dst_y,
	__local double* sharedx)
{
	__private unsigned int global_id = get_global_id(0);
	__private unsigned int local_id = get_local_id(0);
	__private unsigned int i, j;

	__private double r = 0.0;

#pragma unroll
	for (i = 0; i < ((N_MATRIX + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE); i++)
	{
		if ((i * WORKGROUP_SIZE + local_id) < N_MATRIX)
			sharedx[local_id] = d_x[local_id + i * WORKGROUP_SIZE];
		else
			sharedx[local_id] = 0.0f;
		barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
		for (j = 0; j < WORKGROUP_SIZE; j++)
		{
			if (global_id + (j + WORKGROUP_SIZE * i) * N_MATRIX < N_MATRIX * N_MATRIX)
			r += d_val[global_id + (j + WORKGROUP_SIZE * i) * N_MATRIX] * sharedx[j];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (global_id < N_MATRIX)
		dst_y[global_id] += r;
}

#else
/*-------------------------------- Single-precision----------------------------------*/
__kernel void spmv_gmvm(
	__global float* d_val,
	__constant float* d_x,
	__global float* dst_y,
	__local float* sharedx)
{
	__private unsigned int global_id = get_global_id(0);
	__private unsigned int local_id = get_local_id(0);
	__private unsigned int i, j;

	__private float r = 0.0;

#pragma unroll
	for (i = 0; i < ((N_MATRIX + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE); i++)
	{
		if ((i * WORKGROUP_SIZE + local_id) < N_MATRIX)
			sharedx[local_id] = d_x[local_id + i * WORKGROUP_SIZE];
		else
			sharedx[local_id] = 0.0f;
		barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
		for (j = 0; j < WORKGROUP_SIZE; j++)
		{
			if (global_id + (j + WORKGROUP_SIZE * i) * N_MATRIX < N_MATRIX * N_MATRIX)
				r += d_val[global_id + (j + WORKGROUP_SIZE * i) * N_MATRIX] * sharedx[j];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (global_id < N_MATRIX)
		dst_y[global_id] += r;
}
#endif
