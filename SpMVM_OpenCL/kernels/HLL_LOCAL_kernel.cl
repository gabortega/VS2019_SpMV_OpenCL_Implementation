#if PRECISION == 2
/*-------------------------------- Double-precision----------------------------------*/
__kernel void spmv_hll_local(
	__constant unsigned int* d_nell,
	__constant unsigned int* d_jcoeff,
	__constant unsigned int* d_hoff,
	__constant double* d_a,
	__constant double* d_x,
	__global double* dst_y,
	__local unsigned int* sharedhoff)
{
	__private unsigned int row_id = get_global_id(0);
	__private unsigned int row_local_id = get_local_id(0);

	if (row_id >= N_MATRIX) return;

	__private unsigned int row_hack_id = row_id / HACKSIZE;
	__private unsigned int sharedhoff_size = get_local_size(0) / HACKSIZE;
	__private unsigned int row_local_hack_id = row_hack_id % sharedhoff_size;

	if (row_local_id < sharedhoff_size)
	{
		sharedhoff[row_local_id] = d_nell[row_hack_id + row_local_id];
		sharedhoff[row_local_id + sharedhoff_size] = d_hoff[row_hack_id + row_local_id];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	__private unsigned int row_nell = sharedhoff[row_local_hack_id];
	__private unsigned int row_hoff = sharedhoff[row_local_hack_id + sharedhoff_size];

	__private unsigned int i, j, k;
	__private double r;

	r = 0.0;
#pragma unroll
	for (i = 0; i < row_nell; i++)
	{
		j = i * HACKSIZE + (row_id % HACKSIZE) + row_hoff;
		k = d_jcoeff[j];
		r += d_a[j] * d_x[k];
	}
	dst_y[row_id] += r;
}

#else
/*-------------------------------- Single-precision----------------------------------*/
__kernel void spmv_hll_local(
	__constant unsigned int* d_nell,
	__constant unsigned int* d_jcoeff,
	__constant unsigned int* d_hoff,
	__constant float* d_a,
	__constant float* d_x,
	__global float* dst_y,
	__local unsigned int* sharedhoff)
{
	__private unsigned int row_id = get_global_id(0);
	__private unsigned int row_local_id = get_local_id(0);

	if (row_id >= N_MATRIX) return;

	__private unsigned int row_hack_id = row_id / HACKSIZE;
	__private unsigned int sharedhoff_size = get_local_size(0) / HACKSIZE;
	__private unsigned int row_local_hack_id = row_hack_id % sharedhoff_size;

	if (row_local_id < sharedhoff_size)
	{
		sharedhoff[row_local_id] = d_nell[row_hack_id + row_local_id];
		sharedhoff[row_local_id + sharedhoff_size] = d_hoff[row_hack_id + row_local_id];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	__private unsigned int row_nell = sharedhoff[row_local_hack_id];
	__private unsigned int row_hoff = sharedhoff[row_local_hack_id + sharedhoff_size];

	__private unsigned int i, j, k;
	__private float r;

	r = 0.0;
#pragma unroll
	for (i = 0; i < row_nell; i++)
	{
		j = i * HACKSIZE + (row_id % HACKSIZE) + row_hoff;
		k = d_jcoeff[j];
		r += d_a[j] * d_x[k];
	}
	dst_y[row_id] += r;
}
#endif
