#if PRECISION == 2
/*-------------------------------- Double-precision----------------------------------*/
__kernel void spmv_hll(
	__constant unsigned int* d_nell,
	__constant unsigned int* d_jcoeff,
	__constant unsigned int* d_hoff,
	__constant double* d_a,
	__constant double* d_x,
	__global double* dst_y)
{
	__private unsigned int row_id = get_global_id(0);

	__private unsigned int row_hack_id, row_nell, row_hoff;

	__private unsigned int i, j, k;
	__private double r;

	if (row_id >= N_MATRIX) return;

	row_hack_id = row_id / HACKSIZE;
	row_nell = d_nell[row_hack_id];
	row_hoff = d_hoff[row_hack_id];

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
__kernel void spmv_hll(
	__constant unsigned int* d_nell,
	__constant unsigned int* d_jcoeff,
	__constant unsigned int* d_hoff,
	__constant float* d_a,
	__constant float* d_x,
	__global float* dst_y)
{
	__private unsigned int row_id = get_global_id(0);

	__private unsigned int row_hack_id, row_nell, row_hoff;

	__private unsigned int i, j, k;
	__private float r;

	if (row_id >= N_MATRIX) return;

	row_hack_id = row_id / HACKSIZE;
	row_nell = d_nell[row_hack_id];
	row_hoff = d_hoff[row_hack_id];

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
