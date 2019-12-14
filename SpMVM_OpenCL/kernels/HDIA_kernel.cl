#if PRECISION == 2
/*-------------------------------- Double-precision----------------------------------*/
__kernel void spmv_hdia(
	__constant unsigned int* d_ndiags,
	__constant int* d_ioff,
	__constant double* d_diags,
	__constant unsigned int* d_hoff,
	__constant unsigned int* d_memoff,
	__constant double* d_x,
	__global double* dst_y)
{
	__private unsigned int row_id = get_global_id(0);

	__private unsigned int row_hack_id, row_memoff, ndiags, row_hoff;

	__private unsigned int i;
	__private long q;
	__private double r;

	if (row_id >= N_MATRIX) return;

	row_hack_id = row_id / HACKSIZE;
	row_memoff = d_memoff[row_hack_id];
	ndiags = d_ndiags[row_hack_id];
	row_hoff = d_hoff[row_hack_id];

	r = 0.0;
#pragma unroll(UNROLL)
	for (i = 0; i < ndiags; i++)
	{
		q = d_ioff[row_hoff + i] + row_id;
		if (q >= 0 && q < N_MATRIX)
		{
			r += *(d_diags + row_memoff + row_id + i * HACKSIZE) * d_x[q];
		}
	}
	dst_y[row_id] = r;
}

#else
/*-------------------------------- Single-precision----------------------------------*/
__kernel void spmv_hdia(
	__constant unsigned int* d_ndiags,
	__constant int* d_ioff,
	__constant float* d_diags,
	__constant unsigned int* d_hoff,
	__constant unsigned int* d_memoff,
	__constant float* d_x,
	__global float* dst_y)
{
	__private unsigned int row_id = get_global_id(0);

	__private unsigned int row_hack_id, row_memoff, ndiags, row_hoff;

	__private unsigned int i;
	__private long q;
	__private float r;

	if (row_id >= N_MATRIX) return;

	row_hack_id = row_id / HACKSIZE;
	row_memoff = d_memoff[row_hack_id];
	ndiags = d_ndiags[row_hack_id];
	row_hoff = d_hoff[row_hack_id];

	r = 0.0;
#pragma unroll
	for (i = 0; i < ndiags; i++)
	{
		q = d_ioff[row_hoff + i] + row_id;
		if (q >= 0 && q < N_MATRIX)
		{
			r += *(d_diags + row_memoff + row_id + i * HACKSIZE) * d_x[q];
		}
	}
	dst_y[row_id] = r;
}
#endif