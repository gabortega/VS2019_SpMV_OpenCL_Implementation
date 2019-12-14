// Implementation based off CUDA_ITSOL SpMV by: Ruipeng Li, Yousef Saad
// URL: https://www-users.cs.umn.edu/~saad/software/CUDA_ITSOL/CUDA_ITSOL.tar.gz
//
#if PRECISION == 2
/*-------------------------------- Double-precision----------------------------------*/
__kernel void spmv_hdia_local(
	__constant unsigned int* d_ndiags,
	__constant int* d_ioff,
	__constant double* d_diags,
	__constant unsigned int* d_hoff,
	__constant unsigned int* d_memoff,
	__constant double* d_x,
	__global double* dst_y,
	__local int* sharedhoff)
{
	__private unsigned int row_id = get_global_id(0);
	__private int row_local_id = get_local_id(0);

	__private int row_hack_id = row_id / HACKSIZE;
	__private int sharedhoff_size = get_local_size(0) / HACKSIZE;
	__private int row_local_hack_id = row_hack_id % sharedhoff_size;

	if (row_local_id < sharedhoff_size)
	{
		sharedhoff[row_local_id] = d_memoff[row_hack_id + row_local_id];
		sharedhoff[row_local_id + sharedhoff_size] = d_ndiags[row_hack_id + row_local_id];
		sharedhoff[row_local_id + 2 * sharedhoff_size] = d_hoff[row_hack_id + row_local_id];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	__private unsigned int row_memoff = sharedhoff[row_local_hack_id];
	__private unsigned int ndiags = sharedhoff[row_local_hack_id + sharedhoff_size];
	__private unsigned int row_hoff = sharedhoff[row_local_hack_id + 2 * sharedhoff_size];

	__private unsigned int i;
	__private long q;
	__private double r;

	if (row_id >= N_MATRIX) return;

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
	dst_y[row_id] += r;
}

#else
/*-------------------------------- Single-precision----------------------------------*/
__kernel void spmv_hdia_local(
	__constant unsigned int* d_ndiags,
	__constant int* d_ioff,
	__constant float* d_diags,
	__constant unsigned int* d_hoff,
	__constant unsigned int* d_memoff,
	__constant float* d_x,
	__global float* dst_y,
	__local int* sharedhoff)
{
	__private unsigned int row_id = get_global_id(0);
	__private int row_local_id = get_local_id(0);

	__private int row_hack_id = row_id / HACKSIZE;
	__private int sharedhoff_size = get_local_size(0) / HACKSIZE;
	__private int row_local_hack_id = row_hack_id % sharedhoff_size;

	if (row_local_id < sharedhoff_size)
	{
		sharedhoff[row_local_id] = d_memoff[row_hack_id + row_local_id];
		sharedhoff[row_local_id + sharedhoff_size] = d_ndiags[row_hack_id + row_local_id];
		sharedhoff[row_local_id + 2 * sharedhoff_size] = d_hoff[row_hack_id + row_local_id];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	__private unsigned int row_memoff = sharedhoff[row_local_hack_id];
	__private unsigned int ndiags = sharedhoff[row_local_hack_id + sharedhoff_size];
	__private unsigned int row_hoff = sharedhoff[row_local_hack_id + 2 * sharedhoff_size];

	__private unsigned int i;
	__private long q;
	__private float r;

	if (row_id >= N_MATRIX) return;

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
	dst_y[row_id] += r;
}
#endif