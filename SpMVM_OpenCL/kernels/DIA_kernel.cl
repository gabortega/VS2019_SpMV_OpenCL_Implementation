// Implementation based off CUDA_ITSOL SpMV by: Ruipeng Li, Yousef Saad
// URL: https://www-users.cs.umn.edu/~saad/software/CUDA_ITSOL/CUDA_ITSOL.tar.gz
//
#if PRECISION == 2
#define REAL double
#else
#define REAL float
#endif

__kernel void spmv_dia(__private unsigned int ndiags,
	__constant int* d_ioff,
	__constant REAL* d_diags,
	__constant REAL* d_x,
	__global REAL* dst_y,
	__local int* sharedioff,
	__private unsigned int ioff_offset,
	__private unsigned int diags_offset)
{
	__private unsigned int row_id = get_global_id(0);
	__private unsigned int local_row_id = get_local_id(0);

	__private unsigned int i;
	__private long q;
	__private REAL r;

#pragma unroll(UNROLL_SHARED)
	for (i = local_row_id; i < ndiags; i += WORKGROUP_SIZE)
		sharedioff[i] = d_ioff[ioff_offset + i];
	barrier(CLK_LOCAL_MEM_FENCE);

	if (row_id >= N_MATRIX) return;

	r = 0.0;
#pragma unroll
	for (i = 0; i < ndiags; i++)
	{
		q = sharedioff[i] + row_id;
		if (q >= 0 && q < N_MATRIX)
		{
			r += *(d_diags + diags_offset + row_id + i * STRIDE_MATRIX) * d_x[q];
		}
	}
	dst_y[row_id] += r;
}
