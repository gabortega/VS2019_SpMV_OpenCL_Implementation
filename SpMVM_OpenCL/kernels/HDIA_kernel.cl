#if PRECISION == 2
#define REAL double
#else
#define REAL float
#endif

__kernel void spmv_hdia(__private unsigned int ndiags,
	__constant unsigned int* d_ndiags,
	__constant int* d_ioff,
	__constant REAL* d_diags,
	__constant unsigned int* d_hoff,
	__constant unsigned int* d_memoff,
	__constant REAL* d_x,
	__global REAL* dst_y,
	__local int* sharedioff,
	__private unsigned int ioff_offset,
	__private unsigned int diags_offset)
{
	__private unsigned int row_id = get_global_id(0);
	__private unsigned int local_row_id = get_local_id(0);
	__private unsigned int local_hack_id = local_row_id % HACKSIZE;
	__private unsigned int row_hack_id = row_id / HACKSIZE;

	__private unsigned int row_ndiag, local_row_offset, row_memoff, row_hoff, shared_step;

	__private unsigned int i;
	__private long p, q;
	__private REAL r;

	if (row_hack_id >= NHOFF) return;

	row_ndiag = d_ndiags[row_hack_id];

	if (ioff_offset >= row_ndiag) return;

	row_ndiag = min(row_ndiag - ioff_offset, (unsigned int)MAX_NDIAG);
	local_row_offset = (local_row_id / HACKSIZE) * MAX_NDIAG;
	row_hoff = d_hoff[row_hack_id];
	row_memoff = d_memoff[row_hack_id];
	shared_step = min(HACKSIZE, WORKGROUP_SIZE);

#pragma unroll(UNROLL_SHARED)
	for (i = local_hack_id; i < row_ndiag; i += shared_step)
		sharedioff[local_row_offset + i] = d_ioff[row_hoff + ioff_offset + i];
	barrier(CLK_LOCAL_MEM_FENCE);

	if (row_id >= N_MATRIX) return;

	r = 0.0;
	for (i = 0; i < row_ndiag; i++)
	{
		q = sharedioff[local_row_offset + i] + row_id;
		if (q >= 0 && q < N_MATRIX)
		{
			r += *(d_diags + diags_offset + row_memoff + row_id + i * HACKSIZE) * d_x[q];
		}
	}
	dst_y[row_id] += r;
}
