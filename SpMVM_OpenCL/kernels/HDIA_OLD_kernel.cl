#if PRECISION == 2
#define REAL double
#else
#define REAL float
#endif

__kernel void spmv_hdia(
#if USE_CONSTANT_MEM
	__constant unsigned int* d_ndiags,
	__constant int* d_ioff,
	__constant REAL* d_diags,
	__constant unsigned int* d_hoff,
	__constant unsigned int* d_memoff,
	__constant REAL* d_x,
#else
	__global unsigned int* d_ndiags,
	__global int* d_ioff,
	__global REAL* d_diags,
	__global unsigned int* d_hoff,
	__global unsigned int* d_memoff,
	__global REAL* d_x,
#endif
#if OVERRIDE_MEM
	__global REAL* dst_y,
	__local unsigned int* dummy_mem)
#else
	__global REAL* dst_y)
#endif
{
	__private unsigned int row_id = get_global_id(0);

	__private unsigned int row_hack_id, row_memoff, ndiags, row_hoff;

	__private unsigned int i;
	__private long q;
	__private REAL r;

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

// OCCUPANCY TEST
__kernel void occ_spmv_hdia(
#if USE_CONSTANT_MEM
	__constant unsigned int* d_ndiags,
	__constant int* d_ioff,
	__constant REAL* d_diags,
	__constant unsigned int* d_hoff,
	__constant unsigned int* d_memoff,
	__constant REAL* d_x,
#else
	__global unsigned int* d_ndiags,
	__global int* d_ioff,
	__global REAL* d_diags,
	__global unsigned int* d_hoff,
	__global unsigned int* d_memoff,
	__global REAL* d_x,
#endif
#if OVERRIDE_MEM
	__global REAL* dst_y,
	__local unsigned int* dummy_mem)
#else
	__global REAL* dst_y)
#endif
{
	__private unsigned int row_id = get_global_id(0) % N_MATRIX;

	__private unsigned int row_hack_id, row_memoff, ndiags, row_hoff;

	__private unsigned int i;
	__private long q;
	__private REAL r;

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
	if (get_global_id(0) < N_MATRIX)
		dst_y[row_id] = r;
}
