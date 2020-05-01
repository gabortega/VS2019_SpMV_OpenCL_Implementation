#if PRECISION == 2
#define REAL double
#else
#define REAL float
#endif

__kernel void spmv_hll(
#if USE_CONSTANT_MEM
	__constant unsigned int* d_nell,
	__constant unsigned int* d_jcoeff,
	__constant unsigned int* d_hoff,
	__constant REAL* d_a,
	__constant REAL* d_x,
#else
	__global unsigned int* d_nell,
	__global unsigned int* d_jcoeff,
	__global unsigned int* d_hoff,
	__global REAL* d_a,
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

	__private unsigned int row_hack_id, row_nell, row_hoff, hack_row;

	__private unsigned int i, j, k;
	__private REAL r;

	if (row_id >= N_MATRIX) return;

	row_hack_id = row_id / HACKSIZE;
	row_nell = d_nell[row_hack_id];
	row_hoff = d_hoff[row_hack_id];
	hack_row = row_id % HACKSIZE;

	r = 0.0;
#pragma unroll(UNROLL)
	for (i = 0; i < row_nell; i++)
	{
		j = i * HACKSIZE + hack_row + row_hoff;
		r += d_a[j] * d_x[d_jcoeff[j]];
	}
	dst_y[row_id] = r;
}

// OCCUPANCY TEST
__kernel void occ_spmv_hll(
#if USE_CONSTANT_MEM
	__constant unsigned int* d_nell,
	__constant unsigned int* d_jcoeff,
	__constant unsigned int* d_hoff,
	__constant REAL* d_a,
	__constant REAL* d_x,
#else
	__global unsigned int* d_nell,
	__global unsigned int* d_jcoeff,
	__global unsigned int* d_hoff,
	__global REAL* d_a,
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

	__private unsigned int row_hack_id, row_nell, row_hoff, hack_row;

	__private unsigned int i, j, k;
	__private REAL r;

	if (row_id >= N_MATRIX) return;

	row_hack_id = row_id / HACKSIZE;
	row_nell = d_nell[row_hack_id];
	row_hoff = d_hoff[row_hack_id];
	hack_row = row_id % HACKSIZE;

	r = 0.0;
#pragma unroll(UNROLL)
	for (i = 0; i < row_nell; i++)
	{
		j = i * HACKSIZE + hack_row + row_hoff;
		r += d_a[j] * d_x[d_jcoeff[j]];
	}
	if (get_global_id(0) < N_MATRIX)
		dst_y[row_id] = r;
}
