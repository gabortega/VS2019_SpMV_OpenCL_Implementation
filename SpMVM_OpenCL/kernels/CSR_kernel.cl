// Implementation based off the improved CSR implementation in
// "Efficient sparse matrix-vector multiplication on cache - based GPUs"
// by: István Reguly & Mike Giles
//
#if PRECISION == 2
#define REAL double
#else
#define REAL float
#endif

__kernel void spmv_csr(
#if USE_CONSTANT_MEM
	__constant unsigned int* d_ia,
	__constant unsigned int* d_ja,
	__constant REAL* d_a,
	__constant REAL* d_x,
#else
	__global unsigned int* d_ia,
	__global unsigned int* d_ja,
	__global REAL* d_a,
	__global REAL* d_x,
#endif
	__global REAL* dst_y,
	__local volatile REAL* shareddata)
{
	__private unsigned int thread_id = (get_local_id(0) + (get_group_id(0) * WORKGROUP_SIZE * CSR_REPEAT)) / CSR_COOP;
	__private unsigned int local_row_id = get_local_id(0);
	__private unsigned int coop_id = local_row_id % CSR_COOP;
	
	unsigned int i, j, row_ptr;
	unsigned int s;
	REAL r;

	for (i = 0; i < CSR_REPEAT; i++)
	{
		r = 0;
		if (thread_id < N_MATRIX) 
		{
			// do multiplication
			row_ptr = d_ia[thread_id];
#pragma unroll(1)
			for (j = coop_id; j < d_ia[thread_id + 1] - row_ptr; j += CSR_COOP)
			{
				r += d_a[row_ptr + j] * d_x[d_ja[row_ptr + j]];
			}
			// do reduction in shared mem
			shareddata[local_row_id] = r;
#pragma unroll(UNROLL_SHARED)
			for (s = CSR_COOP / 2; s > 0; s >>= 1) 
			{
				if (coop_id < s) shareddata[local_row_id] += shareddata[local_row_id + s];
			}
			if (coop_id == 0) dst_y[thread_id] += shareddata[local_row_id];
			thread_id += WORKGROUP_SIZE / CSR_COOP;
		}
	}
}

// OCCUPANCY TEST
__kernel void occ_spmv_csr(
#if USE_CONSTANT_MEM
	__constant unsigned int* d_ia,
	__constant unsigned int* d_ja,
	__constant REAL* d_a,
	__constant REAL* d_x,
#else
	__global unsigned int* d_ia,
	__global unsigned int* d_ja,
	__global REAL* d_a,
	__global REAL* d_x,
#endif
	__global REAL* dst_y,
	__local volatile REAL* shareddata)
{
	__private unsigned int thread_id = ((get_local_id(0) + (get_group_id(0) * WORKGROUP_SIZE * CSR_REPEAT)) / CSR_COOP) % N_MATRIX;
	__private unsigned int true_thread_id = (get_local_id(0) + (get_group_id(0) * WORKGROUP_SIZE * CSR_REPEAT)) / CSR_COOP;
	__private unsigned int local_row_id = get_local_id(0);
	__private unsigned int coop_id = local_row_id % CSR_COOP;

	unsigned int i, j, row_ptr;
	unsigned int s;
	REAL r;

	for (i = 0; i < CSR_REPEAT; i++)
	{
		r = 0;
		if (thread_id < N_MATRIX)
		{
			// do multiplication
			row_ptr = d_ia[thread_id];
#pragma unroll(1)
			for (j = coop_id; j < d_ia[thread_id + 1] - row_ptr; j += CSR_COOP)
			{
				r += d_a[row_ptr + j] * d_x[d_ja[row_ptr + j]];
			}
			// do reduction in shared mem
			shareddata[local_row_id] = r;
#pragma unroll(UNROLL_SHARED)
			for (s = CSR_COOP / 2; s > 0; s >>= 1)
			{
				if (coop_id < s) shareddata[local_row_id] += shareddata[local_row_id + s];
			}
			if (coop_id == 0 && true_thread_id < N_MATRIX) dst_y[thread_id] += shareddata[local_row_id];
			thread_id += WORKGROUP_SIZE / CSR_COOP;
		}
	}
}
