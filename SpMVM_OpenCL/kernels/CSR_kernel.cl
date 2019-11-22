// Implementation based off the improved CSR implementation in
// "Efficient sparse matrix-vector multiplication on cache - based GPUs"
// by: István Reguly & Mike Giles
//
/*-------------------------------- Single-precision----------------------------------*/
__kernel void spmv_csr_s(unsigned int n, unsigned int repeat, unsigned int coop,
	__constant unsigned int* d_ia,
	__constant unsigned int* d_ja,
	__constant float* d_a,
	__constant float* d_x,
	__global float* dst_y,
	__local volatile float* shareddata)
{
	__private unsigned int thread_id = get_global_id(0) * repeat / coop;
	__private unsigned int local_row_id = get_local_id(0);
	__private unsigned int coop_id = local_row_id % coop;

	unsigned int i, j, row_ptr;
	unsigned int s;
	float r;

	for (i = 0; i < repeat; i++)
	{
		r = 0;
		if (thread_id < n)
		{
			// do multiplication
			row_ptr = d_ia[thread_id];
			for (j = coop_id; j < d_ia[thread_id + 1] - row_ptr; j += coop)
			{
				r += d_a[row_ptr + j] * d_x[d_ja[row_ptr + j]];
			}
			// do reduction in shared mem
			shareddata[local_row_id] = r;
			for (s = coop / 2; s > 0; s >>= 1)
			{
				if (coop_id < s) shareddata[local_row_id] += shareddata[local_row_id + s];
			}
			if (coop_id == 0) dst_y[thread_id] = shareddata[local_row_id];
			thread_id += get_local_size(0) / coop;
		}
	}
}

/*-------------------------------- double-precision----------------------------------*/
__kernel void spmv_csr_d(unsigned int n, unsigned int repeat, unsigned int coop,
	__constant unsigned int* d_ia,
	__constant unsigned int* d_ja,
	__constant double* d_a,
	__constant double* d_x,
	__global double* dst_y,
	__local volatile double* shareddata)
{
	__private unsigned int thread_id = get_global_id(0) * repeat / coop;
	__private unsigned int local_row_id = get_local_id(0);
	__private unsigned int coop_id = local_row_id % coop;
	
	unsigned int i, j, row_ptr;
	unsigned unsigned int s;
	double r;

	for (i = 0; i < repeat; i++)
	{
		r = 0;
		if (thread_id < n) 
		{
			// do multiplication
			row_ptr = d_ia[thread_id];
			for (j = coop_id; j < d_ia[thread_id + 1] - row_ptr; j += coop)
			{
				r += d_a[row_ptr + j] * d_x[d_ja[row_ptr + j]];
			}
			// do reduction in shared mem
			shareddata[local_row_id] = r;
			for (s = coop / 2; s > 0; s >>= 1) 
			{
				if (coop_id < s) shareddata[local_row_id] += shareddata[local_row_id + s];
			}
			if (coop_id == 0) dst_y[thread_id] = shareddata[local_row_id];
			thread_id += get_local_size(0) / coop;
		}
	}
}