// Implementation based off CUDA_ITSOL SpMV by: Ruipeng Li, Yousef Saad
// URL: https://www-users.cs.umn.edu/~saad/software/CUDA_ITSOL/CUDA_ITSOL.tar.gz
//
#if PRECISION == 2
/*-------------------------------- Double-precision----------------------------------*/
__kernel void spmv_jad(unsigned int njad,
	__constant unsigned int* d_ia,
	__constant unsigned int* d_ja,
	__constant double* d_a,
	__constant unsigned int* d_perm,
	__constant double* d_x,
	__global double* dst_y,
	__local unsigned int* sharedia,
	unsigned int ia_offset)
{
	__private unsigned int row_id = get_global_id(0);
	__private unsigned int local_row_id = get_local_id(0);

	__private unsigned int i, j;
	__private long p, q;
	__private double r;

#pragma unroll
	for (i = local_row_id; i < njad + 1; i += WORKGROUP_SIZE)
		sharedia[i] = d_ia[ia_offset + i];
	barrier(CLK_LOCAL_MEM_FENCE);

	if (row_id >= N_MATRIX) return;

	r = 0.0;
	p = sharedia[0];
	q = sharedia[1];
	i = 0;
#pragma unroll
	while (((p + row_id) < q) && (i < njad))
	{
		j = p + row_id;
		r += d_a[j] * d_x[d_ja[j]];
		i++;
		if (i < njad)
		{
			p = q;
			q = sharedia[i + 1];
		}
	}
	dst_y[d_perm[row_id]] += r;
}

#else
/*-------------------------------- Single-precision----------------------------------*/
__kernel void spmv_jad(unsigned int njad,
	__constant unsigned int* d_ia,
	__constant unsigned int* d_ja,
	__constant float* d_a,
	__constant unsigned int* d_perm,
	__constant float* d_x,
	__global float* dst_y,
	__local unsigned int* sharedia,
	unsigned int ia_offset)
{
	__private unsigned int row_id = get_global_id(0);
	__private unsigned int local_row_id = get_local_id(0);

	__private unsigned int i, j;
	__private long p, q;
	__private float r;

#pragma unroll
	for (i = local_row_id; i < njad + 1; i += WORKGROUP_SIZE)
		sharedia[i] = d_ia[ia_offset + i];
	barrier(CLK_LOCAL_MEM_FENCE);

	if (row_id >= N_MATRIX) return;

	r = 0.0;
	p = sharedia[0];
	q = sharedia[1];
	i = 0;
#pragma unroll
	while (((p + row_id) < q) && (i < njad))
	{
		j = p + row_id;
		r += d_a[j] * d_x[d_ja[j]];
		i++;
		if (i < njad)
		{
			p = q;
			q = sharedia[i + 1];
		}
	}
	dst_y[d_perm[row_id]] += r;
}
#endif