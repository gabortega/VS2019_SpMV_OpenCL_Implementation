// Implementation based off CUDA_ITSOL SpMV by: Ruipeng Li, Yousef Saad
// URL: https://www-users.cs.umn.edu/~saad/software/CUDA_ITSOL/CUDA_ITSOL.tar.gz
//
/*-------------------------------- Single-precision----------------------------------*/
__kernel void spmv_dia_s(unsigned int n, unsigned int ndiags, int stride,
	__constant int* d_ioff,
	__constant float* d_diags,
	__constant float* d_x,
	__global float* dst_y,
	__local unsigned int* sharedioff,
	unsigned int ioff_offset,
	unsigned int diags_offset)
{
	__private unsigned int row_id = get_global_id(0);
	__private int local_row_id = get_local_id(0);
	__private int workgroup_size = get_local_size(0);

	__private unsigned int i; 
	__private int q;
	__private float r;

	for (i = local_row_id; i < ndiags; i += workgroup_size)
		sharedioff[i] = d_ioff[ioff_offset + i];
	barrier(CLK_LOCAL_MEM_FENCE);

	while (row_id < n)
	{
		r = 0.0;
		for (i = 0; i < ndiags; i++)
		{
			q = sharedioff[i] + row_id;
			if (q >= 0 && q < n)
			{
				r += *(d_diags + diags_offset + row_id + i * stride) * d_x[q];
			}
		}
		dst_y[row_id] += r;

		row_id += get_global_size(0);
	}
}

__kernel void spmv_hdia_s(unsigned int n, unsigned int nhoff, unsigned int hacksize,
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
	__private int q;
	__private float r;

	while (row_id < n)
	{
		row_hack_id = row_id / hacksize;
		row_memoff = d_memoff[row_hack_id];
		ndiags = d_ndiags[row_hack_id];
		row_hoff = d_hoff[row_hack_id];

		r = 0.0;
		for (i = 0; i < ndiags; i++) 
		{
			q = d_ioff[row_hoff + i] + row_id;
			if (q >= 0 && q < n) 
			{
				r += *(d_diags + row_memoff + row_id + i * hacksize) * d_x[q];
			}
		}
		dst_y[row_id] += r;

		row_id += get_global_size(0);
	}
}

__kernel void spmv_hdia_local_s(unsigned int n, unsigned int nhoff, unsigned int hacksize,
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

	__private int row_hack_id = row_id / hacksize;
	__private int sharedhoff_size = get_local_size(0) / hacksize;
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
	__private int q;
	__private float r;

	while (row_id < n)
	{
		r = 0.0;
		for (i = 0; i < ndiags; i++) 
		{
			q = d_ioff[row_hoff + i] + row_id;
			if (q >= 0 && q < n) 
			{
				r += *(d_diags + row_memoff + row_id + i * hacksize) * d_x[q];
			}
		}
		dst_y[row_id] += r;

		row_id += get_global_size(0);
		row_hack_id = row_id / hacksize;
		row_local_hack_id = row_hack_id % sharedhoff_size;

		if (row_local_id < sharedhoff_size)
		{
			sharedhoff[row_local_id] = d_memoff[row_hack_id + row_local_id];
			sharedhoff[row_local_id + sharedhoff_size] = d_ndiags[row_hack_id + row_local_id];
			sharedhoff[row_local_id + 2 * sharedhoff_size] = d_hoff[row_hack_id + row_local_id];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		row_memoff = sharedhoff[row_local_hack_id];
		ndiags = sharedhoff[row_local_hack_id + sharedhoff_size];
		row_hoff = sharedhoff[row_local_hack_id + 2 * sharedhoff_size];
	}
}

/*-------------------------------- Double-precision----------------------------------*/
__kernel void spmv_dia_d(unsigned int n, unsigned int ndiags, int stride,
	__constant int* d_ioff,
	__constant double* d_diags,
	__constant double* d_x,
	__global double* dst_y,
	__local unsigned int* sharedioff,
	unsigned int ioff_offset,
	int diags_offset)
{
	__private unsigned int row_id = get_global_id(0);
	__private int local_row_id = get_local_id(0);
	__private int workgroup_size = get_local_size(0);

	__private unsigned int i; 
	__private int q;
	__private double r;

	for (i = local_row_id; i < ndiags; i += workgroup_size)
		sharedioff[i] = d_ioff[ioff_offset + i];
	barrier(CLK_LOCAL_MEM_FENCE);

	while (row_id < n)
	{
		r = 0.0;
		for (i = 0; i < ndiags; i++)
		{
			q = sharedioff[i] + row_id;
			if (q >= 0 && q < n)
			{
				r += *(d_diags + diags_offset + row_id + i * stride) * d_x[q];
			}
		}
		dst_y[row_id] += r;

		row_id += get_global_size(0);
	}
}

__kernel void spmv_hdia_d(unsigned int n, unsigned int nhoff, unsigned int hacksize,
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
	__private int q;
	__private double r;

	while (row_id < n)
	{
		row_hack_id = row_id / hacksize;
		row_memoff = d_memoff[row_hack_id];
		ndiags = d_ndiags[row_hack_id];
		row_hoff = d_hoff[row_hack_id];

		r = 0.0;
		for (i = 0; i < ndiags; i++)
		{
			q = d_ioff[row_hoff + i] + row_id;
			if (q >= 0 && q < n)
			{
				r += *(d_diags + row_memoff + row_id + i * hacksize) * d_x[q];
			}
		}
		dst_y[row_id] += r;

		row_id += get_global_size(0);
	}
}

__kernel void spmv_hdia_local_d(unsigned int n, unsigned int nhoff, unsigned int hacksize,
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

	__private int row_hack_id = row_id / hacksize;
	__private int sharedhoff_size = get_local_size(0) / hacksize;
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
	__private int q;
	__private double r;

	while (row_id < n)
	{
		r = 0.0;
		for (i = 0; i < ndiags; i++)
		{
			q = d_ioff[row_hoff + i] + row_id;
			if (q >= 0 && q < n)
			{
				r += *(d_diags + row_memoff + row_id + i * hacksize) * d_x[q];
			}
		}
		dst_y[row_id] += r;

		row_id += get_global_size(0);
		row_hack_id = row_id / hacksize;
		row_local_hack_id = row_hack_id % sharedhoff_size;

		if (row_local_id < sharedhoff_size)
		{
			sharedhoff[row_local_id] = d_memoff[row_hack_id + row_local_id];
			sharedhoff[row_local_id + sharedhoff_size] = d_ndiags[row_hack_id + row_local_id];
			sharedhoff[row_local_id + 2 * sharedhoff_size] = d_hoff[row_hack_id + row_local_id];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		row_memoff = sharedhoff[row_local_hack_id];
		ndiags = sharedhoff[row_local_hack_id + sharedhoff_size];
		row_hoff = sharedhoff[row_local_hack_id + 2 * sharedhoff_size];
	}
}