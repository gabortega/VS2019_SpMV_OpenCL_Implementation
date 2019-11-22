/*-------------------------------- Single-precision----------------------------------*/
__kernel void spmv_ell_s(unsigned int n, unsigned int nell, unsigned int stride,
	__constant unsigned int* d_jcoeff,
	__constant float* d_a,
	__constant float* d_x,
	__global float* dst_y)
{
	__private unsigned int row_id = get_global_id(0);

	__private unsigned int i, j; 
	__private long p, q;
	__private float r;

	while (row_id < n)
	{
		r = 0.0;
		p = -1;
		for (i = 0; i < nell; i++)
		{
			j = i * stride + row_id;
			q = d_jcoeff[j];
			if (p < q)
				r += d_a[j] * d_x[q];
			else
				i = nell;
		}
		dst_y[row_id] += r;

		row_id += get_global_size(0);
	}
}

__kernel void spmv_ellg_s(unsigned int n, unsigned int stride,
	__constant unsigned int* d_nell,
	__constant unsigned int* d_jcoeff,
	__constant float* d_a,
	__constant float* d_x,
	__global float* dst_y)
{
	__private unsigned int row_id = get_global_id(0);
	__private unsigned int row_nell = d_nell[row_id];

	__private unsigned int i, j;
	__private float r;

	while (row_id < n)
	{
		r = 0.0;
		for (i = 0; i < row_nell; i++)
		{
			j = i * stride + row_id;
			r += d_a[j] * d_x[d_jcoeff[j]];
		}
		dst_y[row_id] += r;

		row_id += get_global_size(0);
		row_nell = d_nell[row_id];
	}
}

__kernel void spmv_hll_s(unsigned int n, unsigned int hacksize,
	__constant unsigned int* d_nell,
	__constant unsigned int* d_jcoeff,
	__constant unsigned unsigned int* d_hoff,
	__constant float* d_a,
	__constant float* d_x,
	__global float* dst_y)
{
	__private unsigned int row_id = get_global_id(0);

	__private unsigned int row_hack_id, row_nell, row_hoff;

	__private unsigned int i, j; 
	__private long p, q;
	__private float r;

	while (row_id < n)
	{
		row_hack_id = row_id / hacksize;
		row_nell = d_nell[row_hack_id];
		row_hoff = d_hoff[row_hack_id];

		r = 0.0;
		p = -1;
		for (i = 0; i < row_nell; i++)
		{
			j = i * hacksize + (row_id % hacksize) + row_hoff;
			q = d_jcoeff[j];
			if (p < q)
				r += d_a[j] * d_x[q];
			else
				i = row_nell;
		}
		dst_y[row_id] += r;

		row_id += get_global_size(0);
	}
}

__kernel void spmv_hll_local_s(unsigned int n, unsigned int hacksize,
	__constant unsigned int* d_nell,
	__constant unsigned int* d_jcoeff,
	__constant unsigned unsigned int* d_hoff,
	__constant float* d_a,
	__constant float* d_x,
	__global float* dst_y,
	__local unsigned int* sharedhoff)
{
	__private unsigned int row_id = get_global_id(0);
	__private unsigned int row_local_id = get_local_id(0);

	__private unsigned int row_hack_id = row_id / hacksize;
	__private unsigned int sharedhoff_size = get_local_size(0) / hacksize;
	__private unsigned int row_local_hack_id = row_hack_id % sharedhoff_size;

	if (row_local_id < sharedhoff_size)
	{
		sharedhoff[row_local_id] = d_nell[row_hack_id + row_local_id];
		sharedhoff[row_local_id + sharedhoff_size] = d_hoff[row_hack_id + row_local_id];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	__private unsigned int row_nell = sharedhoff[row_local_hack_id];
	__private unsigned int row_hoff = sharedhoff[row_local_hack_id + sharedhoff_size];

	__private unsigned int i, j; 
	__private long p, q;
	__private float r;

	while (row_id < n)
	{
		r = 0.0;
		p = -1;
		for (i = 0; i < row_nell; i++)
		{
			j = i * hacksize + (row_id % hacksize) + row_hoff;
			q = d_jcoeff[j];
			if (p < q)
				r += d_a[j] * d_x[q];
			else
				i = row_nell;
		}
		dst_y[row_id] += r;

		row_id += get_global_size(0);
		row_hack_id = row_id / hacksize;
		row_local_hack_id = row_hack_id % sharedhoff_size;

		if (row_local_id < sharedhoff_size)
		{
			sharedhoff[row_local_id] = d_nell[row_hack_id + row_local_id];
			sharedhoff[row_local_id + sharedhoff_size] = d_hoff[row_hack_id + row_local_id];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		row_nell = d_nell[row_hack_id];
		row_hoff = d_hoff[row_hack_id];
	}
}

/*-------------------------------- Double-precision----------------------------------*/
__kernel void spmv_ell_d(unsigned int n, unsigned int nell, unsigned int stride,
	__constant unsigned int* d_jcoeff,
	__constant double* d_a,
	__constant double* d_x,
	__global double* dst_y)
{
	__private unsigned int row_id = get_global_id(0);

	__private unsigned int i, j; 
	__private long p, q;
	__private double r;

	while (row_id < n)
	{
		r = 0.0;
		p = -1;
		for (i = 0; i < nell; i++)
		{
			j = i * stride + row_id;
			q = d_jcoeff[j];
			if (p < q)
				r += d_a[j] * d_x[q];
			else
				i = nell;
		}
		dst_y[row_id] += r;

		row_id += get_global_size(0);
	}
}

__kernel void spmv_ellg_d(unsigned int n, unsigned int stride,
	__constant unsigned int* d_nell,
	__constant unsigned int* d_jcoeff,
	__constant double* d_a,
	__constant double* d_x,
	__global double* dst_y)
{
	__private unsigned int row_id = get_global_id(0);
	__private unsigned int row_nell = d_nell[row_id];

	__private unsigned int i, j;
	__private double r;

	while (row_id < n)
	{
		r = 0.0;
		for (i = 0; i < row_nell; i++)
		{
			j = i * stride + row_id;
			r += d_a[j] * d_x[d_jcoeff[j]];
		}
		dst_y[row_id] += r;

		row_id += get_global_size(0);
		row_nell = d_nell[row_id];
	}
}

__kernel void spmv_hll_d(unsigned int n, unsigned int hacksize,
	__constant unsigned int* d_nell,
	__constant unsigned int* d_jcoeff,
	__constant unsigned unsigned int* d_hoff,
	__constant double* d_a,
	__constant double* d_x,
	__global double* dst_y)
{
	__private unsigned int row_id = get_global_id(0);

	__private unsigned int row_hack_id, row_nell, row_hoff;

	__private unsigned int i, j; 
	__private long p, q;
	__private double r;

	while (row_id < n)
	{
		row_hack_id = row_id / hacksize;
		row_nell = d_nell[row_hack_id];
		row_hoff = d_hoff[row_hack_id];

		r = 0.0;
		p = -1;
		for (i = 0; i < row_nell; i++)
		{
			j = i * hacksize + (row_id % hacksize) + row_hoff;
			q = d_jcoeff[j];
			if (p < q)
				r += d_a[j] * d_x[q];
			else
				i = row_nell;
		}
		dst_y[row_id] += r;

		row_id += get_global_size(0);
	}
}

__kernel void spmv_hll_local_d(unsigned int n, unsigned int hacksize,
	__constant unsigned int* d_nell,
	__constant unsigned int* d_jcoeff,
	__constant unsigned unsigned int* d_hoff,
	__constant double* d_a,
	__constant double* d_x,
	__global double* dst_y,
	__local unsigned int* sharedhoff)
{
	__private unsigned int row_id = get_global_id(0);
	__private unsigned int row_local_id = get_local_id(0);

	__private unsigned int row_hack_id = row_id / hacksize;
	__private unsigned int sharedhoff_size = get_local_size(0) / hacksize;
	__private unsigned int row_local_hack_id = row_hack_id % sharedhoff_size;

	if (row_local_id < sharedhoff_size)
	{
		sharedhoff[row_local_id] = d_nell[row_hack_id + row_local_id];
		sharedhoff[row_local_id + sharedhoff_size] = d_hoff[row_hack_id + row_local_id];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	__private unsigned int row_nell = sharedhoff[row_local_hack_id];
	__private unsigned int row_hoff = sharedhoff[row_local_hack_id + sharedhoff_size];

	__private unsigned int i, j; 
	__private long p, q;
	__private double r;

	while (row_id < n)
	{
		r = 0.0;
		p = -1;
		for (i = 0; i < row_nell; i++)
		{
			j = i * hacksize + (row_id % hacksize) + row_hoff;
			q = d_jcoeff[j];
			if (p < q)
				r += d_a[j] * d_x[q];
			else
				i = row_nell;
		}
		dst_y[row_id] += r;

		row_id += get_global_size(0);
		row_hack_id = row_id / hacksize;
		row_local_hack_id = row_hack_id % sharedhoff_size;

		if (row_local_id < sharedhoff_size)
		{
			sharedhoff[row_local_id] = d_nell[row_hack_id + row_local_id];
			sharedhoff[row_local_id + sharedhoff_size] = d_hoff[row_hack_id + row_local_id];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		row_nell = d_nell[row_hack_id];
		row_hoff = d_hoff[row_hack_id];
	}
}