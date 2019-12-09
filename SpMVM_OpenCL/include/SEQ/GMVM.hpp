#ifndef GMVM_SEQ_H
#define GMVM_SEQ_H

#include<compiler_config.h>

#include<vector>
#include<chrono>

#include<IO/mmio.h>
#include<IO/convert_input.h>

// Performs sequential SpMVM for GMVM format
// Returns time.
unsigned long GMVM_sequential(struct mat_t* d_mat, std::vector<REAL> d_x, std::vector<REAL>& dst_y)
{
	typedef std::chrono::high_resolution_clock Clock;
	auto t1 = Clock::now();
	//
	for (IndexType i = 0; i < d_mat->n; i++)
	{
		for (IndexType j = 0; j < d_mat->n; j++)
		{
			dst_y[i] += d_mat->val[i + (j * d_mat->n)] * d_x[j];
		}
	}
	//
	auto t2 = Clock::now();
	//
	return (t2 - t1).count();
}

#endif