#ifndef COO_SEQ_H
#define COO_SEQ_H

#include<compiler_config.h>

#include<vector>
#include<chrono>

#include<IO/mmio.h>
#include<IO/convert_input.h>

// Performs sequential SpMVM for COO format
// Returns time.
unsigned long COO_sequential(struct coo_t* d_coo, std::vector<REAL> d_x, std::vector<REAL> &dst_y)
{
	typedef std::chrono::high_resolution_clock Clock;
	auto t1 = Clock::now();
	//
	for (IndexType i = 0; i < d_coo->nnz; i++)
	{
		dst_y[d_coo->ir[i]] += d_coo->val[i] * d_x[d_coo->jc[i]];
	}
	//
	auto t2 = Clock::now();
	//
	return (t2 - t1).count();
}

#endif