#ifndef CSR_SEQ_H
#define CSR_SEQ_H

#include<compiler_config.h>

#include<vector>
#include<chrono>

#include<IO/mmio.h>
#include<IO/convert_input.h>

// Performs sequential SpMVM for CSR format
// Returns time.
unsigned long CSR_sequential(struct csr_t* d_csr, std::vector<REAL> d_x, std::vector<REAL>& dst_y)
{
	typedef std::chrono::high_resolution_clock Clock;
	auto t1 = Clock::now();
	//
	IndexType curr_row = -1;
	//
	for (IndexType i = 0; i < d_csr->nnz; i++)
	{
		while (i == d_csr->ia[curr_row + 1])
			curr_row++;
		dst_y[curr_row] += d_csr->a[i] * d_x[d_csr->ja[i]];
	}
	//
	auto t2 = Clock::now();
	//
	return (t2 - t1).count();
}
#endif