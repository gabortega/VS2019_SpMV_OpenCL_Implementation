#ifndef HLL_SEQ_H
#define HLL_SEQ_H

#include<compiler_config.h>

#include<vector>
#include<chrono>

#include<IO/mmio.h>
#include<IO/convert_input.h>

// Performs sequential SpMVM for HLL format
// Returns time.
unsigned long HLL_sequential(struct hll_t* d_hll, std::vector<REAL> d_x, std::vector<REAL>& dst_y)
{
	typedef std::chrono::high_resolution_clock Clock;
	auto t1 = Clock::now();
	//
	IndexType row_hack_id, row_nell, row_hoff;
	//
	for (IndexType i = 0; i < d_hll->n; i++)
	{
		row_hack_id = i / HLL_HACKSIZE;
		row_nell = d_hll->nell[row_hack_id];
		row_hoff = d_hll->hoff[row_hack_id];

		for (IndexType j = 0; j < row_nell; j++)
		{
			IndexType q = j * HLL_HACKSIZE + (i % HLL_HACKSIZE) + row_hoff;
			dst_y[i] += d_hll->a[q] * d_x[d_hll->jcoeff[q]];
		}
	}
	//
	auto t2 = Clock::now();
	//
	return (t2 - t1).count();
}
#endif