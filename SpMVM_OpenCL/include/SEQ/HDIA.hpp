#ifndef HDIA_SEQ_H
#define HDIA_SEQ_H

#include<compiler_config.h>

#include<vector>
#include<chrono>

#include<IO/mmio.h>
#include<IO/convert_input.h>

// Performs sequential SpMVM for HDIA format
// Returns time.
unsigned long HDIA_sequential(struct hdia_t* d_hdia, std::vector<REAL> d_x, std::vector<REAL>& dst_y)
{
	typedef std::chrono::high_resolution_clock Clock;
	auto t1 = Clock::now();
	//
	IndexType row_hack_id, row_memoff, ndiags, row_hoff;
	//
	REAL sum;
	for (IndexType i = 0; i < d_hdia->n; i++)
	{
		sum = 0.0;
		row_hack_id = i / HDIA_HACKSIZE;
		row_memoff = d_hdia->memoff[row_hack_id];
		ndiags = d_hdia->ndiags[row_hack_id];
		row_hoff = d_hdia->hoff[row_hack_id];

		for (IndexType j = 0; j < ndiags; j++)
		{
			long q = d_hdia->ioff[row_hoff + j] + i;
			if (q >= 0 && q < d_hdia->n)
				sum += d_hdia->diags[row_memoff + i + (j * d_hdia->stride)] * d_x[q];
		}
		dst_y[i] = sum;
	}
	//
	auto t2 = Clock::now();
	//
	return (t2 - t1).count();
}
#endif