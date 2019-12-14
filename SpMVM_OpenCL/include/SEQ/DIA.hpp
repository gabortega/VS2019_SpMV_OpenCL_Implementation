#ifndef DIA_SEQ_H
#define DIA_SEQ_H

#include<compiler_config.h>

#include<vector>
#include<chrono>

#include<IO/mmio.h>
#include<IO/convert_input.h>

// Performs sequential SpMVM for DIA format
// Returns time.
unsigned long DIA_sequential(struct dia_t* d_dia, std::vector<REAL> d_x, std::vector<REAL>& dst_y)
{
	typedef std::chrono::high_resolution_clock Clock;
	auto t1 = Clock::now();
	//
	REAL sum;
	for (IndexType i = 0; i < d_dia->n; i++)
	{
		sum = 0.0;
		for (IndexType j = 0; j < d_dia->ndiags; j++)
		{
			long q = i + d_dia->ioff[j];
			if (q >= 0 && q < d_dia->n)
				sum += d_dia->diags[i + (j * d_dia->stride)] * d_x[q];
		}
		dst_y[i] = sum;
	}
	//
	auto t2 = Clock::now();
	//
	return (t2 - t1).count();
}
#endif