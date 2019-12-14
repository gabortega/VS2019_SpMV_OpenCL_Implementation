#ifndef ELL_SEQ_H
#define ELL_SEQ_H

#include<compiler_config.h>

#include<vector>
#include<chrono>

#include<IO/mmio.h>
#include<IO/convert_input.h>

// Performs sequential SpMVM for ELL format
// Returns time.
unsigned long ELL_sequential(struct ellg_t* d_ell, std::vector<REAL> d_x, std::vector<REAL>& dst_y)
{
	typedef std::chrono::high_resolution_clock Clock;
	auto t1 = Clock::now();
	//
	IndexType nell = d_ell->nell[d_ell->n];
	REAL sum;
	for (IndexType i = 0; i < d_ell->n; i++)
	{
		sum = 0.0;
		for (IndexType j = 0; j < nell; j++)
		{
			IndexType q = i + (j * d_ell->stride);
			sum += d_ell->a[q] * d_x[d_ell->jcoeff[q]];
		}
		dst_y[i] = sum;
	}
	//
	auto t2 = Clock::now();
	//
	return (t2 - t1).count();
}
#endif