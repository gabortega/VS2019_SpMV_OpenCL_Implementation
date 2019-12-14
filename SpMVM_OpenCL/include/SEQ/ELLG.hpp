#ifndef ELLG_SEQ_H
#define ELLG_SEQ_H

#include<compiler_config.h>

#include<vector>
#include<chrono>

#include<IO/mmio.h>
#include<IO/convert_input.h>

// Performs sequential SpMVM for ELLG format
// Returns time.
unsigned long ELLG_sequential(struct ellg_t* d_ellg, std::vector<REAL> d_x, std::vector<REAL>& dst_y)
{
	typedef std::chrono::high_resolution_clock Clock;
	auto t1 = Clock::now();
	//
	REAL sum;
	for (IndexType i = 0; i < d_ellg->n; i++)
	{
		sum = 0.0;
		for (IndexType j = 0; j < d_ellg->nell[i]; j++)
		{
			IndexType q = i + (j * d_ellg->stride);
			sum += d_ellg->a[q] * d_x[d_ellg->jcoeff[q]];
		}
		dst_y[i] = sum;
	}
	//
	auto t2 = Clock::now();
	//
	return (t2 - t1).count();
}
#endif