#ifndef JAD_SEQ_H
#define JAD_SEQ_H

#include<compiler_config.h>

#include<vector>
#include<chrono>
#include<iostream>

#include<IO/mmio.h>
#include<IO/convert_input.h>

// Performs sequential SpMVM for JAD format
// Returns time.
unsigned long JAD_sequential(struct jad_t* d_jad, std::vector<REAL> d_x, std::vector<REAL>& dst_y)
{
	typedef std::chrono::high_resolution_clock Clock;
	auto t1 = Clock::now();	
	//
	for (IndexType i = 0, j = 0; j < d_jad->njad[d_jad->n]; j++)
	{
		IndexType p = d_jad->ia[j], q = d_jad->ia[j + 1];
		for (i = 0; ((i + p) < q) && (i < d_jad->n); i++)
		{
			dst_y[d_jad->perm[i]] += d_jad->a[i + p] * d_x[d_jad->ja[i + p]];
		}
	}
	//
	auto t2 = Clock::now();
	//
	return (t2 - t1).count();
}
#endif