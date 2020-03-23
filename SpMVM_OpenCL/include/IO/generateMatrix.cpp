#include "generateMatrix.hpp"

void generateMatrixGaussMethodRow(long n, float row_mean, float row_stddev, struct coo_rand_t* coo, bool flip, bool zigzag)
{
	coo->n = n;
	/*-------- Allocate mem for COO */
	coo->ir = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->jc = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->val = (float*)malloc(n * n * sizeof(float));
	/*-------- Generate normal dist. & randomness */
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_real_distribution<> dist_float(-9.9999999999999, 9.9999999999999);
	std::uniform_int_distribution<> dist_exp(-10, 10);
	std::uniform_int_distribution<> dist_row_chance(0, n);
	std::normal_distribution<float> distribution(row_mean, row_stddev);
	/*-------- Start/End Params */
	long start = (flip) ? -(n - 1) : 0;
	long start_flipped = (!flip) ? -(n - 1) : 0;
	long end = (flip) ? 1 : n;
	long end_flipped = (!flip) ? 1 : n;
	/*-------- Create matrix */
	long coo_index = 0;
	for (long i = start; i < end; i++)
	{
		float dist_row = distribution(generator);
		if (dist_row > 0)
		{
			for (long j = ((zigzag && i % 2 != 0) ? start_flipped : start); j < ((zigzag && i % 2 != 0) ? end_flipped : end); j++)
			{
				long roll_chance = (n + dist_row - 1) / dist_row;
				if (dist_row_chance(generator) / roll_chance >= 1)
				{
					coo->ir[coo_index] = abs(i) + 1;
					coo->jc[coo_index] = abs(j) + 1;
					coo->val[coo_index] = (float)dist_float(generator) * (float)pow(10.0, dist_exp(generator));
					coo_index++;
					dist_row--;
				}
			}
		}
	}
	coo->nnz = coo_index;
	coo->ir = (IndexType*)realloc(coo->ir, coo_index * sizeof(IndexType));
	coo->jc = (IndexType*)realloc(coo->jc, coo_index * sizeof(IndexType));
	coo->val = (float*)realloc(coo->val, coo_index * sizeof(float));
}

void generateMatrixGaussMethodCol(long n, float col_mean, float col_stddev, struct coo_rand_t* coo, bool flip, bool zigzag)
{
	coo->n = n;
	/*-------- Allocate mem for COO */
	coo->ir = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->jc = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->val = (float*)malloc(n * n * sizeof(float));
	/*-------- Generate normal dist. & randomness */
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_real_distribution<> dist_float(-9.9999999999999, 9.9999999999999);
	std::uniform_int_distribution<> dist_exp(-10, 10);
	std::uniform_int_distribution<> dist_col_chance(0, n);
	std::normal_distribution<float> distribution(col_mean, col_stddev);
	/*-------- Start/End Params */
	long start = (flip) ? -(n - 1) : 0;
	long start_flipped = (!flip) ? -(n - 1) : 0;
	long end = (flip) ? 1 : n;
	long end_flipped = (!flip) ? 1 : n;
	/*-------- Create matrix */
	long coo_index = 0;
	for (long j = start; j < end; j++)
	{
		float dist_col = distribution(generator);
		if (dist_col > 0)
		{
			for (long i = ((zigzag && j % 2 != 0) ? start_flipped : start); i < ((zigzag && j % 2 != 0) ? end_flipped : end); i++)
			{
				long roll_chance = (n + dist_col - 1) / dist_col;
				if (dist_col_chance(generator) / roll_chance >= 1)
				{
					coo->ir[coo_index] = abs(i) + 1;
					coo->jc[coo_index] = abs(j) + 1;
					coo->val[coo_index] = (float)dist_float(generator) * (float)pow(10.0, dist_exp(generator));
					coo_index++;
					dist_col--;
				}
			}
		}
	}
	coo->nnz = coo_index;
	coo->ir = (IndexType*)realloc(coo->ir, coo_index * sizeof(IndexType));
	coo->jc = (IndexType*)realloc(coo->jc, coo_index * sizeof(IndexType));
	coo->val = (float*)realloc(coo->val, coo_index * sizeof(float));
}

void generateMatrixGaussMethodFull(long n, float row_mean, float row_stddev, float col_mean, float col_stddev, struct coo_rand_t* coo, bool flip, bool zigzag)
{
	coo->n = n;
	/*-------- Allocate mem for COO */
	coo->ir = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->jc = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->val = (float*)malloc(n * n * sizeof(float));
	/*-------- Generate normal dist. & randomness */
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_real_distribution<> dist_float(-9.9999999999999, 9.9999999999999);
	std::uniform_int_distribution<> dist_exp(-10, 10);
	std::uniform_int_distribution<> dist_slot_chance(0, n);
	std::normal_distribution<float> distribution_row(row_mean, row_stddev);
	std::normal_distribution<float> distribution_col(col_mean, col_stddev);
	/*-------- Start/End Params */
	long start = (flip) ? -(n - 1) : 0;
	long start_flipped = (!flip) ? -(n - 1) : 0;
	long end = (flip) ? 1 : n;
	long end_flipped = (!flip) ? 1 : n;
	/*-------- for each row & column*/
	float* row_mod_array = (float*)malloc(n * sizeof(float));
	float* col_mod_array = (float*)malloc(n * sizeof(float));
	for (long k = 0; k < n; k++)
	{
		row_mod_array[k] = 0;
		col_mod_array[k] = 0;
	}
	/*-------- Create matrix */
	long coo_index = 0;
	for (long j = start; j < end; j++)
	{
		for (long i = ((zigzag && j % 2 != 0) ? start_flipped : start); i < ((zigzag && j % 2 != 0) ? end_flipped : end); i++)
		{
			long dist_row = distribution_row(generator) - row_mod_array[abs(i)];
			long dist_col = distribution_col(generator) - col_mod_array[abs(j)];
			if (dist_row > 0 && dist_col > 0)
			{
				long row_roll_chance = (n + dist_row - 1) / dist_row;
				long col_roll_chance = (n + dist_col - 1) / dist_col;
				long roll = dist_slot_chance(generator);
				if (roll / (row_roll_chance * col_roll_chance) >= 1)
				{
					coo->ir[coo_index] = abs(i) + 1;
					coo->jc[coo_index] = abs(j) + 1;
					coo->val[coo_index] = (float)dist_float(generator) * (float)pow(10.0, dist_exp(generator));
					coo_index++;
					row_mod_array[abs(i)]++;
					col_mod_array[abs(j)]++;
				}
			}
		}
	}
	coo->nnz = coo_index;
	coo->ir = (IndexType*)realloc(coo->ir, coo_index * sizeof(IndexType));
	coo->jc = (IndexType*)realloc(coo->jc, coo_index * sizeof(IndexType));
	coo->val = (float*)realloc(coo->val, coo_index * sizeof(float));
}

void generateMatrixImbalancedRow(long n, long start, long skip, struct coo_rand_t* coo, bool flip)
{
	coo->n = n;
	/*-------- Allocate mem for COO */
	coo->ir = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->jc = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->val = (float*)malloc(n * n * sizeof(float));
	/*-------- Generate normal dist. & randomness */
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_real_distribution<> dist_float(-9.9999999999999, 9.9999999999999);
	std::uniform_int_distribution<> dist_exp(-10, 10);
	/*-------- Start/End Params */
	long start_row = (flip) ? -(n - start - 1) : start;
	long start_col = (flip) ? -(n - 1) : 0;
	long end = (flip) ? 1 : n;
	/*-------- Create matrix */
	long coo_index = 0;
	for (long j = start_col; j < end; j++)
	{
		for (long i = start_row; i < end; i+=skip)
		{
			coo->ir[coo_index] = abs(i) + 1;
			coo->jc[coo_index] = abs(j) + 1;
			coo->val[coo_index] = (float)dist_float(generator) * (float)pow(10.0, dist_exp(generator));
			coo_index++;
		}
	}
	coo->nnz = coo_index;
	coo->ir = (IndexType*)realloc(coo->ir, coo_index * sizeof(IndexType));
	coo->jc = (IndexType*)realloc(coo->jc, coo_index * sizeof(IndexType));
	coo->val = (float*)realloc(coo->val, coo_index * sizeof(float));
}

void generateMatrixImbalancedCol(long n, long start, long skip, struct coo_rand_t* coo, bool flip, bool zigzag)
{
	coo->n = n;
	/*-------- Allocate mem for COO */
	coo->ir = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->jc = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->val = (float*)malloc(n * n * sizeof(float));
	/*-------- Generate normal dist. & randomness */
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_real_distribution<> dist_float(-9.9999999999999, 9.9999999999999);
	std::uniform_int_distribution<> dist_exp(-10, 10);
	/*-------- Start/End Params */
	long start_row = (flip) ? -(n - 1) : 0;
	long start_row_flipped = (!flip) ? -(n - 1) : 0;
	long start_col = (flip) ? -(n - start - 1) : start;
	long start_col_flipped = (!flip) ? -(n - start - 1) : start;
	long end = (flip) ? 1 : n;
	long end_flipped = (!flip) ? 1 : n;
	/*-------- Create matrix */
	long coo_index = 0;
	for (long i = start_row; i < end; i++)
	{
		for (long j = ((zigzag && i % 2 != 0) ? start_col_flipped : start_col); j < ((zigzag && i % 2 != 0) ? end_flipped : end); j += skip)
		{
			coo->ir[coo_index] = abs(i) + 1;
			coo->jc[coo_index] = abs(j) + 1;
			coo->val[coo_index] = (float)dist_float(generator) * (float)pow(10.0, dist_exp(generator));
			coo_index++;
		}
	}
	coo->nnz = coo_index;
	coo->ir = (IndexType*)realloc(coo->ir, coo_index * sizeof(IndexType));
	coo->jc = (IndexType*)realloc(coo->jc, coo_index * sizeof(IndexType));
	coo->val = (float*)realloc(coo->val, coo_index * sizeof(float));
}
