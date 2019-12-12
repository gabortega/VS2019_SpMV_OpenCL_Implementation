#include "generateMatrix.hpp"

void generateMatrixGaussMethodRow(IndexType n, REAL row_mean, REAL row_stddev, struct coo_t* coo)
{
	coo->n = n;
	/*-------- Allocate mem for COO */
	coo->ir = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->jc = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->val = (REAL*)malloc(n * n * sizeof(REAL));
	/*-------- Generate normal dist. & randomness */
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_real_distribution<> dist_real(-9.9999999999999, 9.9999999999999);
	std::uniform_int_distribution<> dist_exp(-10, 10);
	std::uniform_int_distribution<> dist_row_chance(0, n);
	std::normal_distribution<REAL> distribution(row_mean, row_stddev);
	/*-------- Create matrix */
	IndexType coo_index = 0;
	for (IndexType i = 0; i < n; i++)
	{
		REAL dist_row = distribution(generator);
		if (dist_row > 0)
		{
			for (IndexType j = 0; j < n; j++)
			{
				int roll_chance = (n + dist_row - 1) / dist_row;
				if (dist_row_chance(generator) / roll_chance >= 1)
				{
					coo->ir[coo_index] = i + 1;
					coo->jc[coo_index] = j + 1;
					coo->val[coo_index] = (REAL)dist_real(generator) * (REAL)pow(10.0, dist_exp(generator));
					coo_index++;
					dist_row--;
				}
			}
		}
	}

	coo->nnz = coo_index;
	coo->ir = (IndexType*)realloc(coo->ir, coo_index * sizeof(IndexType));
	coo->jc = (IndexType*)realloc(coo->jc, coo_index * sizeof(IndexType));
	coo->val = (REAL*)realloc(coo->val, coo_index * sizeof(REAL));
}

void generateMatrixGaussMethodCol(IndexType n, REAL col_mean, REAL col_stddev, struct coo_t* coo)
{
	coo->n = n;
	/*-------- Allocate mem for COO */
	coo->ir = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->jc = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->val = (REAL*)malloc(n * n * sizeof(REAL));
	/*-------- Generate normal dist. & randomness */
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_real_distribution<> dist_real(-9.9999999999999, 9.9999999999999);
	std::uniform_int_distribution<> dist_exp(-10, 10);
	std::uniform_int_distribution<> dist_col_chance(0, n);
	std::normal_distribution<REAL> distribution(col_mean, col_stddev);
	/*-------- Create matrix */
	IndexType coo_index = 0;
	for (IndexType j = 0; j < n; j++)
	{
		REAL dist_col = distribution(generator);
		if (dist_col > 0)
		{
			for (IndexType i = 0; i < n; i++)
			{
				int roll_chance = (n + dist_col - 1) / dist_col;
				if (dist_col_chance(generator) / roll_chance >= 1)
				{
					coo->ir[coo_index] = i + 1;
					coo->jc[coo_index] = j + 1;
					coo->val[coo_index] = (REAL)dist_real(generator) * (REAL)pow(10.0, dist_exp(generator));
					coo_index++;
					dist_col--;
				}
			}
		}
	}
	coo->nnz = coo_index;
	coo->ir = (IndexType*)realloc(coo->ir, coo_index * sizeof(IndexType));
	coo->jc = (IndexType*)realloc(coo->jc, coo_index * sizeof(IndexType));
	coo->val = (REAL*)realloc(coo->val, coo_index * sizeof(REAL));
}

void generateMatrixGaussMethodFull(IndexType n, REAL row_mean, REAL row_stddev, REAL col_mean, REAL col_stddev, struct coo_t* coo)
{
	coo->n = n;
	/*-------- Allocate mem for COO */
	coo->ir = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->jc = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->val = (REAL*)malloc(n * n * sizeof(REAL));
	/*-------- Generate normal dist. & randomness */
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_real_distribution<> dist_real(-9.9999999999999, 9.9999999999999);
	std::uniform_int_distribution<> dist_exp(-10, 10);
	std::uniform_int_distribution<> dist_slot_chance(0, n);
	std::normal_distribution<REAL> distribution_row(row_mean, row_stddev);
	std::normal_distribution<REAL> distribution_col(col_mean, col_stddev);
	/*-------- for each row & column*/
	REAL* row_mod_array = (REAL*)malloc(n * sizeof(REAL));
	REAL* col_mod_array = (REAL*)malloc(n * sizeof(REAL));
	for (IndexType k = 0; k < n; k++)
	{
		row_mod_array[k] = 0;
		col_mod_array[k] = 0;
	}
	/*-------- Create matrix */
	IndexType coo_index = 0;
	for (IndexType j = 0; j < n; j++)
	{
		for (IndexType i = 0; i < n; i++)
		{
			REAL dist_row = distribution_row(generator) - row_mod_array[i];
			REAL dist_col = distribution_col(generator) - col_mod_array[j];
			if (dist_row > 0 && dist_col > 0)
			{
				int row_roll_chance = (n + dist_row - 1) / dist_row;
				int col_roll_chance = (n + dist_col - 1) / dist_col;
				int roll = dist_slot_chance(generator);
				if (roll / (row_roll_chance * col_roll_chance) >= 1)
				{
					coo->ir[coo_index] = i + 1;
					coo->jc[coo_index] = j + 1;
					coo->val[coo_index] = (REAL)dist_real(generator) * (REAL)pow(10.0, dist_exp(generator));
					coo_index++;
					row_mod_array[i]++;
					col_mod_array[j]++;
				}
			}
		}
	}
	coo->nnz = coo_index;
	coo->ir = (IndexType*)realloc(coo->ir, coo_index * sizeof(IndexType));
	coo->jc = (IndexType*)realloc(coo->jc, coo_index * sizeof(IndexType));
	coo->val = (REAL*)realloc(coo->val, coo_index * sizeof(REAL));
}

void generateMatrixImbalancedRow(IndexType n, IndexType start, IndexType skip, struct coo_t* coo)
{
	coo->n = n;
	/*-------- Allocate mem for COO */
	coo->ir = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->jc = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->val = (REAL*)malloc(n * n * sizeof(REAL));
	/*-------- Generate normal dist. & randomness */
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_real_distribution<> dist_real(-9.9999999999999, 9.9999999999999);
	std::uniform_int_distribution<> dist_exp(-10, 10);
	/*-------- Create matrix */
	IndexType coo_index = 0;
	for (IndexType j = 0; j < n; j++)
	{
		for (IndexType i = start; i < n; i+=skip)
		{
			coo->ir[coo_index] = i + 1;
			coo->jc[coo_index] = j + 1;
			coo->val[coo_index] = (REAL)dist_real(generator) * (REAL)pow(10.0, dist_exp(generator));
			coo_index++;
		}
	}
	coo->nnz = coo_index;
	coo->ir = (IndexType*)realloc(coo->ir, coo_index * sizeof(IndexType));
	coo->jc = (IndexType*)realloc(coo->jc, coo_index * sizeof(IndexType));
	coo->val = (REAL*)realloc(coo->val, coo_index * sizeof(REAL));
}

void generateMatrixImbalancedCol(IndexType n, IndexType start, IndexType skip, struct coo_t* coo)
{
	coo->n = n;
	/*-------- Allocate mem for COO */
	coo->ir = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->jc = (IndexType*)malloc(n * n * sizeof(IndexType));
	coo->val = (REAL*)malloc(n * n * sizeof(REAL));
	/*-------- Generate normal dist. & randomness */
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_real_distribution<> dist_real(-9.9999999999999, 9.9999999999999);
	std::uniform_int_distribution<> dist_exp(-10, 10);
	/*-------- Create matrix */
	IndexType coo_index = 0;
	for (IndexType j = start; j < n; j+=skip)
	{
		for (IndexType i = 0; i < n; i++)
		{
			coo->ir[coo_index] = i + 1;
			coo->jc[coo_index] = j + 1;
			coo->val[coo_index] = (REAL)dist_real(generator) * (REAL)pow(10.0, dist_exp(generator));
			coo_index++;
		}
	}
	coo->nnz = coo_index;
	coo->ir = (IndexType*)realloc(coo->ir, coo_index * sizeof(IndexType));
	coo->jc = (IndexType*)realloc(coo->jc, coo_index * sizeof(IndexType));
	coo->val = (REAL*)realloc(coo->val, coo_index * sizeof(REAL));
}
