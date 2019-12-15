#ifndef GEN_MAT_H
#define GEN_MAT_H

#include<compiler_config.h>

#include<IO/convert_input.h>

#include<chrono>
#include<random>
#include<math.h>

void generateMatrixGaussMethodRow(long n, float row_mean, float row_stddev, struct coo_rand_t* coo, bool flip);
void generateMatrixGaussMethodCol(long n, float col_mean, float col_stddev, struct coo_rand_t* coo, bool flip);
void generateMatrixGaussMethodFull(long n, float row_mean, float row_stddev, float col_mean, float col_stddev, struct coo_rand_t* coo, bool flip);

void generateMatrixImbalancedRow(long n, long start_row, long skip, struct coo_rand_t* coo, bool flip);
void generateMatrixImbalancedCol(long n, long start_col, long skip, struct coo_rand_t* coo, bool flip);

#endif