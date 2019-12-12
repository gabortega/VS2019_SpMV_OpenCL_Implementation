#ifndef GEN_MAT_H
#define GEN_MAT_H

#include<compiler_config.h>

#include<IO/convert_input.h>

#include<chrono>
#include<random>
#include<math.h>

void generateMatrixGaussMethodRow(IndexType n, REAL row_mean, REAL row_stddev, struct coo_t* coo);
void generateMatrixGaussMethodCol(IndexType n, REAL col_mean, REAL col_stddev, struct coo_t* coo);
void generateMatrixGaussMethodFull(IndexType n, REAL row_mean, REAL row_stddev, REAL col_mean, REAL col_stddev, struct coo_t* coo);

void generateMatrixImbalancedRow(IndexType n, IndexType start, IndexType skip, struct coo_t* coo);
void generateMatrixImbalancedCol(IndexType n, IndexType start, IndexType skip, struct coo_t* coo);

#endif