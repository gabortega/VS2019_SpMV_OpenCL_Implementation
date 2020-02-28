// Implementation altered from CUDA_ITSOL SpMV by: Ruipeng Li, Yousef Saad
// URL: https://www-users.cs.umn.edu/~saad/software/CUDA_ITSOL/CUDA_ITSOL.tar.gz

#include<compiler_config.h>

#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<IO/mmio.h>

#ifndef CONVERT_IO_H
#define CONVERT_IO_H

#define IndexType unsigned int

#ifdef __cplusplus
extern "C"
{
#endif

#if PRECISION == 2
#define REAL double
#else
#define REAL float
#endif

/*---- sparse matrix data structure */
/* COO format type for randomly generated matrices */
struct coo_rand_t {
	IndexType n;
	IndexType nnz;
	IndexType* ir;
	IndexType* jc;
	float* val;
};

/* COO format type */
struct coo_t {
	IndexType n;
	IndexType nnz;
	IndexType* ir;
	IndexType* jc;
	REAL* val;
};

/* General Matrix format */
struct mat_t {
	IndexType n;
	IndexType nnz;
	REAL* val;
};

/* CSR format type */
struct csr_t {
	IndexType n;
	IndexType nnz;
	IndexType* ia;
	IndexType* ja;
	REAL* a;
};

/* JAD format type */
struct jad_t {
	IndexType n;
	IndexType nnz;
	IndexType total;
	IndexType* ia;
	IndexType* ja;
	REAL* a;
	IndexType* njad;
	IndexType* perm;
};

/* ELL-G format type */
struct ellg_t {
	IndexType n;
	IndexType nnz;
	IndexType* nell;
	IndexType stride;
	REAL* a;
	IndexType* jcoeff;
};

/* HLL format type */
struct hll_t {
	IndexType n;
	IndexType nnz;
	IndexType total_mem;
	IndexType* nell;
	IndexType nhoff;
	IndexType stride;
	REAL* a;
	IndexType* jcoeff;
	IndexType* hoff;
};

/* DIA format type */
struct dia_t {
	IndexType n;
	IndexType nnz;
	IndexType ndiags;
	IndexType stride;
	REAL* diags;
	int* ioff;
};

/* HDIA format type */
struct hdia_t {
	IndexType n;
	IndexType nnz;
	IndexType* memoff;
	IndexType* ndiags;
	IndexType nhoff;
	IndexType stride;
	REAL* diags;
	int* ioff;
	IndexType* hoff;
};

/* HYB format type (ELLG) */
struct hybellg_t {
	IndexType n;
	IndexType nnz;
	struct csr_t csr;
	struct ellg_t ellg;
};

/* HYB format type (HLL) */
struct hybhll_t {
	IndexType n;
	IndexType nnz;
	struct csr_t csr;
	struct hll_t hll;
};

void MM_To_COO(const char* filename, struct coo_t* coo, int log);
void COO_To_MM(struct coo_rand_t* coo, const char* filename);
void COO_To_CSR(struct coo_t* coo, struct csr_t* csr, int log);
void COO_To_MAT(struct coo_t* coo, struct mat_t* mat, int log);
void CSR_To_JAD(struct csr_t* csr, struct jad_t* jad, int log);
int CSR_To_ELLG(struct csr_t* csr, struct ellg_t* ellg, int log);
int CSR_To_HLL(struct csr_t* csr, struct hll_t* hll, int log);
int CSR_To_DIA(struct csr_t* csr, struct dia_t* dia, int log);
int CSR_To_HDIA(struct csr_t* csr, struct hdia_t* hdia, int log);
void CSR_To_HYBELLG(struct csr_t* coo, struct hybellg_t* hyb, int log);
void CSR_To_HYBHLL(struct csr_t* coo, struct hybhll_t* hyb, int log);

void transpose_ELLG(struct ellg_t* ellg, int log);
void transpose_HLL(struct hll_t* hll, int log);
void transpose_DIA(struct dia_t* dia, int log);
void transpose_HDIA(struct hdia_t* hdia, int log);

void dcsort(IndexType* ival, IndexType n, IndexType* icnt, IndexType* index, long ilo, long ihi);
void PadJADWARP(struct jad_t* jadg);
void infdia(IndexType n, IndexType* ja, IndexType* ia, IndexType* ind, IndexType idiag);
void hinfdia(IndexType lowerb, IndexType higherb, IndexType n, IndexType* ja, IndexType* ia, IndexType* ind, IndexType idiag);

void FreeCOORAND(struct coo_rand_t* coo);
void FreeCOO(struct coo_t* coo);
void FreeMAT(struct mat_t* mat);
void FreeCSR(struct csr_t* csr);
void FreeJAD(struct jad_t* jad);
void FreeELLG(struct ellg_t* ellg);
void FreeHLL(struct hll_t* hll);
void FreeDIA(struct dia_t* dia);
void FreeHDIA(struct hdia_t* hdia);
void FreeHYBELLG(struct hybellg_t* hyb);
void FreeHYBHLL(struct hybhll_t* hyb);

#ifdef __cplusplus
}
#endif

#endif