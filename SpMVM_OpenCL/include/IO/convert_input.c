#include "convert_input.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "mmio.h"

#define MAX_LINE 200

/*--------------------------------------------------*/
void MM_To_COO(const char* filename, struct coo_t* coo, int log)
{
	MM_typecode matcode;
	FILE* p;
	if ((fopen_s(&p, filename, "r")) != 0 || p == NULL) {
		fprintf(stdout, "Unable to open file %s\n", filename);
		exit(1);
	}
	/*----------- READ MM banner */
	if (mm_read_banner(p, &matcode) != 0) {
		fprintf(stdout, "Could not process Matrix Market banner.\n");
		exit(1);
	}
	if (!mm_is_valid(matcode)) {
		fprintf(stdout, "Invalid Matrix Market file.\n");
		exit(1);
	}
	if (!(mm_is_real(matcode) && mm_is_coordinate(matcode) && mm_is_sparse(matcode))) {
		fprintf(stdout, "Only sparse real-valued coordinate matrices are supported\n");
		exit(1);
	}
	IndexType nrow, ncol, nnz, nnz2, k, j;
	char line[MAX_LINE];
	/*------------- Read size */
	if (mm_read_mtx_crd_size(p, &nrow, &ncol, &nnz) != 0) {
		fprintf(stdout, "MM read size error !\n");
		exit(1);
	}
	if (nrow != ncol) {
		fprintf(stdout, "This is not a square matrix!\n");
		exit(1);
	}
	/*--------------------------------------
	 * symmetric case : only L part stored,
	 * so nnz2 := 2*nnz - nnz of diag,
	 * so nnz2 <= 2*nnz
	 *-------------------------------------*/
	if (mm_is_symmetric(matcode))
		nnz2 = 2 * nnz;
	else
		nnz2 = nnz;
	/*-------- Allocate mem for COO */
	coo->ir = (IndexType*) malloc(nnz2 * sizeof(IndexType));
	coo->jc = (IndexType*) malloc(nnz2 * sizeof(IndexType));
	coo->val = (REAL* )malloc(nnz2 * sizeof(REAL));
	/*-------- read line by line */
	char* p1, * p2;
	for (k = 0; k < nnz; k++) {
		fgets(line, MAX_LINE, p);
		for (p1 = line; ' ' == *p1; p1++);
		/*----------------- 1st entry - row index */
		for (p2 = p1; ' ' != *p2; p2++);
		*p2 = '\0';
		float tmp1 = atof(p1);
		//coo->ir[k] = atoi(p1);
		coo->ir[k] = (IndexType)tmp1;
		/*-------------- 2nd entry - column index */
		for (p1 = p2 + 1; ' ' == *p1; p1++);
		for (p2 = p1; ' ' != *p2; p2++);
		*p2 = '\0';
		float tmp2 = atof(p1);
		coo->jc[k] = (IndexType)tmp2;
		//coo->jc[k]  = atoi(p1);
	/*------------- 3rd entry - nonzero entry */
		p1 = p2 + 1;
		coo->val[k] = atof(p1);
	}
	/*------------------ Symmetric case */
	j = nnz;
	if (mm_is_symmetric(matcode)) {
		for (k = 0; k < nnz; k++)
			if (coo->ir[k] != coo->jc[k]) {
				/*------------------ off-diag entry */
				coo->ir[j] = coo->jc[k];
				coo->jc[j] = coo->ir[k];
				coo->val[j] = coo->val[k];
				j++;
			}
		if (j != nnz2) {
			coo->ir = (IndexType*) realloc(coo->ir, j * sizeof(IndexType));
			coo->jc = (IndexType*) realloc(coo->jc, j * sizeof(IndexType));
			coo->val = (REAL*) realloc(coo->val, j * sizeof(REAL));
		}
	}
	coo->n = nrow;
	coo->nnz = j;

	if (log)
	{
		fprintf(stdout, "COO: Matrix N = %d, NNZ = %d\n", nrow, j);
		for (IndexType i = 0; i < coo->nnz; i++)
			fprintf(stdout, "%d %d %20.19g\n", coo->ir[i], coo->jc[i], coo->val[i]);
		fprintf(stdout, "\n");
	}
	fclose(p);
}

void COO_To_MAT(struct coo_t* coo, struct mat_t* mat, int log)
{
	mat->n = coo->n;
	mat->nnz = coo->nnz;
	mat->val = (REAL*)calloc((unsigned long)mat->n * mat->n, sizeof(REAL));
	for (IndexType i = 0; i < mat->nnz; i++)
		*(mat->val + ((*(coo->jc + i) - 1) * mat->n) + (*(coo->ir + i) - 1)) = *(coo->val + i);

	if (log)
	{
		fprintf(stdout, "MAT: Matrix N = %d, NNZ = %d\n", mat->n, mat->nnz);
		for (IndexType i = 0; i < mat->n; i++)
		{
			for (IndexType j = 0; j < mat->n; j++)
				fprintf(stdout, "%20.19g ", mat->val[j + (i * mat->n)]);
			fprintf(stdout, "\n");
		}
		fprintf(stdout, "\n");
		
	}
}

/*--------------------------------------------------*/
void COO_To_CSR(struct coo_t *coo, struct csr_t *csr, int log) 
{
	//Allocate CSR
	csr->n = coo->n;
	csr->nnz = coo->nnz;
	csr->ia = (IndexType*) malloc((csr->n+1)*sizeof(IndexType));
	csr->ja = (IndexType*) malloc(csr->nnz*sizeof(IndexType));
	csr->a = (REAL *) malloc(csr->nnz*sizeof(REAL));
	//COO -> CSR (taken from SpMV Fortran code)
	IndexType k, k0, i, j, iad;
	REAL x;
	//set all values to 0
	for (k = 0; k < csr->n + 1; k++)
		*(csr->ia + k) = 0;
	//determine row-lengths
	for (k = 0; k < csr->nnz; k++)
		*(csr->ia + *(coo->ir + k)) = *(csr->ia + *(coo->ir + k))+1;
	//determine starting position of each row
	k = 0;
	for (j = 0; j < csr->n + 1; j++)
	{
		k0 = *(csr->ia + j);
		*(csr->ia + j) = k;
		k += k0;
	}
	//Fill out csr arrays
	for (k = 0; k < csr->nnz; k++)
	{
		i = *(coo->ir + k);
		j = *(coo->jc + k);
		x = *(coo->val + k);
		iad = *(csr->ia + i);
		*(csr->a + iad) = x;
		*(csr->ja + iad) = j;
		*(csr->ia + i) = iad + 1;
	}
	for (k = 0; k < csr->n + 1; k++)
	{
		*(csr->ia + k) = *(csr->ia + k) + 1;
	}

	if (log)
	{
		fprintf(stdout, "CSR: Matrix N = %d, NNZ = %d\n", csr->n, csr->nnz);
		fprintf(stdout, "csr->ja | csr->a\n");
		for (i = 0; i < csr->nnz; i++)
			fprintf(stdout, "%d %20.19g\n", csr->ja[i], csr->a[i]);
		fprintf(stdout, "\n");
		fprintf(stdout, "csr->ia: \n");
		for (i = 0; i < csr->n + 1; i++)
			fprintf(stdout, "%d ", csr->ia[i]);
		fprintf(stdout, "\n");
	}
}

/*-------------------------------------------------*/
void CSR_To_JAD(struct csr_t* csr, struct jad_t* jad, int log)
{
	// Allocate JAD
	IndexType n = jad->n = csr->n;
	jad->nnz = csr->nnz;
	jad->total = csr->nnz;
	jad->njad = (IndexType*)malloc((csr->n + 1) * sizeof(IndexType));
	jad->ia = (IndexType*)malloc((csr->n + 1) * sizeof(IndexType));
	jad->ja = (IndexType*)malloc(csr->nnz * sizeof(IndexType));
	jad->a = (REAL*)malloc(csr->nnz * sizeof(REAL));
	jad->perm = (IndexType*)malloc(csr->n * sizeof(IndexType));
	// CSR -> JAD (taken from SpMV Fortran code)
	IndexType ilo, j, len, k, i, k0, k1, jj;
	//
	for (i = 0; i < jad->n + 1; i++)
		*(jad->njad + i) = 0;
	//
	ilo = jad->n;
	//
	for (j = 0; j < jad->n; j++)
	{
		*(jad->perm + j) = j;
		len = *(csr->ia + j + 1) - *(csr->ia + j);
		ilo = min(ilo, len);
		*(jad->njad + n) = max(*(jad->njad + n), len);
		*(jad->njad + j) = len;
	}
	//
	long* work_array = (long*)malloc((csr->n + 1) * sizeof(long));
	dcsort(jad->njad, jad->n, work_array, jad->perm, ilo, *(jad->njad + n));
	free(work_array);
	//
	for (j = 0; j < jad->n; j++)
		*(jad->ia + j) = 0;
	//
	for (k = 0; k < jad->n; k++)
	{
		len = *(jad->njad + *(jad->perm + k));
		for (i = 0; i < len; i++)
		{
			*(jad->ia + i) = *(jad->ia + i) + 1;
		}
	}
	//
	k1 = 0;
	k0 = k1;
	for (jj = 0; jj < *(jad->njad + n); jj++)
	{
		len = *(jad->ia + jj);
		for (k = 0; k < len; k++)
		{
			i = *(csr->ia + *(jad->perm + k)) + jj - 1;
			
			*(jad->a + k1) = *(csr->a + i);
			*(jad->ja + k1) = *(csr->ja + i);
			k1 += 1;
		}
		*(jad->ia + jj) = k0 + 1;
		k0 = k1;
	}
	*(jad->ia + *(jad->njad + n)) = k1 + 1;

  /*------ pad jad each jad to have multiple of the WARP_SIZE */
	PadJADWARP(jad);

	if (log)
	{
		fprintf(stdout, "JAD: Matrix N = %d, NNZ = %d\n", jad->n, jad->nnz);
		fprintf(stdout, "jad->njad: ");
		for (IndexType i = 0; i < n + 1; i++)
			fprintf(stdout, "%d ", jad->njad[i]);
		fprintf(stdout, "\n");

		fprintf(stdout, "jad->ia: ");
		for (IndexType i = 0; i < *(jad->njad + n) + 1; i++)
			fprintf(stdout, "%d ", jad->ia[i]);
		fprintf(stdout, "\n");

		fprintf(stdout, "jad->ja: ");
		for (IndexType i = 0; i < jad->total; i++)
			fprintf(stdout, "%d ", jad->ja[i]);
		fprintf(stdout, "\n");

		fprintf(stdout, "jad->a: ");
		for (IndexType i = 0; i < jad->total; i++)
			fprintf(stdout, "%g ", jad->a[i]);
		fprintf(stdout, "\n");

		fprintf(stdout, "jad->perm: ");
		for (IndexType i = 0; i < n; i++)
			fprintf(stdout, "%d ", jad->perm[i]);
		fprintf(stdout, "\n");
	}
}

/*--------------------------------------------------*/
/* distribution count sort subroutine (taken from SpMV Fortran code) */
void dcsort(IndexType* ival, IndexType n, long* icnt, IndexType* index, IndexType ilo, IndexType ihi)
{
	long i, j, ivalj;
	//
	for (i = ilo - 1; i < ihi; i++)
		*(icnt + i) = 0;
	//
	for (i = 0; i < n; i++)
	{
		if (*(ival + i) - 1 > ilo - 2 && *(ival + i) - 1 < ihi)
			*(icnt + *(ival + i) - 1) = *(icnt + *(ival + i) - 1) + 1;
	}
	//
	for (i = ihi - 2; i > ilo - 2; i--)
	{
		if ((i > ilo - 2 && i < ihi) && (i + 1 > ilo - 2 && i + 1 < ihi))
			*(icnt + i) = *(icnt + i) + *(icnt + i + 1);
	}
	//
	for (j = n - 1; j > -1; j--)
	{
		ivalj = *(ival + j) - 1;
		if (ivalj > ilo - 2 && ivalj < ihi)
			*(index + *(icnt + ivalj) - 1) = j;
		if (ivalj > ilo - 2 && ivalj < ihi)
			*(icnt + ivalj) = *(icnt + ivalj) - 1;
	}
}

/*---------------------------------------------------------*/
void PadJADWARP(struct jad_t* jad)
{
	IndexType i;
	IndexType n = jad->n;
	IndexType nnz2 = 0;
	IndexType* oldia = jad->ia;
	jad->ia = (IndexType*)malloc((*(jad->njad + n) + 1) * sizeof(IndexType));
	jad->ia[0] = 1;

	for (i = 0; i < *(jad->njad + n); i++) {
		jad->ia[i + 1] = jad->ia[i] + (oldia[i + 1] - oldia[i] + WARP_SIZE-1) / WARP_SIZE * WARP_SIZE;
		nnz2 += (jad->ia[i + 1] - jad->ia[i]);
	}

	REAL* olda = jad->a;
	IndexType* oldja = jad->ja;

	//fprintf(stdout, "Pading with zeros %.2f\n",\
    (double)nnz2/(double)jad->nnz);

	jad->total = nnz2;
	jad->a = (REAL*)calloc(nnz2, sizeof(REAL));
	jad->ja = (IndexType*)calloc(nnz2, sizeof(IndexType));
	for (i = 0; i < nnz2; i++)
		jad->ja[i] = 1;

	for (i = 0; i < *(jad->njad + n); i++) {
		memcpy(&jad->a[jad->ia[i] - 1], &olda[oldia[i] - 1],
			(oldia[i + 1] - oldia[i]) * sizeof(REAL));
		memcpy(&jad->ja[jad->ia[i] - 1], &oldja[oldia[i] - 1],
			(oldia[i + 1] - oldia[i]) * sizeof(IndexType));
	}

	free(olda);
	free(oldia);
	free(oldja);
}

/*---------------------------------------------------------------*/
// Same as ELL but length of each row is saved during construction
int CSR_To_ELLG(struct csr_t* csr, struct ellg_t* ellg, int log)
{
	// Allocate ELLG
	IndexType n = ellg->n = csr->n, lenmax = 0, nnz = 0;
	/*------------ pad each diag to be multiple of the WARP_SIZE */
	ellg->stride = (n + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
	ellg->nell = (IndexType*)malloc((n + 1) * sizeof(IndexType));
	ellg->jcoeff = (IndexType*)malloc(MAX_ELLG * ellg->stride * sizeof(IndexType));
	ellg->a = (REAL*)malloc(MAX_ELLG * ellg->stride * sizeof(REAL));
	// CSR -> ELLG (based off ELL code)
	IndexType j, k, len, k1, i, jj;
	//
	for (j = 0; j < MAX_ELLG; j++)
	{
		for (i = 0; i < ellg->stride; i++)
		{
			*(ellg->a + (i * MAX_ELLG) + j) = 0.0f;
			*(ellg->jcoeff + (i * MAX_ELLG) + j) = 1;
		}
	}
	//
	for (i = 0; i < ellg->n + 1; i++)
		*(ellg->nell + i) = 0;
	//
	// Determine most non-zero elements in a single row
	for (j = 0; j < ellg->n; j++)
	{
		lenmax = max(*(csr->ia + j + 1) - *(csr->ia + j), lenmax);
		*(ellg->nell + j) = min(*(csr->ia + j + 1) - *(csr->ia + j), MAX_ELLG);
		*(ellg->nell + n) = max(*(ellg->nell + n), *(ellg->nell + j));
	}
	//
	for (jj = 0; jj < ellg->n; jj++)
	{
		k1 = jj;
		len = *(ellg->nell + jj);
		for (k = 0; k < len; k++)
		{
			i = *(csr->ia + jj) + k - 1;

			nnz++;
			*(ellg->a + k1) = *(csr->a + i);
			*(ellg->jcoeff + k1) = *(csr->ja + i);
			k1 += ellg->stride;
		}
	}

	ellg->a = (REAL*)realloc(ellg->a, *(ellg->nell + n) * ellg->stride * sizeof(REAL));
	ellg->jcoeff = (IndexType*)realloc(ellg->jcoeff, *(ellg->nell + n) * ellg->stride * sizeof(IndexType));
	ellg->nnz = nnz;

	if (log)
	{
		fprintf(stdout, "ELL-G: Matrix N = %d, NNZ = %d\n", ellg->n, ellg->nnz);

		fprintf(stdout, "ellg->nell: ");
		for (IndexType i = 0; i < ellg->n + 1; i++)
			fprintf(stdout, "%d ", ellg->nell[i]);
		fprintf(stdout, "\n");

		fprintf(stdout, "ellg->a: ");
		for (IndexType i = 0; i < ellg->stride * *(ellg->nell + n); i++)
			fprintf(stdout, "%g ", ellg->a[i]);
		fprintf(stdout, "\n");

		fprintf(stdout, "ellg->jcoeff: ");
		for (IndexType i = 0; i < ellg->stride * *(ellg->nell + n); i++)
			fprintf(stdout, "%d ", ellg->jcoeff[i]);
		fprintf(stdout, "\n");
	}

	/*-------------------------------*/
	if (lenmax <= MAX_ELLG)
	{
		return 1;
	}
	/*---- fail to convert */
	return 0;
}

/*-------------------------------------------------*/
int CSR_To_HLL(struct csr_t* csr, struct hll_t* hll, int log)
{
	// Allocate HLL
	IndexType n = hll->n = csr->n, lenmax = 0, nnz = 0;
	IndexType hoff_size = hll->nhoff = (((n + HLL_HACKSIZE - 1) / HLL_HACKSIZE) + 1);
	/*------------ pad each diag to be multiple of the WARP_SIZE */
	hll->nell = (IndexType*)malloc(hoff_size * sizeof(IndexType));
	hll->stride = (n + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
	hll->jcoeff = (IndexType*)malloc(MAX_HLL * hll->stride * sizeof(IndexType));
	hll->hoff = (IndexType*)malloc(hoff_size * sizeof(unsigned IndexType));
	hll->a = (REAL*)malloc(MAX_HLL * hll->stride * sizeof(REAL));
	// CSR -> HLL (based off ELL code)
	IndexType j, k, len, k1, i, jj, hack, lowerb, higherb;
	//
	for (j = 0; j < MAX_HLL; j++)
	{
		for (i = 0; i < hll->stride; i++)
		{
			*(hll->a + (i * MAX_HLL) + j) = 0.0f;
			*(hll->jcoeff + (i * MAX_HLL) + j) = 1;
		}
	}
	//
	for (i = 0; i < hoff_size; i++)
		*(hll->nell + i) = 0;
	//
	*(hll->hoff + 0) = 1;
	//
	for (hack = 0; hack < (hll->n + HLL_HACKSIZE - 1) / HLL_HACKSIZE; hack++)
	{
		lowerb = hack * HLL_HACKSIZE;
		higherb = min((hack + 1) * HLL_HACKSIZE, hll->n);
		//
		// Determine most non-zero elements in a single row inside the hack
		for (j = lowerb; j < higherb; j++)
		{
			lenmax = max(*(csr->ia + j + 1) - *(csr->ia + j), lenmax);
			*(hll->nell + hack) = max(min(*(csr->ia + j + 1) - *(csr->ia + j), MAX_HLL), *(hll->nell + hack));
			*(hll->nell + hoff_size - 1) = max(*(hll->nell + hoff_size - 1), *(hll->nell + hack));
		}
		//
		for (jj = lowerb; jj < higherb; jj++)
		{
			k1 = *(hll->hoff + hack) - 1 + jj - lowerb;
			len = min(*(csr->ia + jj + 1) - *(csr->ia + jj), MAX_HLL);
			for (k = 0; k < len; k++)
			{
				i = *(csr->ia + jj) + k - 1;

				nnz++;
				*(hll->a + k1) = *(csr->a + i);
				*(hll->jcoeff + k1) = *(csr->ja + i);
				k1 += HLL_HACKSIZE;
			}
		}
		*(hll->hoff + hack + 1) = *(hll->hoff + hack) + *(hll->nell + hack) * HLL_HACKSIZE;
	}

	IndexType total_mem = 0;
	for (i = 0; i < hoff_size - 1; i++)
		total_mem += *(hll->hoff + i + 1) - *(hll->hoff + i);
	hll->total_mem = total_mem;
	hll->a = (REAL*)realloc(hll->a, total_mem * sizeof(REAL));
	hll->jcoeff = (IndexType*)realloc(hll->jcoeff, total_mem * sizeof(IndexType));
	hll->nnz = nnz;

	if (log)
	{
		fprintf(stdout, "HLL: Matrix N = %d, NNZ = %d, Memory = %d Units\n", hll->n, hll->nnz, total_mem);
		fprintf(stdout, "hll->nell: ");
		for (IndexType i = 0; i < hoff_size; i++)
			fprintf(stdout, "%d ", hll->nell[i]);
		fprintf(stdout, "\n");

		fprintf(stdout, "hll->a: ");
		for (IndexType i = 0; i < total_mem; i++)
			fprintf(stdout, "%g ", hll->a[i]);
		fprintf(stdout, "\n");

		fprintf(stdout, "hll->jcoeff: ");
		for (IndexType i = 0; i < total_mem; i++)
			fprintf(stdout, "%d ", hll->jcoeff[i]);
		fprintf(stdout, "\n");

		fprintf(stdout, "hll->hoff: ");
		for (IndexType i = 0; i < hll->nhoff; i++)
			fprintf(stdout, "%d ", hll->hoff[i]);
		fprintf(stdout, "\n");
	}

	/*-------------------------------*/
	if (lenmax <= MAX_HLL)
	{
		return 1;
	}
	/*---- fail to convert? */
	return 0;
}

/*--------------------------------------------------*/
int CSR_To_DIA(struct csr_t *csr, struct dia_t *dia, int log)
{
	IndexType n = csr->n;
	dia->ndiags = MAX_DIAG;
	/*------------------ Allocate DIA */
	dia->n = n;
	/*------------ pad each diag to be multiple of the WARP_SIZE */
	dia->stride = (n+ WARP_SIZE-1)/ WARP_SIZE * WARP_SIZE;
	dia->diags = (REAL*) malloc(dia->stride * MAX_DIAG * sizeof(REAL));
	dia->ioff = (int*) malloc(MAX_DIAG * sizeof(int));
	/*--------------- work array */
	int* ind = (IndexType*) malloc((2*n-1)*sizeof(int));

	// (taken from SpMV Fortran code)
	long n2, ii, k, j, i, l;
	bool breakWhile = false;
	//
	n2 = 2 * n - 1;
	infdia(dia->n, csr->ja, csr->ia, ind, 0);
	i = 0;
	ii = -1;
	do {
		ii += 1;
		for (k = 0; k < n2; k++)
		{
			j = *(ind + k);
			if (j > 0)
			{
				i = k;
				k = n2;
			}
		}
		if (j <= 0)
		{
			breakWhile = true;
		}
		else
		{
			*(dia->ioff + ii) = i + 1 - dia->n;
			*(ind + i) = -j;
		}
	} while (ii < dia->ndiags && !breakWhile);
	dia->ndiags = ii;
	for (j = 0; j <  dia->ndiags; j++)
	{
		for (i = 0; i < dia->stride; i++)
			*(dia->diags + (i * dia->ndiags) + j) = 0.0f;
	}
	for (i = 0; i < n; i++)
	{
		for (k = *(csr->ia + i) - 1; k < *(csr->ia + i + 1) - 1; k++)
		{
			j = *(csr->ja + k);
			for (l = 0; l < dia->ndiags; l++)
			{
				if (j - i - 1 == *(dia->ioff + l))
				{
					*(dia->diags + (l * dia->stride) + i) = *(csr->a + k);
					l = dia->ndiags;
				}
			}
		}
	}

	free(ind);
	/*-------------------------------*/
	if (dia->ndiags <= MAX_DIAG)
	{
		dia->diags = (REAL*) realloc(dia->diags, dia->stride * dia->ndiags * sizeof(REAL));
		dia->ioff = (int*) realloc(dia->ioff, dia->ndiags * sizeof(int));
		dia->nnz = csr->nnz;
		
		if (log)
		{
			fprintf(stdout, "DIA: Matrix N = %d, NNZ = %d\n", dia->n, dia->nnz);
			fprintf(stdout, "dia->ndiags: %d\n", dia->ndiags);

			fprintf(stdout, "dia->diags: ");
			for (IndexType i = 0; i < dia->stride * dia->ndiags; i++)
				fprintf(stdout, "%g ", dia->diags[i]);
			fprintf(stdout, "\n");

			fprintf(stdout, "dia->ioff: ");
			for (IndexType i = 0; i < dia->ndiags; i++)
				fprintf(stdout, "%d ", dia->ioff[i]);
			fprintf(stdout, "\n");
		}
		return 1;
	}
	/*---- fail to convert */
	return 0;
}

/*----------------------------------------------------------*/
/* obtains information on the diagonals of the input matrix */
void infdia(IndexType n, IndexType* ja, IndexType* ia, IndexType* ind, IndexType idiag)
{
	IndexType n2, i, k, j;
	//
	n2 = 2 * n - 1;
	for (i = 0; i < n2; i++)
		*(ind + i) = 0;
	for (i = 0; i < n; i++)
	{
		for (k = *(ia + i) - 1; k < *(ia + i + 1) - 1; k++)
		{
			j = *(ja + k);
			*(ind + n + j - i - 2) = *(ind + n + j - i - 2) + 1;
		}
	}
	//count the non-zeros
	idiag = 0;
	for (k = 0; k < n2; k++)
	{
		if (*(ind + k) != 0)
			idiag += 1;
	}
} 

/*--------------------------------------------------*/
int CSR_To_HDIA(struct csr_t* csr, struct hdia_t* hdia, int log)
{
	IndexType n = hdia->n = csr->n;
	IndexType ndiags = MAX_HDIAG;
	IndexType hoff_size = hdia->nhoff = (((n + HDIA_HACKSIZE - 1) / HDIA_HACKSIZE) + 1);
	/*------------------ Allocate HDIA */
	/*------------ pad each diag to be multiple of the WARP_SIZE */
	hdia->memoff = (IndexType*)malloc(hoff_size * sizeof(IndexType));
	hdia->stride = (n + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
	hdia->ndiags = (IndexType*)malloc(hoff_size * sizeof(IndexType));
	hdia->diags = (REAL*)malloc(hdia->stride * ndiags * sizeof(REAL));
	hdia->ioff = (int*)malloc((hoff_size - 1) * ndiags * sizeof(int));
	hdia->hoff = (IndexType*)malloc(hoff_size * sizeof(IndexType));
	/*--------------- work array */
	int* ind = (IndexType*)malloc((2 * n - 1) * sizeof(int));
	// (based off DIA & HLL code)
	long hack, lowerb, higherb, n2, ii, k, j, i, l, memoff, k1;
	//
	*(hdia->hoff + 0) = 0;
	//
	for (i = 0; i < hoff_size - 1; i++)
		*(hdia->ndiags + i) = MAX_HDIAG;
	*(hdia->ndiags + hoff_size - 1) = 0;
	//
	n2 = 2 * n - 1;
	//
	*(hdia->memoff + 0) = memoff = 0; // memory offset for diags array
	//
	for (hack = 0; hack < (hdia->n + HDIA_HACKSIZE - 1) / HDIA_HACKSIZE; hack++)
	{
		lowerb = hack * HDIA_HACKSIZE;
		higherb = min((hack + 1) * HDIA_HACKSIZE, hdia->n);
		bool breakWhile = false;
		//
		hinfdia(lowerb, higherb, hdia->n, csr->ja, csr->ia, ind, 0);
		//
		i = 0;
		ii = -1;
		do {
			ii += 1;
			for (k = 0; k < n2; k++)
			{
				j = *(ind + k);
				if (j > 0)
				{
					i = k;
					k = n2;
				}
			}
			if (j <= 0)
			{
				breakWhile = true;
			}
			else
			{
				*(hdia->ioff + ii + *(hdia->hoff + hack)) = i + 1 - higherb;
				*(ind + i) = -j;
			}
		} while (ii < *(hdia->ndiags + hack) && !breakWhile);
		*(hdia->ndiags + hack) = ii;
		*(hdia->ndiags + hoff_size - 1) = max(*(hdia->ndiags + hoff_size - 1), ii);
		*(hdia->hoff + hack + 1) = ii + *(hdia->hoff + hack);
		for (j = 0; j < *(hdia->ndiags + hack); j++)
		{
			for (i = 0; i < HDIA_HACKSIZE; i++)
				*(hdia->diags + (i * *(hdia->ndiags + hack)) + j + memoff) = 0.0f;
		}
		for (i = lowerb; i < higherb; i++)
		{
			k1 = memoff + i - lowerb;
			for (k = *(csr->ia + i) - 1; k < *(csr->ia + i + 1) - 1; k++)
			{
				j = *(csr->ja + k);
				for (l = 0; l < *(hdia->ndiags + hack); l++)
				{
					if (j - i - 1 == *(hdia->ioff + l + *(hdia->hoff + hack)))
					{
						*(hdia->diags + (l * HDIA_HACKSIZE) + k1) = *(csr->a + k);
						l = *(hdia->ndiags + hack);
					}
				}
			}
		}
		memoff += *(hdia->ndiags + hack) * HDIA_HACKSIZE;
		*(hdia->memoff + hack + 1) = memoff - higherb;
	}

	free(ind);
	/*-------------------------------*/
	if (*(hdia->ndiags + hoff_size - 1) <= MAX_HDIAG)
	{
		ndiags = *(hdia->ndiags + hoff_size - 1);
		*(hdia->memoff + hdia->nhoff - 1) = memoff;
		hdia->diags = (REAL*)realloc(hdia->diags, memoff * sizeof(REAL));
		hdia->ioff = (int*)realloc(hdia->ioff, *(hdia->hoff + hdia->nhoff - 1) * sizeof(int));
		hdia->nnz = csr->nnz;

		if (log)
		{
			fprintf(stdout, "HDIA: Matrix N = %d, NNZ = %d\n", hdia->n, hdia->nnz);
			fprintf(stdout, "hdia->ndiags: ");
			for (IndexType i = 0; i < hoff_size; i++)
				fprintf(stdout, "%d ", hdia->ndiags[i]);
			fprintf(stdout, "\n");

			fprintf(stdout, "hdia->diags: ");
			for (IndexType i = 0; i < memoff; i++)
				fprintf(stdout, "%g ", hdia->diags[i]);
			fprintf(stdout, "\n");

			fprintf(stdout, "hdia->ioff: ");
			for (IndexType i = 0; i < *(hdia->hoff + hdia->nhoff - 1); i++)
				fprintf(stdout, "%d ", hdia->ioff[i]);
			fprintf(stdout, "\n");

			fprintf(stdout, "hdia->hoff: ");
			for (IndexType i = 0; i < hdia->nhoff; i++)
				fprintf(stdout, "%d ", hdia->hoff[i]);
			fprintf(stdout, "\n");

			fprintf(stdout, "hdia->memoff: ");
			for (IndexType i = 0; i < hdia->nhoff; i++)
				fprintf(stdout, "%d ", hdia->memoff[i]);
			fprintf(stdout, "\n");
		}
		return 1;
	}
	/*---- fail to convert */
	return 0;
}

/*----------------------------------------------------------*/
/* obtains information on the diagonals of the input matrix */
void hinfdia(IndexType lowerb, IndexType higherb, IndexType n, IndexType* ja, IndexType* ia, IndexType* ind, IndexType idiag)
{
	IndexType n2, i, k, j;
	//
	n2 = 2 * n - 1;
	for (i = 0; i < n2; i++)
		*(ind + i) = 0;
	for (i = lowerb; i < higherb; i++)
	{
		for (k = *(ia + i) - 1; k < *(ia + i + 1) - 1; k++)
		{
			j = *(ja + k);
			*(ind + higherb + j - i - 2) = *(ind + higherb + j - i - 2) + 1;
		}
	}
	//count the non-zeros
	idiag = 0;
	for (k = 0; k < n2; k++)
	{
		if (*(ind + k) != 0)
			idiag += 1;
	}
}

/*--------------------------------------------------*/
void CSR_To_HYBELLG(struct csr_t* csr, struct hybellg_t* hyb, int log)
{
	struct csr_t temp_csr; // used to generate ELL-G
	IndexType n = hyb->n = csr->n;
	IndexType nnz = hyb->nnz = csr->nnz, temp_nnz = 0, k, csr_nnz = 0, offset = 0;
	IndexType i, j, index, avg_nnz_len_per_row = 0;
	IndexType* nnz_len_per_row = (IndexType*)malloc(n * sizeof(IndexType));
	for (i = 0; i < n; i++)
	{
		*(nnz_len_per_row + i) = *(csr->ia + i + 1) - *(csr->ia + i);
		avg_nnz_len_per_row += *(csr->ia + i + 1) - *(csr->ia + i);
	}
	avg_nnz_len_per_row /= n;
	for (k = 1; avg_nnz_len_per_row >= k && k <= ELL_ROW_MAX; k <<= 1);

	/*-------- Allocate mem for temp CSR */
	temp_csr.n = n;
	temp_csr.ia = (IndexType*)malloc((n + 1) * sizeof(IndexType));
	temp_csr.ja = (IndexType*)malloc(nnz * sizeof(IndexType));
	temp_csr.a = (REAL*)malloc(nnz * sizeof(REAL));
	/*-------- Allocate mem for new CSR */
	hyb->csr.n = n;
	hyb->csr.ia = (IndexType*)malloc((n + 1) * sizeof(IndexType));
	hyb->csr.ja = (IndexType*)malloc(nnz * sizeof(IndexType));
	hyb->csr.a = (REAL*)malloc(nnz * sizeof(REAL));
	/*-------- Find and assign corresponding elements with the new threshold */
	*(temp_csr.ia + 0) = 1;
	*(hyb->csr.ia + 0) = 1;
	for (i = 0; i < n; i++)
	{
		*(temp_csr.ia + i + 1) = *(temp_csr.ia + i);
		*(hyb->csr.ia + i + 1) = *(hyb->csr.ia + i);
		if (*(nnz_len_per_row + i) <= k)
		{
			for (j = 0; j < *(nnz_len_per_row + i); j++)
			{
				*(temp_csr.ja + temp_nnz) = *(csr->ja + offset);
				*(temp_csr.a + temp_nnz) = *(csr->a + offset);
				*(temp_csr.ia + i + 1) += 1;
				temp_nnz++;
				offset += 1;
			}
		}
		else
		{
			for (j = 0; j < *(nnz_len_per_row + i); j++)
			{
				*(hyb->csr.ja + csr_nnz) = *(csr->ja + offset);
				*(hyb->csr.a + csr_nnz) = *(csr->a + offset);
				*(hyb->csr.ia + i + 1) += 1;
				csr_nnz++;
				offset += 1;
			}
		}
	}
	/*-------- Complete temp CSR */
	temp_csr.nnz = temp_nnz;
	temp_csr.ja = (IndexType*)realloc(temp_csr.ja, temp_nnz * sizeof(IndexType));
	temp_csr.a = (REAL*)realloc(temp_csr.a, temp_nnz * sizeof(REAL));
	/*-------- Complete new CSR */
	hyb->csr.nnz = csr_nnz;
	hyb->csr.ja = (IndexType*)realloc(hyb->csr.ja, csr_nnz * sizeof(IndexType));
	hyb->csr.a = (REAL*)realloc(hyb->csr.a, csr_nnz * sizeof(REAL));

	fprintf(stdout, "HYB(ELL-G): Original Matrix N = %d, NNZ = %d, AVG NNZ/ROW = %d\n", n, nnz, avg_nnz_len_per_row);
	fprintf(stdout, "            ELL-G part: NNZ = %d            CSR part: NNZ = %d\n", temp_nnz, csr_nnz);
	CSR_To_ELLG(&temp_csr, &hyb->ellg, log);

	FreeCSR(&temp_csr);

	if (log)
	{
		fprintf(stdout, "CSR: Matrix N = %d, NNZ = %d\n", hyb->csr.n, hyb->csr.nnz);
		fprintf(stdout, "hyb->csr.ja | hyb->csr.a\n");
		for (i = 0; i < hyb->csr.nnz; i++)
			fprintf(stdout, "%d %20.19g\n", hyb->csr.ja[i], hyb->csr.a[i]);
		fprintf(stdout, "\n");
		fprintf(stdout, "hyb->csr.ia: \n");
		for (i = 0; i < hyb->csr.n + 1; i++)
			fprintf(stdout, "%d ", hyb->csr.ia[i]);
		fprintf(stdout, "\n");
	}
}

/*--------------------------------------------------*/
void CSR_To_HYBHLL(struct csr_t* csr, struct hybhll_t* hyb, int log)
{
	struct csr_t temp_csr; // used to generate ELL-G
	IndexType n = hyb->n = csr->n;
	IndexType nnz = hyb->nnz = csr->nnz, temp_nnz = 0, k, csr_nnz = 0, offset = 0;
	IndexType i, j, index, avg_nnz_len_per_row = 0;
	IndexType* nnz_len_per_row = (IndexType*)malloc(n * sizeof(IndexType));
	for (i = 0; i < n; i++)
	{
		*(nnz_len_per_row + i) = *(csr->ia + i + 1) - *(csr->ia + i);
		avg_nnz_len_per_row += *(csr->ia + i + 1) - *(csr->ia + i);
	}
	avg_nnz_len_per_row /= n;
	for (k = 1; avg_nnz_len_per_row >= k && k <= ELL_ROW_MAX; k <<= 1);

	/*-------- Allocate mem for temp CSR */
	temp_csr.n = n;
	temp_csr.ia = (IndexType*)malloc((n + 1) * sizeof(IndexType));
	temp_csr.ja = (IndexType*)malloc(nnz * sizeof(IndexType));
	temp_csr.a = (REAL*)malloc(nnz * sizeof(REAL));
	/*-------- Allocate mem for new CSR */
	hyb->csr.n = n;
	hyb->csr.ia = (IndexType*)malloc((n + 1) * sizeof(IndexType));
	hyb->csr.ja = (IndexType*)malloc(nnz * sizeof(IndexType));
	hyb->csr.a = (REAL*)malloc(nnz * sizeof(REAL));
	/*-------- Find and assign corresponding elements with the new threshold */
	*(temp_csr.ia + 0) = 1;
	*(hyb->csr.ia + 0) = 1;
	for (i = 0; i < n; i++)
	{
		*(temp_csr.ia + i + 1) = *(temp_csr.ia + i);
		*(hyb->csr.ia + i + 1) = *(hyb->csr.ia + i);
		if (*(nnz_len_per_row + i) <= k)
		{
			for (j = 0; j < *(nnz_len_per_row + i); j++)
			{
				*(temp_csr.ja + temp_nnz) = *(csr->ja + offset);
				*(temp_csr.a + temp_nnz) = *(csr->a + offset);
				*(temp_csr.ia + i + 1) += 1;
				temp_nnz++;
				offset += 1;
			}
		}
		else
		{
			for (j = 0; j < *(nnz_len_per_row + i); j++)
			{
				*(hyb->csr.ja + csr_nnz) = *(csr->ja + offset);
				*(hyb->csr.a + csr_nnz) = *(csr->a + offset);
				*(hyb->csr.ia + i + 1) += 1;
				csr_nnz++;
				offset += 1;
			}
		}
	}
	/*-------- Complete temp CSR */
	temp_csr.nnz = temp_nnz;
	temp_csr.ja = (IndexType*)realloc(temp_csr.ja, temp_nnz * sizeof(IndexType));
	temp_csr.a = (REAL*)realloc(temp_csr.a, temp_nnz * sizeof(REAL));
	/*-------- Complete new CSR */
	hyb->csr.nnz = csr_nnz;
	hyb->csr.ja = (IndexType*)realloc(hyb->csr.ja, csr_nnz * sizeof(IndexType));
	hyb->csr.a = (REAL*)realloc(hyb->csr.a, csr_nnz * sizeof(REAL));

	fprintf(stdout, "HYB(HLL): Original Matrix N = %d, NNZ = %d, AVG NNZ/ROW = %d\n", n, nnz, avg_nnz_len_per_row);
	fprintf(stdout, "          HLL part: NNZ = %d              CSR part: NNZ = %d\n", temp_nnz, csr_nnz);
	CSR_To_HLL(&temp_csr, &hyb->hll, log);

	FreeCSR(&temp_csr);

	if (log)
	{
		fprintf(stdout, "CSR: Matrix N = %d, NNZ = %d\n", hyb->csr.n, hyb->csr.nnz);
		fprintf(stdout, "hyb->csr.ja | hyb->csr.a\n");
		for (i = 0; i < hyb->csr.nnz; i++)
			fprintf(stdout, "%d %20.19g\n", hyb->csr.ja[i], hyb->csr.a[i]);
		fprintf(stdout, "\n");
		fprintf(stdout, "hyb->csr.ia: \n");
		for (i = 0; i < hyb->csr.n + 1; i++)
			fprintf(stdout, "%d ", hyb->csr.ia[i]);
		fprintf(stdout, "\n");
	}
}

/*---------------------------*/
void FreeCOO(struct coo_t* coo)
{
	free(coo->ir);
	free(coo->jc);
	free(coo->val);
}

/*---------------------------*/
void FreeMAT(struct mat_t* mat)
{
	free(mat->val);
}

/*---------------------------*/
void FreeCSR(struct csr_t* csr)
{
	free(csr->a);
	free(csr->ia);
	free(csr->ja);
}

/*---------------------------*/
void FreeJAD(struct jad_t* jad)
{
	free(jad->njad);
	free(jad->ia);
	free(jad->ja);
	free(jad->a);
	free(jad->perm);
}

/*---------------------------*/
void FreeELLG(struct ellg_t* ellg)
{
	free(ellg->nell);
	free(ellg->a);
	free(ellg->jcoeff);
}

/*---------------------------*/
void FreeHLL(struct hll_t* hll)
{
	free(hll->nell);
	free(hll->jcoeff);
	free(hll->a);
	free(hll->hoff);
}

/*---------------------------*/
void FreeDIA(struct dia_t* dia)
{
	free(dia->diags);
	free(dia->ioff);
}

/*---------------------------*/
void FreeHDIA(struct hdia_t* hdia)
{
	free(hdia->ndiags);
	free(hdia->diags);
	free(hdia->ioff);
	free(hdia->hoff);
}

/*---------------------------*/
void FreeHYBELLG(struct hybellg_t* hyb)
{
	FreeELLG(&hyb->ellg);
	FreeCSR(&hyb->csr);
}

/*---------------------------*/
void FreeHYBHLL(struct hybhll_t* hyb)
{
	FreeHLL(&hyb->hll);
	FreeCSR(&hyb->csr);
}
