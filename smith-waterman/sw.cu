#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>


#define S_LEN 512
#define N 1000

#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

double get_time() // function that returns the time of day in seconds
{   
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int max4(int n1, int n2, int n3, int n4)
{
	int tmp1, tmp2;
	tmp1 = n1 > n2 ? n1 : n2;
	tmp2 = n3 > n4 ? n3 : n4;
	tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
	return tmp1;
}

void backtrace(char *simple_rev_cigar, char **dir_mat, int i, int j, int max_cigar_len)
{
	int n;
	for (n = 0; n < max_cigar_len && dir_mat[i][j] != 0; n++)
	{
		int dir = dir_mat[i][j];
		if (dir == 1 || dir == 2)
		{
			i--;
			j--;
		}
		else if (dir == 3)
			i--;
		else if (dir == 4)
			j--;

		simple_rev_cigar[n] = dir;
	}
}

void cpuSW(char ** query, char ** reference, int ** sc_mat, char ** dir_mat, int * res, char ** simple_rev_cigar){
    double start_cpu = get_time();

    int ins = -2, del = -2, match = 1, mismatch = -1; // penalties
	
	for (int n = 0; n < N; n++)
	{
		int max = ins; // in sw all scores of the alignment are >= 0, so this will be for sure changed
		int maxi, maxj;
		// initialize the scoring matrix and direction matrix to 0
		for (int i = 0; i < S_LEN + 1; i++)
		{
			for (int j = 0; j < S_LEN + 1; j++)
			{
				sc_mat[i][j] = 0;
				dir_mat[i][j] = 0;
			}
		}
		// compute the alignment
		for (int i = 1; i < S_LEN; i++)
		{
			for (int j = 1; j < S_LEN; j++)
			{
				// compare the sequences characters
				int comparison = (query[n][i - 1] == reference[n][j - 1]) ? match : mismatch;
				// compute the cell knowing the comparison result
				int tmp = max4(sc_mat[i - 1][j - 1] + comparison, sc_mat[i - 1][j] + del, sc_mat[i][j - 1] + ins, 0);
				char dir;

				if (tmp == (sc_mat[i - 1][j - 1] + comparison))
					dir = comparison == match ? 1 : 2;
				else if (tmp == (sc_mat[i - 1][j] + del))
					dir = 3;
				else if (tmp == (sc_mat[i][j - 1] + ins))
					dir = 4;
				else
					dir = 0;

				dir_mat[i][j] = dir;
				sc_mat[i][j] = tmp;

				if (tmp > max)
				{
					max = tmp;
					maxi = i;
					maxj = j;
				}
			}
		}
		res[n] = sc_mat[maxi][maxj];
		backtrace(simple_rev_cigar[n], dir_mat, maxi, maxj, S_LEN * 2);
    }
	double end_cpu = get_time();
	printf("SW Time CPU: %.10lf\n", end_cpu - start_cpu);

}

__device__ int max4GPU(int n1, int n2, int n3, int n4)
{
	int tmp1, tmp2;
	tmp1 = n1 > n2 ? n1 : n2;
	tmp2 = n3 > n4 ? n3 : n4;
	tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
	return tmp1;
}

__device__ void gpuBacktrace(char *simple_rev_cigar, char **dir_mat, int i, int j, int max_cigar_len)
{
	int n;
	for (n = 0; n < max_cigar_len && dir_mat[i][j] != 0; n++)
	{
		int dir = dir_mat[i][j];
		if (dir == 1 || dir == 2)
		{
			i--;
			j--;
		}
		else if (dir == 3)
			i--;
		else if (dir == 4)
			j--;

		simple_rev_cigar[n] = dir;
	}
}

__global__ void gpuSW(char ** query, char ** reference, int *** sc_mat_list, char *** dir_mat_list, int * res, char ** simple_rev_cigar) {
	int ins = -2, del = -2, match = 1, mismatch = -1; // penalties

	int n = blockIdx.x;
	int tid = threadIdx.x;

	int ** sc_mat = sc_mat_list[n];
    char ** dir_mat = dir_mat_list[n];

    int i, j=1;
    
    int max, maxi, maxj;

	__shared__ int value_max[S_LEN];
	__shared__ int pos_max[S_LEN];

	// initialize the scoring matrix and direction matrix to 0
    for(i=0; i<S_LEN; i++) {
        sc_mat[i][tid] = 0;
        dir_mat[i][tid] = 0;
    }
    
	// compute the alignment
    for(i=1; i<S_LEN*2; i++) {
        if(tid < i && j <= S_LEN) {
			char *temp_query;
			char *temp_reference;
			temp_query = query[n];
			temp_reference = reference[n];
			int i = tid + 1;
			// compare the sequences characters
			int comparison = (temp_query[i - 1] == temp_reference[j - 1]) ? match : mismatch;
			// compute the cell knowing the comparison resultS
			int tmp = max4GPU(sc_mat[i - 1][j - 1] + comparison, sc_mat[i - 1][j] + del, sc_mat[i][j- 1] + ins, 0);
			char dir;
			if (tmp == (sc_mat[i - 1][j - 1] + comparison))
				dir = comparison == match ? 1 : 2;
			else if (tmp == (sc_mat[i - 1][j] + del))
				dir = 3;
			else if (tmp == (sc_mat[i][j - 1] + ins))
				dir = 4;
			else
				dir = 0;
			dir_mat[i][j] = dir;
			sc_mat[i][j] = tmp;
            j++;
        }
        __syncthreads();
    }

	value_max[tid] = sc_mat[tid][1];
	pos_max[tid] = -1; //this value MUST be changed in the for loop
	for(i=1; i<S_LEN; i++) {
        if(sc_mat[tid][i] > value_max[tid]) {
           value_max[tid] = sc_mat[tid][i];
        	pos_max[tid] = i;
        }
    }  
    __syncthreads();      
	  
    if(tid == 0) {
        max = ins;
        maxi = tid;
        maxj = pos_max[0];
        for(i=1; i<S_LEN; i++) {
            if(value_max[i] > max) {
                max = value_max[i];
                maxi = i;
                maxj = pos_max[i];
            }
        }    
        res[n] = sc_mat[maxi][maxj];
        gpuBacktrace(simple_rev_cigar[n], dir_mat, maxi, maxj, S_LEN*2);
    }
}

int main(int argc, char const *argv[])
{
    srand(time(NULL));

	char alphabet[5] = {'A', 'C', 'G', 'T', 'N'};

	char **query = (char **)malloc(N * sizeof(char *));
	for (int i = 0; i < N; i++)
		query[i] = (char *)malloc(S_LEN * sizeof(char));

	char **reference = (char **)malloc(N * sizeof(char *));
	for (int i = 0; i < N; i++)
		reference[i] = (char *)malloc(S_LEN * sizeof(char));

	int **sc_mat = (int **)malloc((S_LEN + 1) * sizeof(int *));
	for (int i = 0; i < (S_LEN + 1); i++)
		sc_mat[i] = (int *)malloc((S_LEN + 1) * sizeof(int));
	char **dir_mat = (char **)malloc((S_LEN + 1) * sizeof(char *));
	for (int i = 0; i < (S_LEN + 1); i++)
		dir_mat[i] = (char *)malloc((S_LEN + 1) * sizeof(char));

	int *res = (int *)malloc(N * sizeof(int));
	char **simple_rev_cigar = (char **)malloc(N * sizeof(char *));
	for (int i = 0; i < N; i++)
		simple_rev_cigar[i] = (char *)malloc(S_LEN * 2 * sizeof(char));

	// randomly generate sequences
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < S_LEN; j++)
		{
			query[i][j] = alphabet[rand() % 5];
			reference[i][j] = alphabet[rand() % 5];
	}
	}

    //START CPU
    cpuSW(query, reference, sc_mat, dir_mat, res, simple_rev_cigar);
    
	//START GPU
	//allocate memory on the device

    char **d_query;
    char **d_queryh = (char **)malloc(N * sizeof(char *));    
    CHECK(cudaMalloc((void***)&d_query,  N * sizeof(char *)));
    for(int i=0; i<N; i++) {
        CHECK(cudaMalloc((void**) &(d_queryh[i]), S_LEN*sizeof(char)));
        CHECK(cudaMemcpy (d_queryh[i], query[i], S_LEN*sizeof(char), cudaMemcpyHostToDevice));
    }
    CHECK(cudaMemcpy (d_query, d_queryh, N*sizeof(char *), cudaMemcpyHostToDevice));

    char **d_reference;
    char **d_referenceh = (char **)malloc(N * sizeof(char *));    
    CHECK(cudaMalloc((void***)&d_reference,  N * sizeof(char *)));
    for(int i=0; i<N; i++) {
        CHECK(cudaMalloc((void**) &(d_referenceh[i]), S_LEN*sizeof(char)));
        CHECK(cudaMemcpy (d_referenceh[i], reference[i], S_LEN*sizeof(char), cudaMemcpyHostToDevice));
    }
    CHECK(cudaMemcpy (d_reference, d_referenceh, N*sizeof(char *), cudaMemcpyHostToDevice));
	
	int **d_sc_mat;
    int **d_sc_math;
	int ***d_sc_mat_list;
	int ***d_sc_mat_listh = (int ***)malloc(N * sizeof(int **));

    CHECK(cudaMalloc((void***)&d_sc_mat_list, N * sizeof(int **)));
	for(int j=0; j<N; j++) {
		d_sc_math = (int **)malloc((S_LEN+1) * sizeof(int *));
		CHECK(cudaMalloc((void***)&d_sc_mat, (S_LEN+1) * sizeof(int *)));
		for(int i=0; i<(S_LEN+1); i++) {
			CHECK(cudaMalloc((void**) &(d_sc_math[i]), (S_LEN+1)*sizeof(int)));
		}
		CHECK(cudaMemcpy (d_sc_mat, d_sc_math, (S_LEN+1)*sizeof(int *), cudaMemcpyHostToDevice));
		d_sc_mat_listh[j] = d_sc_mat;
	}
	CHECK(cudaMemcpy (d_sc_mat_list, d_sc_mat_listh, N*sizeof(int **), cudaMemcpyHostToDevice));

	char **d_dir_mat;
    char **d_dir_math;
	char ***d_dir_mat_list;
	char ***d_dir_mat_listh = (char ***)malloc(N * sizeof(char **));
    CHECK(cudaMalloc((void***)&d_dir_mat_list, N * sizeof(char **)));
	for(int j=0; j<N; j++) {
		d_dir_math = (char **)malloc((S_LEN+1) * sizeof(char *));    
		CHECK(cudaMalloc((void***)&d_dir_mat, (S_LEN+1) * sizeof(char *)));
		for(int i=0; i<(S_LEN+1); i++) {
			CHECK(cudaMalloc((void**) &(d_dir_math[i]), (S_LEN+1)*sizeof(char)));
		}
		CHECK(cudaMemcpy (d_dir_mat, d_dir_math, (S_LEN+1)*sizeof(char *), cudaMemcpyHostToDevice));
		d_dir_mat_listh[j] = d_dir_mat;
	}
	CHECK(cudaMemcpy (d_dir_mat_list, d_dir_mat_listh, N*sizeof(char **), cudaMemcpyHostToDevice));

    int * d_res;
    CHECK(cudaMalloc((void**) &d_res, N*sizeof(int)));

    char **d_simple_rev_cigar;
    char **d_simple_rev_cigarh = (char **)malloc(N * sizeof(char *));    
    CHECK(cudaMalloc((void***)&d_simple_rev_cigar, N * sizeof(char *)));
    for(int i=0; i<N; i++) {
        CHECK(cudaMalloc((void**) &(d_simple_rev_cigarh[i]), S_LEN * 2 * sizeof(char)));
    }
    CHECK(cudaMemcpy (d_simple_rev_cigar, d_simple_rev_cigarh, N*sizeof(char *), cudaMemcpyHostToDevice));

    // Execution on GPU started
    double start_gpu = get_time();
    gpuSW<<<N, S_LEN>>>(d_query, d_reference, d_sc_mat_list, d_dir_mat_list, d_res, d_simple_rev_cigar);
	CHECK_KERNELCALL();
	double end_gpu = get_time();
    printf("SW Time GPU: %.10lf\n", end_gpu - start_gpu);

	// Transfer data to host
	int gpu_result[N];
	char gpu_simple_rev_cigar[N][S_LEN*2];
	char * tmp_pointer;
	CHECK(cudaMemcpy((void *) gpu_result, d_res, sizeof(int)*N, cudaMemcpyDeviceToHost));
	for(int i=0; i<N; i++) {
		tmp_pointer = d_simple_rev_cigarh[i];
		CHECK(cudaMemcpy((void *) gpu_simple_rev_cigar[i], tmp_pointer, sizeof(char)*S_LEN*2, cudaMemcpyDeviceToHost));
	}
	int counterr = 0;
	for(int i=0;i<N;i++){
		if (gpu_result[i]!=res[i])
		{
			counterr++;
			printf("Error in result in position %d: CPU_VALUE: %d \t GPU_VALUE: %d \n", i, res[i], gpu_result[i]);
		}
		
	}
	for(int i=0; (i<N ); i++) {
		for(int j=0; (j<S_LEN*2 ); j++) {
			if(gpu_simple_rev_cigar[i][j] != simple_rev_cigar[i][j]){
				counterr++;
				printf("Error in simple_rev_cigar");
			}
		}
	}
	if (counterr == 0)
	{
		printf("CPU and GPU returned the same result\n");
	}

	// Deallocation of memory
	for(int i=0; i<N; i++) {
		CHECK(cudaFree(d_simple_rev_cigarh[i]));
		CHECK(cudaFree(d_dir_mat_listh[i]));
		CHECK(cudaFree(d_sc_mat_listh[i]));
		CHECK(cudaFree(d_referenceh[i]));
		CHECK(cudaFree(d_queryh[i]));
    }
	CHECK(cudaFree(d_query));
    CHECK(cudaFree(d_reference));
    CHECK(cudaFree(d_sc_mat_list));
    CHECK(cudaFree(d_dir_mat_list));
    CHECK(cudaFree(d_res));
    CHECK(cudaFree(d_simple_rev_cigar));

    return 0;
}
