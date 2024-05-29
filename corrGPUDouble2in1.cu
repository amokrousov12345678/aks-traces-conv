//nvcc corrGPUDouble2in1.cu -lcufft --compiler-options -fPIC -std=c++11
#pragma GCC optimize("O3")

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cufft.h"
#include "cuComplex.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fcntl.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <string.h>
#include <assert.h>
#include <algorithm>
#include <unistd.h>

#define checkCudaErrors(val) checkCuda( (val), #val, __FILE__, __LINE__)

template<typename T>
void checkCuda(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

const char *cufftGetErrorString( cufftResult cufft_error_type ) {

    switch( cufft_error_type ) {

        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS: The CUFFT operation was performed";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN: The CUFFT plan to execute is invalid";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED: The allocation of data for CUFFT in memory failed";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE: The data type used by CUFFT is invalid";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE: The data value used by CUFFT is invalid";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR: An internal error occurred in CUFFT";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED: The execution of a plan by CUFFT failed";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED: The setup of CUFFT failed";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE: The size of the data to be used by CUFFT is invalid";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA: The data to be used by CUFFT is unaligned in memory";

    }

    return "Unknown CUFFT Error";
}

#define checkCufftErrors(val) checkCufft( (val), #val, __FILE__, __LINE__)

template<typename T>
void checkCufft(T err, const char* const func, const char* const file, const int line) {
  if (err != CUFFT_SUCCESS) {
    std::cerr << "CUFFT error at: " << file << ":" << line << std::endl;
    std::cerr << cufftGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

//S - opora.dat (size N), X - trace (L+N), W - output (size L)
void calcCorrelationNaive(float* S, int N, int L, float* X, float* W) {
    for (int j = 0; j < L; j++) {
        double acc = 0;
        for (int i = 0; i < N; i++) {
            acc += ((double)1.) * X[i + j] * S[i];
        }
        W[j] = acc;

    }
}

__global__ void ComplexPointwiseMulAndScale(cufftDoubleComplex *c, cufftDoubleComplex *a, const cufftDoubleComplex *b, int size, float scale) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

  cufftDoubleComplex tmp;
  for (int i = threadID; i < size; i += numThreads) {
    tmp = cuCmul(a[i], b[i]);
    c[i] = make_cuDoubleComplex(scale*cuCreal(tmp), scale*cuCimag(tmp));
  }
}

void calcCorrelationFFT(float *S, int N, int L, float *X, float* W) {
    int arraySize = N + (L + N) - 1;
    int complexSize = arraySize / 2 + 1;

    double* a = (double*) calloc(arraySize, sizeof(double));
    for (int i=0;i<N;i++) {
        a[i] = S[i];
    }
    std::reverse(a + 1, a + arraySize);

    cufftDoubleReal* dev_ab;
    checkCudaErrors(cudaMalloc((void**)&dev_ab, 2 * arraySize * sizeof(cufftDoubleReal)));
    checkCudaErrors(cudaMemcpy(dev_ab, a, arraySize * sizeof(double), cudaMemcpyHostToDevice));
    free(a);

    double* b = (double*) calloc(arraySize, sizeof(double));
    for (int i=0;i<L+N;i++) {
        b[i] = X[i];
    }

    checkCudaErrors(cudaMemcpy(dev_ab + arraySize, b, arraySize * sizeof(double), cudaMemcpyHostToDevice));
    free(b);

    cufftHandle plan;
    checkCufftErrors(cufftPlan1d(&plan, arraySize, CUFFT_D2Z, 2));

    cufftDoubleComplex* dev_ab_fft;
    checkCudaErrors(cudaMalloc((void**)&dev_ab_fft, 2 * complexSize * sizeof(cufftDoubleComplex)));
    checkCufftErrors(cufftExecD2Z(plan, dev_ab, dev_ab_fft));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(dev_ab));

    checkCufftErrors(cufftDestroy(plan));

    cufftDoubleComplex* dev_c_fft;
    checkCudaErrors(cudaMalloc((void**)&dev_c_fft, complexSize * sizeof(cufftDoubleComplex)));
    ComplexPointwiseMulAndScale<<<32, 256>>>(dev_c_fft, dev_ab_fft, dev_ab_fft + complexSize, complexSize, 1./arraySize);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(dev_ab_fft));
    
    cufftHandle planInv;
    checkCufftErrors(cufftPlan1d(&planInv, arraySize, CUFFT_Z2D, 1));

    cufftDoubleReal* dev_c;
    checkCudaErrors(cudaMalloc((void**)&dev_c, arraySize * sizeof(cufftDoubleReal)));
    checkCufftErrors(cufftExecZ2D(planInv, dev_c_fft, dev_c));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(dev_c_fft));

    checkCufftErrors(cufftDestroy(planInv));
    
    double* c = (double*) calloc(arraySize, sizeof(double));
    checkCudaErrors(cudaMemcpy(c, dev_c, L * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(dev_c));
    for (int i=0;i<L;i++) {
        W[i] = c[i];
    }
    free(c);
}

void normalize(float* W, int L, double* step) {
    double B = 0, C = 0;
    for (int i = 0; i < L; i++) if ((B = fabs(W[i])) > C) C = B;
    B = 32767. / C;

    for (int i = 0; i < L; i++) W[i] = (short) (W[i] * B);
    *step = B;
}
void validate(float *S, int N, int L, float *X) {
    float* W_naive = (float*) calloc(L, sizeof(float));
    float* W_fft = (float*) calloc(L, sizeof(float));
    calcCorrelationFFT(S, N, L, X, W_fft);
    calcCorrelationNaive(S, N, L, X, W_naive);
    {
        double maxAbsDiff = 0;
        double sumAbsDiff = 0;
        for (int i = 0; i < L; i++) {
            double val1 = W_naive[i];
            double val2 = W_fft[i];
            double absDiff = fabs(val1 - val2);
            sumAbsDiff += absDiff;
            maxAbsDiff = fmax(maxAbsDiff, absDiff);
        }
        double avgAbsDiff = sumAbsDiff / L;
        printf("max abs error: %f\n", maxAbsDiff);
        printf("avg abs error: %f\n", avgAbsDiff);
    }

    double stepNaive, stepFFT;
    normalize(W_naive, L, &stepNaive);
    normalize(W_fft, L, &stepFFT);

    {
        double maxAbsDiff = 0;
        double sumAbsDiff = 0;
        for (int i = 0; i < L; i++) {
            double val1 = ((double) 1.) * W_naive[i] * stepNaive;
            double val2 = ((double) 1.) * W_fft[i] * stepFFT;
            double absDiff = fabs(val1 - val2);
            sumAbsDiff += absDiff;
            maxAbsDiff = fmax(maxAbsDiff, absDiff);
        }
        double avgAbsDiff = sumAbsDiff / L;
        printf("max discrete error: %f\n", maxAbsDiff);
        printf("avg discrete error: %f\n", avgAbsDiff);
    }

    free(W_naive);
    free(W_fft);
}

#include <iostream>
using namespace std;

int main(int argc,char**argv) {

    if ((argc != 4)) {

        printf("\
   ╔══════════════════════════════════════════════════════╗\n\
   ║ CORR - корреляционная свертка сейсмотрасс    2010г.  ║\n\
   ║                                                      ║\n\
   ║ Вызов: corr.exe filename1 filename2 T                ║\n\
   ║ Вызов через меню Far: path\\corr.exe !@! filename2 T  ║\n\
   ║                                                      ║\n\
   ║  filename1 - текстовый файл со списком сейсмотрасс   ║\n\
   ║              в формате РС-A                          ║\n\
   ║  filename2 - файл опорного сигнала в формате РС-A    ║\n\
   ║          T - интервал корреляции в секундах          ║\n\
   ║                                                      ║\n\
   ║ Файлы результата записываются в поддиректорию _corr  ║\n\
   ║ Отчет о работе выводится на экран и в файл corr.log  ║\n\
   ║ В отчете для полученных коррелотрасс приводится      ║\n\
   ║ отношение сигнал/шум (SNR) и эквивалентная           ║\n\
   ║ амплитуда сигнала (A) в дискретах АЦП.               ║\n\
   ╚══════════════════════════════════════════════════════╝\n\
   ...press any key...");

        //getch();
        exit(1);
    }

    double A = 0, B = 0, C = 0;
    int h, p, N;
    int i, j;
    short Fs, st;
    char c;
    double sum = 0, sums = 0;

    if ((h = open(argv[2], O_RDONLY)) == -1) {
        perror("\tCan't open opora");
        exit(1);
    }
    read(h, &Fs, 2);
    if (Fs != 0x4350) {
        printf("\tInvalid format of %s", argv[2]);
        exit(1);
    }
    lseek(h, 32u, 0);
    read(h, &Fs, 2);  // read sampling frequency
    read(h, &N, 4);   //read sampl.num
    read(h, &st, 2);   //read sampl.type

    int L = atoi(argv[3]) * Fs;
    int M = N + L;

    struct stat stbuf;
    if (stat("_corr", &stbuf) == -1)
        if (mkdir("_corr", 0777)) {
            printf("\nCan't create directory.");
            exit(1);
        }

    float *X = (float *) calloc(M, sizeof(float)), //current trace samples
    *S = (float *) calloc(M, sizeof(float)),//opora.dat samples
    *W = (float *) calloc(M, sizeof(float)); //autocorellation
    int *D = (int *) calloc(M, 4);
    short *D16 = (short *) calloc(M, 2),
            *d = (short *) calloc(21u, 2);

    lseek(h, 42u, 0);

    int a = 0, b = 0; // a - max abs item
    //A - sum squares
    if (st == 2) {
        read(h, D16, N * 2);
        for (i = 0; i < N; i++) {
            *(S + i) = *(D16 + i);
            if ((b = abs(*(D16 + i))) > a)a = b;
            A += *(D16 + i) * *(D16 + i);
            *(D16 + i) = 0;
        }
    } else {
        read(h, D, N * 4);
        for (i = 0; i < N; i++) {
            *(S + i) = *(D + i);
            if ((b = abs(*(D + i))) > a)a = b;
            A += *(D + i) * *(D + i);
            *(D + i) = 0;
        }
    }

    close(h);

    b = (int) (0.5 * L);

    char trace[40], str[80];
    FILE *f, *ff;
    if ((f = fopen(argv[1], "r")) == NULL) {
        printf("\nCan't open filelist");
        exit(1);
    }
    if ((ff = fopen("corr.log", "a")) == NULL) {
        printf("\nCan't open log file");
        exit(1);
    }

    printf("\nProcessing: ");
    fprintf(ff, "\nProcessing: ");

    while (fscanf(f, "%s", trace) != EOF) {
        printf("\n%s - ", trace);
        fprintf(ff, "\n%s - ", trace);
        if ((h = open(trace, O_RDONLY)) == -1) {
            perror("Can't open trace");
            continue;
        }
        read(h, d, 42u);
        lseek(h, 41u, 0);
        read(h, &c, 1);
        if (*d != 0x4350) {
            printf("error: invalid format");
            fprintf(ff, "error: invalid format");
            continue;
        }
        if (*(d + 16) != Fs) {
            printf("error: sampling frequency mismatch");
            fprintf(ff, "error: sampling frequency mismatch");
            continue;
        }
        st = *(d + 19);

        for (i = 0; i < M; i++)*(W + i) = *(X + i) = *(D + i) = *(D16 + i) = 0;
        if (st == 2) {
            read(h, D16, M * 2);
            for (i = 0; i < M; i++)*(X + i) = *(D16 + i);
        }
        else {
            read(h, D, M * 4);
            for (i = 0; i < M; i++)*(X + i) = *(D + i);
        }
        close(h);
//      calcCorrelationNaive(S, N, L, X, W); //naive convolution
        calcCorrelationFFT(S, N, L, X, W);
//        validate(S, N, L, X); //validation
//        break;

        for (i = 0; i < L; i++) if ((B = fabs(*(W + i))) > C) C = B;
        B = 32767. / C; // C-correlation

        for (i = 0; i < L; i++) *(D16 + i) = (short) (*(W + i) * B);

        B = C / A * a; // real amplituda

        sum = 0;
        sums = 0;
        for (i = b; i < L; i++) {
            sum += *(W + i);
            sums += *(W + i) * *(W + i);
        } //b=1.5*L;
        C = C / sqrt((sums - sum * sum / (L - b)) / (L - b - 1)); // SNR


        strcpy(str, "_corr\\");
        strcat(str, trace);

        if ((h = open(str, O_WRONLY | O_CREAT | O_TRUNC, 0777)) == -1) {
            perror(" error open file ");
            exit(1);
        }
        write(h, d, 42u);
        lseek(h, 14u, 0);
        write(h, &B, 8);
        lseek(h, 34u, 0);
        write(h, &L, 4);
        st = 2;
        c += 10;
        write(h, &st, 2); //write(h, &c, 1);
        lseek(h, 41u, 0);
        write(h, &c, 1);

        if (write(h, D16, 2 * L) == -1) {
            perror(" error writing file ");
            exit(1);
        }   // write data
        close(h);
        printf("OK\tA=%.2f\tSNR=%.1f", B, C);
        fprintf(ff, "OK\tA=%.2f\tSNR=%.1f", B, C);
        B = C = 0;
    }

    clock_t t = clock();
    printf("\nprocessing time: %.2fs\n", ((double) t) / CLOCKS_PER_SEC);
    fprintf(ff, "\nprocessing time: %.2fs\n", ((double) t) / CLOCKS_PER_SEC);
    fclose(f);
    fclose(ff);

    return 0;
}

