#pragma GCC optimize("O3")

//#include <alloc.h>
#include <stdlib.h>
#include <stdio.h>
#include <io.h>
#include <fcntl.h>
#include <math.h>
#include <time.h>
#include <sys\stat.h>
#include <string.h>
#include <conio.h>
#include <dir.h>
#include <assert.h>
#include <complex.h>

typedef long double real_t;

typedef double complex_real_t;
typedef double _Complex complex_t;

enum bool {
    false,
    true
};

//S - opora.dat (size N), X - trace (L+N), W - output (size L)
void calcCorrelationNaive(float *S, int N, int L, float *X, float* W) {
    for (int j = 0; j < L; j++) {
        real_t acc = 0;
        for (int i = 0; i < N; i++) {
            acc += ((real_t) 1.) * X[i+j] * S[i];
        }
        W[j] = acc;
    }
}

void calc_rev(int n, int log_n, int* rev) {
    rev[0] = 0;
    for (int i=1; i<n; ++i) {
        //rev[i] = 1;
        rev[i] = (rev[i >> 1] >> 1) + ((i & 1) << (log_n - 1));
    }
}

void fft (complex_t* a, int n, enum bool invert, const int* rev, complex_t* wlen_pw) {
    complex_real_t PI = acosl(-1);
    for (int i=0; i<n; ++i)
        if (i < rev[i]) {
            complex_t tmp = a[i];
            a[i] = a[rev[i]];
            a[rev[i]] = tmp;
        }

    for (int len=2; len<=n; len<<=1) {
        complex_real_t ang = 2*PI/len * (invert?-1:+1);
        int len2 = len>>1;

        complex_t wlen = cos(ang) + sin(ang) * I;
        wlen_pw[0] = 1;
        for (int i=1; i<len2; ++i)
            wlen_pw[i] = wlen_pw[i-1] * wlen;

        for (int i=0; i<n; i+=len) {
            complex_t t,
                    *pu = a+i,
                    *pv = a+i+len2,
                    *pu_end = a+i+len2,
                    *pw = wlen_pw;
            for (; pu!=pu_end; ++pu, ++pv, ++pw) {
                t = *pv * *pw;
                *pv = *pu - t;
                *pu += t;
            }
        }
    }

    if (invert)
        for (int i=0; i<n; ++i)
            a[i] /= n;
}


void calcCorrelationFFT(float *S, int N, int L, float *X, float* W) {
    int LEN = N+L-1;//L + (N + L) - 1;
    int log2Len = 0;
    while ((1 << log2Len) < LEN) log2Len++;
    LEN = (1 << log2Len);

    complex_t* S_ = (complex_t*) calloc(LEN, sizeof(complex_t));
    complex_t* X_ = (complex_t*) calloc(LEN, sizeof(complex_t));
    for (int i=0;i<N;i++) {
        S_[i] = S[i];
    }
    for (int i=0;i<L+N;i++) {
        X_[i] = X[i];
    }

    int l = 1, r = LEN-1;//reverse S_ to transform corellation to convolution
    while (l < r) {
        complex_t tmp = S_[l];
        S_[l] = S_[r];
        S_[r] = tmp;
        l++; r--;
    }

    complex_t* W_ = (complex_t*) calloc(LEN, sizeof(complex_t));

    int* rev = (int*) calloc(LEN, sizeof(int));
    complex_t* wlen_pow = (complex_t*) calloc(LEN, sizeof(complex_t));
    calc_rev(LEN, log2Len, rev);

    /*fft(S_, LEN, false, rev, wlen_pow);
    fft(X_, LEN, false, rev, wlen_pow);*/
    for (int j=0;j<LEN;j++) {
        W_[j] = S_[j] + X_[j]*I;
    }
    fft(W_, LEN, false, rev, wlen_pow);
    for (int j=0;j<LEN;j++) {
        S_[j] = (W_[j] + conj(W_[LEN-j]))/2;
        X_[j] = (W_[j] - conj(W_[LEN-j]))/(2*I);
    }
    for (int j=0;j<LEN;j++) {
        W_[j] = S_[j] * X_[j];
    }
    fft(W_, LEN, true, rev, wlen_pow);
    free(wlen_pow);
    free(rev);

    for (int i=0;i<L;i++) {
        W[i] = creal(W_[i]);
    }

    free(S_);
    free(X_);
    free(W_);
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

        getch();
        exit(1);
    }

    double A = 0, B = 0, C = 0;
    int h, p, N;
    int i, j;
    short Fs, st;
    char c;
    double sum = 0, sums = 0;

    if ((h = open(argv[2], O_RDONLY | O_BINARY)) == -1) {
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
        if (mkdir("_corr")) {
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
        if ((h = open(trace, O_RDONLY | O_BINARY)) == -1) {
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

//        calcCorrelationNaive(S, N, L, X, W); //naive convolution
//        calcCorrelationFFT(S, N, L, X, W);
        validate(S, N, L, X); //validation
        break;

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

        if ((h = open(str, O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, 0777)) == -1) {
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

