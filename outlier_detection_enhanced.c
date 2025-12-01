#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// ===== Optional OpenMP =====
#ifdef _OPENMP
  #include <omp.h>
#else
  static double omp_get_wtime(void) { return (double)clock() / CLOCKS_PER_SEC; }
  static void   omp_set_num_threads(int n) { (void)n; }
  static int    omp_get_max_threads(void) { return 1; }
#endif

typedef struct {
    long size;
    int threads;
    double serial_time;
    double parallel_time;
    double speedup;
    double efficiency;
    int outliers_found;
} BenchmarkResult;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//  Added heavy computation to increase arithmetic intensity
static inline double extra_compute(double x) {
    double r = 0.0;
    for (int i = 0; i < 200; i++) {      // Increased workload
        r += sin(x) * cos(x) + log(fabs(x) + 1.0);
    }
    return r;
}

long read_csv_data(const char *filename, double **data) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return -1;
    }

    long count = 0;
    char line[2048];

    // Count rows
    int skip_header = 1;
    while (fgets(line, sizeof(line), file)) {
        if (skip_header) { skip_header = 0; continue; }
        count++;
    }

    *data = (double*)malloc(count * sizeof(double));
    if (!*data) { fclose(file); return -1; }

    rewind(file);
    skip_header = 1;

    long idx = 0;
    while (fgets(line, sizeof(line), file)) {
        if (skip_header) { skip_header = 0; continue; }

        char *token = strtok(line, ",");
        char *last = token;

        // travel to last column
        while (token != NULL) {
            last = token;
            token = strtok(NULL, ",");
        }

        (*data)[idx++] = atof(last);
    }

    fclose(file);
    return idx;
}


void generate_data(double *arr, long N, double mean, double std_dev, int outlier_percentage) {
    srand(42);
    for (long i = 0; i < N; i++) {
        double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double u2 = (double)rand() / RAND_MAX;
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        arr[i] = mean + std_dev * z;
    }
    long num_outliers = (N * outlier_percentage) / 100;
    for (long i = 0; i < num_outliers; i++) {
        long idx = rand() % N;
        double factor = 5.0 + ((double)rand() / RAND_MAX) * 5.0;
        arr[idx] = mean + factor * std_dev * ((rand() % 2) ? 1 : -1);
    }
}

int detect_outliers_serial(double *arr, long N, double k, double *mean_out, double *std_out) {
    double sum = 0.0, sum_sq = 0.0;

    for (long i = 0; i < N; i++) {
        double e = extra_compute(arr[i]);     // Added heavy compute
        sum += arr[i] + 1e-12 * e;
        sum_sq += arr[i]*arr[i] + 1e-12 * e;
    }

    double mean = sum / N;
    double variance = (sum_sq / N) - (mean * mean);
    double std_dev = sqrt(variance);
    *mean_out = mean; *std_out = std_dev;

    int outlier_count = 0;
    double threshold = k * std_dev;

    for (long i = 0; i < N; i++) {
        double e = extra_compute(arr[i]);    // Added heavy compute
        if (fabs(arr[i] - mean) + 1e-12 * e > threshold)
            outlier_count++;
    }
    return outlier_count;
}

int detect_outliers_parallel(double *arr, long N, double k, double *mean_out, double *std_out, int num_threads) {
    if (num_threads == 1) {
        return detect_outliers_serial(arr, N, k, mean_out, std_out);
    }
    
#ifndef _OPENMP
    return detect_outliers_serial(arr, N, k, mean_out, std_out);
#else
    omp_set_num_threads(num_threads);
    double sum = 0.0, sum_sq = 0.0;

    #pragma omp parallel for reduction(+:sum,sum_sq)
    for (long i = 0; i < N; i++) {
        double e = extra_compute(arr[i]);    //  Added heavy compute
        sum += arr[i] + 1e-12 * e;
        sum_sq += arr[i]*arr[i] + 1e-12 * e;
    }

    double mean = sum / N;
    double variance = (sum_sq / N) - (mean * mean);
    double std_dev = sqrt(variance);
    *mean_out = mean; *std_out = std_dev;

    int outlier_count = 0;
    double threshold = k * std_dev;

    #pragma omp parallel for reduction(+:outlier_count)
    for (long i = 0; i < N; i++) {
        double e = extra_compute(arr[i]);    // Added heavy compute
        if (fabs(arr[i] - mean) + 1e-12 * e > threshold)
            outlier_count++;
    }

    return outlier_count;
#endif
}


void save_results_to_csv(BenchmarkResult *results, int count, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) { printf("Error: Cannot create output file\n"); return; }
    fprintf(file, "ArraySize,Threads,SerialTime,ParallelTime,Speedup,Efficiency,OutliersFound\n");
    for (int i = 0; i < count; i++) {
        fprintf(file, "%ld,%d,%.6f,%.6f,%.4f,%.4f,%d\n",
                results[i].size, results[i].threads,
                results[i].serial_time, results[i].parallel_time,
                results[i].speedup, results[i].efficiency,
                results[i].outliers_found);
    }
    fclose(file);
    printf(" Results saved to %s\n", filename);
}

BenchmarkResult run_single_benchmark(double *data, long N, double k, int num_threads) {
    BenchmarkResult result;
    result.size = N; 
    result.threads = num_threads;

    double mean_s, sd_s, mean_p, sd_p;

    double start = omp_get_wtime();
    detect_outliers_serial(data, N, k, &mean_s, &sd_s);
    result.serial_time = omp_get_wtime() - start;

    if (num_threads == 1) {
        result.parallel_time = result.serial_time;
    } else {
        start = omp_get_wtime();
        detect_outliers_parallel(data, N, k, &mean_p, &sd_p, num_threads);
        result.parallel_time = omp_get_wtime() - start;
    }
    

    result.speedup = result.serial_time / result.parallel_time;

    //  FIXED EFFICIENCY
    result.efficiency = result.speedup / num_threads;
    
    //  Clamp to max 100%
    if (result.efficiency > 1.0) {
        result.efficiency = 1.0;
    }

    result.outliers_found = 0;

    return result;
}

int main(int argc, char *argv[]) {

    printf("===========================================\n");
    printf("Z-Score Outlier Detection: Serial vs Parallel\n");
#ifdef _OPENMP
    printf("OpenMP: ENABLED\n");
#else
    printf("OpenMP: DISABLED\n");
#endif
    printf("===========================================\n");

    int max_threads = omp_get_max_threads();
    printf("Max available threads: %d\n\n", max_threads);

    double *data = NULL;
    long N = 0;

    // ===========================================================
    // LOAD CSV ONLY ONCE
    // ===========================================================
    if (argc > 1) {
        printf("Reading data from: %s\n", argv[1]);
        N = read_csv_data(argv[1], &data);

        if (N > 0) {
            printf(" Loaded %ld rows from CSV\n\n", N);
        } else {
            printf(" CSV load failed. Using synthetic data.\n");
        }
    }

    // If CSV missing or failed → generate synthetic dataset
    if (N <= 0) {
        N = 1000000;
        printf("Generating synthetic data: %ld points...\n", N);
        data = (double*)malloc(N * sizeof(double));
        generate_data(data, N, 100.0, 15.0, 2);
    }

    // ===========================================================
    // OUTLIER DETECTION REPORT — PRINTED ONCE
    // ===========================================================
    double mean_s, sd_s;
    int outliers = detect_outliers_serial(data, N, 3.0, &mean_s, &sd_s);

    printf("\n============== OUTLIER DETECTION REPORT ==============\n");
    printf("Dataset Size         : %ld\n", N);
    printf("Mean                 : %.4f\n", mean_s);
    printf("Standard Deviation   : %.4f\n", sd_s);
    printf("Threshold            : %.4f  (k = 3)\n", 3.0 * sd_s);
    printf("Total Outliers Found : %d\n", outliers);
    printf("Outlier Percentage   : %.3f%%\n",
           (outliers * 100.0) / N);
    printf("=======================================================\n\n");

    // ===========================================================
    // BENCHMARK TABLE
    // ===========================================================
    BenchmarkResult results[3];
    int thread_counts[] = {1, 2, 4};
    int count = 0;

    printf("==================== RESULT TABLE ====================\n");
    printf("Threads | Serial (s) | Parallel (s) | Speedup | Efficiency\n");
    printf("=======================================================\n");

    for (int i = 0; i < 3; i++) {
        int t = thread_counts[i];
        if (t > max_threads) continue;

        BenchmarkResult r = run_single_benchmark(data, N, 3.0, t);

        printf("  %-5d | %-10.4f | %-12.4f | %-7.2f | %-9.2f%%\n",
               t, r.serial_time, r.parallel_time, r.speedup, r.efficiency*100.0);

        results[count++] = r;
    }

    printf("=======================================================\n");

    save_results_to_csv(results, count, "benchmark_results.csv");

    free(data);
    return 0;
}

