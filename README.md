# Z-Score Outlier Detection: Serial vs Parallel

A performance comparison of serial and OpenMP-parallel implementations of z-score outlier detection for large datasets.

## Requirements

- GCC with OpenMP support
- Python 3 with pandas, matplotlib, numpy
- Dataset in CSV format

Install Python dependencies:

pip install pandas matplotlib numpy

text

## How to Run

### Step 1: Compile

gcc -o outlier_detection outlier_detection_enhanced.c -fopenmp -lm -O3 -Wall

text

### Step 2: Run with Dataset

./outlier_detection your_dataset.csv

Example with 5M Sales Records:

./outlier_detection 5m Sales Records.csv

text

### Step 3: Generate Graphs

python3 plot_results.py

text

This creates `performance_graphs.png` with speedup and efficiency visualizations.

## Optional: Control Thread Count

export OMP_NUM_THREADS=4

./outlier_detection dataset.csv

text

## What You'll Get

- Serial vs Parallel execution times
- Speedup metrics (1, 2, 4 threads)
- Parallel efficiency calculations
- Number of outliers detected
- Results saved to `benchmark_results.csv`
- Performance graphs in `performance_graphs.png`

## Tested Datasets

- ExcelBIA 5M Sales Records 
- UNSW-NB15 Network Traffic (2.5M rows)
- MNIST4OD (17K rows)

---

**Note:** Parallel speedup is only visible with large datasets (1M+ rows). Small datasets show overhead effects.
