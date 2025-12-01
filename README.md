# Z-Score Outlier Detection: Serial vs Parallel (OpenMP)

A complete performance comparison of **serial vs OpenMP-parallel implementations of Z-score based outlier detection** on large-scale datasets. This project benchmarks execution time, speedup, and parallel efficiency using real-world and synthetic datasets.

---

## Project Overview

This project implements a **two-pass Z-score based outlier detection algorithm** in:

- **Serial (single-threaded) C**
- **Parallel (multi-threaded) C using OpenMP**

The goal is to:

- Detect statistical outliers from large numeric datasets.
- Analyze the scalability of the algorithm.
- Measure **speedup and efficiency** across multiple threads.
- Visualize benchmarking results using Python.

---

## Project Structure

```text
.
├── outlier_detection_enhanced.c   # Main Serial + Parallel C implementation
├── plot_results.py                # Python script to generate graphs
├── benchmark_results.csv          # Auto-generated benchmark results
├── performance_graphs.png         # Output graph (speedup & efficiency)
├── README.md                      # Documentation
└── .gitignore                     # Ignores large datasets and compiled binaries
```

## Requirements

- GCC with OpenMP support
- Python 3 with pandas, matplotlib, numpy
- Dataset in CSV format

Install Python dependencies:
```bash
pip install pandas matplotlib numpy
```


## How to Run

### Step 1: Compile
```bash
gcc -o outlier_detection outlier_detection_enhanced.c -fopenmp -lm -O3 -Wall
```


### Step 2: Run with Dataset
```bash
./outlier_detection your_dataset.csv
```
Example with 5M Sales Records:
```bash
./outlier_detection 5m Sales Records.csv
```


### Step 3: Generate Graphs
```bash
python3 plot_results.py
```

This creates `performance_graphs.png` with speedup and efficiency visualizations.

## Optional: Control Thread Count

export OMP_NUM_THREADS=4
```bash
./outlier_detection dataset.csv
```

## What You'll Get

- Serial vs Parallel execution times
- Speedup metrics (1, 2, 4 threads)
- Parallel efficiency calculations
- Number of outliers detected
- Results saved to `benchmark_results.csv`
- Performance graphs in `performance_graphs.png`

## Tested Datasets

- ExcelBIA 5M Sales Records ✓
- UNSW-NB15 Network Traffic (2.5M rows)
- MNIST4OD (17K rows)

---

**Note:** Parallel speedup is only visible with large datasets (1M+ rows). Small datasets show overhead effects.
