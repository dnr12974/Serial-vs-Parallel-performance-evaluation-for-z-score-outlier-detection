import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read benchmark results
df = pd.read_csv('benchmark_results.csv')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Z-Score Outlier Detection: Serial vs Parallel Performance Analysis', 
             fontsize=16, fontweight='bold')

# Plot 1: Execution Time Comparison
ax1 = axes[0, 0]
threads = df['Threads'].unique()
x = np.arange(len(threads))
width = 0.35

serial_times = [df[df['Threads']==t]['SerialTime'].values[0] for t in threads]
parallel_times = [df[df['Threads']==t]['ParallelTime'].values[0] for t in threads]

bars1 = ax1.bar(x - width/2, serial_times, width, label='Serial', color='#e74c3c', alpha=0.8)
bars2 = ax1.bar(x + width/2, parallel_times, width, label='Parallel', color='#3498db', alpha=0.8)

ax1.set_xlabel('Number of Threads', fontweight='bold')
ax1.set_ylabel('Execution Time (seconds)', fontweight='bold')
ax1.set_title('Execution Time: Serial vs Parallel')
ax1.set_xticks(x)
ax1.set_xticklabels(threads)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s',
                ha='center', va='bottom', fontsize=8)

# Plot 2: Speedup vs Thread Count
ax2 = axes[0, 1]
ax2.plot(df['Threads'], df['Speedup'], marker='o', linewidth=2, 
         markersize=8, color='#2ecc71', label='Actual Speedup')
ax2.plot(df['Threads'], df['Threads'], linestyle='--', linewidth=2, 
         color='#95a5a6', label='Ideal Speedup')
ax2.set_xlabel('Number of Threads', fontweight='bold')
ax2.set_ylabel('Speedup', fontweight='bold')
ax2.set_title('Speedup vs Number of Threads')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add data labels
for i, row in df.iterrows():
    ax2.text(row['Threads'], row['Speedup'], f"{row['Speedup']:.2f}x",
            ha='center', va='bottom', fontsize=9)

# Plot 3: Parallel Efficiency
ax3 = axes[1, 0]
efficiency_pct = df['Efficiency'] * 100
colors = ['#e74c3c' if e < 70 else '#f39c12' if e < 85 else '#2ecc71' 
          for e in efficiency_pct]
bars = ax3.bar(df['Threads'], efficiency_pct, color=colors, alpha=0.8, edgecolor='black')
ax3.axhline(y=100, color='#95a5a6', linestyle='--', linewidth=2, label='Ideal (100%)')
ax3.set_xlabel('Number of Threads', fontweight='bold')
ax3.set_ylabel('Efficiency (%)', fontweight='bold')
ax3.set_title('Parallel Efficiency')
ax3.set_ylim([0, 110])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

# Plot 4: Performance Summary Table
ax4 = axes[1, 1]
ax4.axis('tight')
ax4.axis('off')

table_data = []
for _, row in df.iterrows():
    table_data.append([
        int(row['Threads']),
        f"{row['SerialTime']:.4f}s",
        f"{row['ParallelTime']:.4f}s",
        f"{row['Speedup']:.2f}x",
        f"{row['Efficiency']*100:.1f}%"
    ])

table = ax4.table(cellText=table_data,
                  colLabels=['Threads', 'Serial', 'Parallel', 'Speedup', 'Efficiency'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.15, 0.2, 0.2, 0.2, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, len(table_data) + 1):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        else:
            table[(i, j)].set_facecolor('white')

ax4.set_title('Performance Summary', fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('performance_graphs.png', dpi=300, bbox_inches='tight')
print(" Graphs saved to performance_graphs.png")
plt.show()

