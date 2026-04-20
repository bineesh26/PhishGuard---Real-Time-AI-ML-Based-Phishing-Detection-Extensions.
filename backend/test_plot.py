import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

print("Starting plot test...")
try:
    plt.figure()
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.savefig('test_plot.png')
    print("Plot saved successfully.")
except Exception as e:
    print(f"Error: {e}")
