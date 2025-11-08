import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Dataset information with actual computational efficiency data
datasets = [
    {
        'name': 'Chest CT',
        'color': 'orange',
        'inference_times': [0.0788, 0.0743, 0.0784, 0.0862, 0.0927, 0.0902]
    },
    {
        'name': 'Kvasir Seg',
        'color': 'green',
        'inference_times': [0.0745, 0.0813, 0.0958, 0.1201, 0.1327, 0.1510]
    },
    {
        'name': 'BUSI (Breast Ultrasound)',
        'color': 'blue',
        'inference_times': [0.0774, 0.0834, 0.0994, 0.1219, 0.1334, 0.1329]
    },
    {
        'name': 'HAM10000 (Skin Lesions)',
        'color': 'red',
        'inference_times': [0.0895, 0.0947, 0.1056, 0.1316, 0.1409, 0.1492]
    },
    {
        'name': 'Brain MRI',
        'color': 'purple',
        'inference_times': [0.0763, 0.0882, 0.0974, 0.1226, 0.1336, 0.1338]
    }
]

# Common batch sizes for all datasets
batch_sizes = [1, 2, 4, 8, 16, 32]

# Create figure with two subplots
plt.figure(figsize=(16, 6))

# Plot 1: Inference Time vs Batch Size
plt.subplot(1, 2, 1)
for dataset in datasets:
    plt.plot(batch_sizes, dataset['inference_times'], 'o-', 
             color=dataset['color'], label=dataset['name'])

plt.xlabel('Batch Size')
plt.ylabel('Inference Time (s)')
plt.title('Inference Time vs Batch Size')
plt.grid(True)
plt.legend()

# Plot 2: Throughput vs Batch Size
plt.subplot(1, 2, 2)
for dataset in datasets:
    throughput = [b/t for b, t in zip(batch_sizes, dataset['inference_times'])]
    plt.plot(batch_sizes, throughput, 'o-', 
             color=dataset['color'], label=dataset['name'])

plt.xlabel('Batch Size')
plt.ylabel('Throughput (images/s)')
plt.title('Throughput vs Batch Size')
plt.grid(True)
plt.legend()

plt.tight_layout()
output_path = 'combined_computational_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Combined computational analysis plot saved to: {os.path.abspath(output_path)}")
