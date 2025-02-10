import json
import os
import numpy as np
directory = '/sensei-fs/users/hyou/Efficient-Diffusion/Efficient-Diffusion/t2i-512/pixart_sigma_adaptive_ratio_0.2_timewise_lora/ratios_3'
filename = 'ratios_step57002.json'

layer_dir = '/sensei-fs/users/hyou/Efficient-Diffusion/Efficient-Diffusion/t2i-512/pixart_sigma_adaptive_ratio_0.2_lora/ratios_0.3'
layer_file = 'ratios_step37500.json'
with open(os.path.join(directory, filename), 'r') as f:
    ratios = json.load(f)
    del ratios["avg_ratio"]
with open(os.path.join(layer_dir, layer_file), 'r') as f:
    layer_ratios = json.load(f)
    del layer_ratios["avg_ratio"]
layer_ratios = np.array(list(layer_ratios.values()))
np_array = np.array(list(ratios.values()))
print(np_array.shape)
np.save('epoch_1_step_57k.npy', np_array)
sorted_values = sorted(list(np_array.reshape(-1)))

drop_ratio = 1 - 0.8 / 0.8744981150541987
# drop_ratio = 0.1
index_20_percentile = int(len(sorted_values) * drop_ratio)
k = sorted_values[index_20_percentile]
print(sorted_values)

print(f"The threshold value k such that 20% of the values are smaller than k is: {k}")

mask = np_array > k
layer_ratios = np.zeros((28, 4)) + layer_ratios.reshape(-1,1)
layer_ratios = layer_ratios * mask
layer_ratios = np.clip(layer_ratios, 0, 1)
print(f"Actual Avg Ratio: {np.mean(layer_ratios)}")