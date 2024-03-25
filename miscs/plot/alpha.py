import matplotlib.pyplot as plt
import numpy as np

# Number of views
x = [
    -2.5,
    -1.5,
    -0.5,
    0.5,
    1.5,
    2.5,
    3.5
]
# Error
y1 = [
    [2.014, 1.87, 1.918, 1.865, 1.88],
    [0.5794, 0.2675, 0.5773, 0.3727, 0.5598],
    [0.3082, 0.3077, 0.3988, 0.5301, 0.4935],
    [0.5805, 0.5056, 0.8713, 1.006, 0.8651],
    [1.818, 1.151, 1.551, 1.399, 1.684],
    [1.52, 1.284, 1.687, 1.279, 2.249],
    [1.431, 1.894, 2.661, 2.429, 1.968]
]
# Distance
y2 = [
    [0.6079, 0.611, 0.5968, 0.6146, 0.6162],
    [0.4726, 0.4643, 0.4778, 0.4637, 0.4633],
    [0.492, 0.3822, 0.4385, 0.5012, 0.394],
    [0.4409, 0.4118, 0.4493, 0.4491, 0.4066],
    [0.443, 0.4638, 0.5004, 0.4919, 0.405],
    [0.3647, 0.4047, 0.383, 0.4307, 0.3865],
    [0.2251, 0.269, 0.234, 0.264, 0.2747]
]

x = np.array(x)
y1 = np.array(y1)
y2 = np.array(y2)

fig, ax1 = plt.subplots()

ax1.set_xlabel("Alpha (Log Scale)")
ax1.set_ylabel("Error")
ax1.errorbar(x, y1.mean(axis=-1), y1.std(axis=-1), fmt='o', color='tab:red',
             ecolor='pink', elinewidth=3, capsize=0)
ax1.plot(x, y1.mean(axis=-1), color="tab:red", label="Error")
ax1.legend(bbox_to_anchor=(0.3, 1), frameon=False)

ax2 = ax1.twinx()
ax2.set_ylabel("Distance")
ax2.errorbar(x, y2.mean(axis=-1), y2.std(axis=-1), fmt='o', color='tab:blue',
             ecolor='lightblue', elinewidth=3, capsize=0)
ax2.plot(x, y2.mean(axis=-1), color="tab:blue", label="Distance")
ax2.legend(bbox_to_anchor=(0.1, 0.9), frameon=False)

fig.tight_layout()
# save
plt.savefig("alpha.pdf")