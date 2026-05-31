import numpy as np
import matplotlib.pyplot as plt

# load your magic square
grid = np.loadtxt("./runs/lunar_magic_9x9.csv", delimiter=",")

# get coordinates of each number 1–81
coords = {}
for i in range(9):
    for j in range(9):
        coords[int(grid[i,j])] = (j + 0.5, 8.5 - i)

# build sigil path
path_x = []
path_y = []
for n in range(1, 82):
    x, y = coords[n]
    path_x.append(x)
    path_y.append(y)

# plot
plt.figure(figsize=(6,6))
plt.plot(path_x, path_y, '-', linewidth=2, color='black')
plt.axis('equal')
plt.axis('off')
plt.title("Sigil of the Lunar 9×9 Magic Square")
plt.tight_layout()
plt.savefig("lunar_9x9_sigil.png", dpi=300)
plt.show()
