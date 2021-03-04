import matplotlib.pyplot as plt
import numpy as np

# construct some data like what you have:
x = np.random.randn(100, 8)
mins = x.min(0)
maxes = x.max(0)
means = x.mean(0)
std = x.std(0)

# create stacked errorbars:
plt.errorbar(np.arange(8), means, std, fmt='ok', lw=3)
#plt.errorbar(np.arange(8), means, [means - mins, maxes - means],
#             fmt='.k', ecolor='gray', lw=1)
plt.xlim(-1, 8)
plt.show()