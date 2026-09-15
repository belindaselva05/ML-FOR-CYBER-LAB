# Time Series Decomposition of Network Traffic

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Step 1: Create dates
dates = pd.date_range(start='2024-01-01', periods=120, freq='D')

# Step 2: Create components
trend = np.linspace(1000, 1800, 120)
seasonal = 200 * np.sin(np.arange(120) * 2 * np.pi / 7)
noise = np.random.normal(0, 50, 120)

# Step 3: Create network traffic
traffic = trend + seasonal + noise

# Step 4: Create DataFrame
data = pd.DataFrame({
    'Date': dates,
    'Network_Traffic': traffic
})

# Step 5: Set index
data.set_index('Date', inplace=True)

# Step 6: Plot original data
plt.figure(figsize=(12,4))
plt.plot(data.index, data['Network_Traffic'])
plt.title('Original Network Traffic')
plt.xlabel('Date')
plt.ylabel('Traffic')
plt.grid(True)
plt.show()

# Step 7: Apply decomposition
result = seasonal_decompose(
    data['Network_Traffic'],
    model='additive',
    period=7
)

# Step 8: Plot decomposition
result.plot()
plt.suptitle('Time Series Decomposition of Network Traffic', fontsize=14)
plt.show()

# Step 9: Save graph
result.plot()
plt.savefig('decomposition.png', dpi=300, bbox_inches='tight')
plt.show()

print('Experiment completed successfully')