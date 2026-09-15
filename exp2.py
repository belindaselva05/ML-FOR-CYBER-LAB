import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ==========================================================
# EXPERIMENT 2: TIME SERIES FORECASTING USING ARIMA
# ==========================================================

print("TIME SERIES FORECASTING USING ARIMA")
print("====================================")


# ----------------------------------------------------------
# Step 1: Create Historical Network Traffic Data
# ----------------------------------------------------------

np.random.seed(42)

hours = np.arange(100)

# Normal network traffic
traffic = np.random.normal(
    loc=300,
    scale=40,
    size=100
)

# Simulated high traffic periods
traffic[70:75] = [600, 700, 800, 750, 650]

traffic[85:90] = [650, 750, 850, 900, 800]


print("\nHistorical Network Traffic:")
print("--------------------------------")

for i in range(10):
    print(
        "Hour", i + 1,
        "->",
        round(traffic[i], 2)
    )

print("\nTotal number of records:", len(traffic))


# ----------------------------------------------------------
# Step 2: Plot Historical Traffic
# ----------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    hours,
    traffic,
    label="Historical Network Traffic"
)

plt.title("Historical Network Traffic")
plt.xlabel("Time (Hours)")
plt.ylabel("Traffic Volume")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "historical_network_traffic.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ----------------------------------------------------------
# Step 3: ARIMA Differencing
# ----------------------------------------------------------

# ARIMA(1,1,1)
# d = 1 means first-order differencing

differenced_data = np.diff(traffic)

print("\nFirst-order differencing completed.")

print(
    "Number of differenced observations:",
    len(differenced_data)
)


# ----------------------------------------------------------
# Step 4: Define ARIMA(1,1,1) Model
# ----------------------------------------------------------

def arima_sse(parameters, data):

    c = parameters[0]
    phi = parameters[1]
    theta = parameters[2]

    errors = np.zeros(len(data))

    predictions = np.zeros(len(data))

    for t in range(1, len(data)):

        predictions[t] = (
            c
            + phi * data[t - 1]
            + theta * errors[t - 1]
        )

        errors[t] = (
            data[t]
            - predictions[t]
        )

    # Ignore first observation
    sse = np.sum(errors[1:] ** 2)

    return sse


# ----------------------------------------------------------
# Step 5: Estimate ARIMA Parameters
# ----------------------------------------------------------

initial_parameters = [
    0.0,     # constant
    0.2,     # AR coefficient
    0.2      # MA coefficient
]

result = minimize(
    arima_sse,
    initial_parameters,
    args=(differenced_data,),
    method="L-BFGS-B",
    bounds=[
        (None, None),
        (-0.99, 0.99),
        (-0.99, 0.99)
    ]
)

c = result.x[0]
phi = result.x[1]
theta = result.x[2]


print("\nARIMA(1,1,1) Model Parameters:")
print("--------------------------------")

print("Constant :", round(c, 4))
print("AR (phi) :", round(phi, 4))
print("MA (theta):", round(theta, 4))


# ----------------------------------------------------------
# Step 6: Calculate Model Errors
# ----------------------------------------------------------

errors = np.zeros(len(differenced_data))

for t in range(1, len(differenced_data)):

    predicted_difference = (
        c
        + phi * differenced_data[t - 1]
        + theta * errors[t - 1]
    )

    errors[t] = (
        differenced_data[t]
        - predicted_difference
    )


# ----------------------------------------------------------
# Step 7: Forecast Future Traffic
# ----------------------------------------------------------

forecast_steps = 10

future_differences = []

last_difference = differenced_data[-1]

last_error = errors[-1]


for i in range(forecast_steps):

    predicted_difference = (
        c
        + phi * last_difference
        + theta * last_error
    )

    future_differences.append(
        predicted_difference
    )

    last_difference = predicted_difference

    # Future error is assumed to be zero
    last_error = 0


# Convert differenced forecast back to
# original traffic scale

forecast = (
    traffic[-1]
    + np.cumsum(future_differences)
)


# ----------------------------------------------------------
# Step 8: Display Forecast
# ----------------------------------------------------------

future_hours = np.arange(
    len(traffic),
    len(traffic) + forecast_steps
)


print("\nPredicted Future Network Traffic:")
print("--------------------------------")

for hour, value in zip(
    future_hours,
    forecast
):

    print(
        "Hour",
        hour + 1,
        "->",
        round(value, 2)
    )


# ----------------------------------------------------------
# Step 9: Plot Forecast
# ----------------------------------------------------------

plt.figure(figsize=(12, 5))

# Historical traffic
plt.plot(
    hours,
    traffic,
    label="Historical Traffic"
)

# Forecast
plt.plot(
    future_hours,
    forecast,
    linestyle="--",
    marker="o",
    label="ARIMA Forecast"
)

plt.title(
    "Network Traffic Forecast Using ARIMA(1,1,1)"
)

plt.xlabel("Time (Hours)")
plt.ylabel("Traffic Volume")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "arima_network_forecast.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ----------------------------------------------------------
# Step 10: DDoS Detection
# ----------------------------------------------------------

threshold = 500

print("\nDDoS Attack Detection Results:")
print("--------------------------------")

attack_detected = False


for hour, value in zip(
    future_hours,
    forecast
):

    if value > threshold:

        print(
            "Hour",
            hour + 1,
            "->",
            round(value, 2),
            "-> POSSIBLE DDoS ATTACK"
        )

        attack_detected = True

    else:

        print(
            "Hour",
            hour + 1,
            "->",
            round(value, 2),
            "-> Normal Traffic"
        )


# ----------------------------------------------------------
# Step 11: Final Alert
# ----------------------------------------------------------

print("\nFinal Result:")
print("--------------------------------")

if attack_detected:

    print(
        "ALERT: Potential DDoS attack detected "
        "in forecasted traffic."
    )

else:

    print(
        "No potential DDoS attack detected "
        "in the forecasted traffic."
    )


print("\nExperiment 2 completed successfully.")