import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- delivery_times.csv code mein banana: 100 deliveries ----
times = np.random.normal(loc=32, scale=8, size=100).round(1)
times = np.clip(times, 10, None)      # koi delivery 10 min se kam na ho
times[[10, 45, 80]] = [68, 71, 75]    # 3 jaan boojh kar bohat late deliveries
pd.DataFrame({"OrderID": range(1, 101), "DeliveryTime(min)": times}).to_csv(
    "delivery_times.csv", index=False
)

df = pd.read_csv("delivery_times.csv")
real = df["DeliveryTime(min)"]

# (a) real data ka mean aur std
mean, std = real.mean(), real.std()
print(f"Real delivery times -- Mean: {mean:.1f} min  Std: {std:.1f} min")

# (b) isi mean/std se 100 "expected" times simulate karna
simulated = np.random.normal(loc=mean, scale=std, size=100)
print(f"Simulated 100 values from Normal(mean={mean:.1f}, std={std:.1f})")

# (c) dono ke overlapping histograms
plt.figure(figsize=(8, 5))
bins = np.linspace(min(real.min(), simulated.min()), max(real.max(), simulated.max()), 20)
plt.hist(real, bins=bins, alpha=0.6, color="steelblue", label="Real", edgecolor="black")
plt.hist(simulated, bins=bins, alpha=0.6, color="indianred", label="Simulated", edgecolor="black")
plt.xlabel("Delivery time (min)")
plt.ylabel("Frequency")
plt.title("Real vs. Simulated Delivery Times")
plt.legend()
plt.tight_layout()
plt.savefig("delivery_comparison_histogram.png")
plt.close()

# (d) EDA: missing values, percentiles, boxplot
print("Missing values:", df["DeliveryTime(min)"].isnull().sum())
q1, q2, q3 = real.quantile([0.25, 0.50, 0.75])
print(f"Percentiles -- 25th: {q1:.1f}  50th: {q2:.1f}  75th: {q3:.1f}")

plt.figure(figsize=(4, 5))
plt.boxplot(real)
plt.ylabel("Delivery time (min)")
plt.title("Delivery Time Boxplot")
plt.tight_layout()
plt.savefig("delivery_boxplot.png")
plt.close()

# (e) IQR method se outliers
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"IQR outlier bounds: {lower:.1f} - {upper:.1f}")

# sirf unusually LONG deliveries (upper bound se zyada) flag karni hain
outliers = real[real > upper]
print(f"Outliers detected: {len(outliers)} deliveries ({sorted(outliers.tolist())} minutes)")
print(f"Percentage flagged as outliers: {len(outliers) / len(real) * 100:.1f}%")

print("Saved delivery_comparison_histogram.png, delivery_boxplot.png")
