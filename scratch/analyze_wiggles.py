import pandas as pd
import numpy as np

df = pd.read_csv("outputs/predictions_full.csv")
print("First 10 rows of outputs/predictions_full.csv:")
print(df.head(10))

# Calculate absolute difference between true and pred
diff_x = df["true_x"] - df["pred_x"]
diff_y = df["true_y"] - df["pred_y"]
diff_z = df["true_z"] - df["pred_z"]

print("\nPrediction deviations:")
print(f"  Max X diff: {diff_x.abs().max():.4f} m")
print(f"  Max Y diff: {diff_y.abs().max():.4f} m")
print(f"  Max Z diff: {diff_z.abs().max():.4f} m")
print(f"  Mean X diff: {diff_x.mean():.4f} m, Std: {diff_x.std():.4f} m")
print(f"  Mean Y diff: {diff_y.mean():.4f} m, Std: {diff_y.std():.4f} m")
print(f"  Mean Z diff: {diff_z.mean():.4f} m, Std: {diff_z.std():.4f} m")
