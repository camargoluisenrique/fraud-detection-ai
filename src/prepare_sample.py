import pandas as pd

df = pd.read_csv("data/fraud.csv")

# separar clases
fraud = df[df["Class"] == 1]
normal = df[df["Class"] == 0].sample(len(fraud)*5, random_state=42)

# combinar + shuffle + reset index
df_sample = pd.concat([fraud, normal]).sample(frac=1, random_state=42).reset_index(drop=True)

# guardar
df_sample.to_csv("data/fraud_sample.csv", index=False)

# validación
print("Sample creado:", df_sample.shape)
print(df_sample["Class"].value_counts())