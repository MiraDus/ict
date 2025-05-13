import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Расширенный синтетический датасет
data = [
    ["ALA", "IST", 90000, "KC", 1],
    ["ALA", "IST", 85000, "PC", 1],
    ["ALA", "IST", 79000, "J9", 0],
    ["ALA", "IST", 72000, "DV", 1],
    ["ALA", "IST", 99000, "FS", 0],
    ["ALA", "TSE", 40000, "KC", 1],
    ["ALA", "TSE", 38000, "PC", 0],
    ["ALA", "TSE", 36000, "DV", 1],
    ["TSE", "IST", 78000, "J9", 0],
    ["TSE", "IST", 82000, "KC", 1],
    ["TSE", "IST", 90000, "PC", 1],
    ["TSE", "ALA", 60000, "FS", 0],
    ["ALA", "IST", 50000, "PC", 1],
    ["ALA", "IST", 55000, "KC", 1],
]

df = pd.DataFrame(data, columns=["origin", "destination", "price", "airline", "liked"])

# Кодировка
le_dep = LabelEncoder()
le_arr = LabelEncoder()
le_airline = LabelEncoder()

df["origin_enc"] = le_dep.fit_transform(df["origin"])
df["destination_enc"] = le_arr.fit_transform(df["destination"])
df["airline_enc"] = le_airline.fit_transform(df["airline"])

X = df[["origin_enc", "destination_enc", "price"]].values
y = pd.get_dummies(df["liked"]).values  # One-hot (0/1)

# Модель
model = Sequential()
model.add(Dense(16, input_dim=3, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(2, activation="softmax"))  # Классы: [0, 1]
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X, y, epochs=50, verbose=0)

# Сохранение
model.save("flight_model.h5")
joblib.dump(le_dep, "le_dep.pkl")
joblib.dump(le_arr, "le_arr.pkl")
joblib.dump(le_airline, "le_airline.pkl")

print("✅ Модель и энкодеры успешно сохранены.")