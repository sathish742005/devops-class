

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


data = pd.read_csv("agriculture.csv")


X = data.drop("Yield", axis=1)   
y = data["Yield"]                

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 

y_pred = model.predict(X_test)


print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

plt.scatter(y_test, y_pred, color="green")
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("AI-based Yield Prediction")
plt.show()


new_data = pd.DataFrame({
    "Rainfall": [120],
    "Temperature": [28],
    "SoilQuality": [7],
    "Fertilizer": [50]
})
print("Predicted Yield:", model.predict(new_data)[0])