import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
data = pd.read_csv(r'C:\Users\meila\OneDrive\Stalinis kompiuteris2\lab4_DI\dataset.csv')

# 2. Data exploration
print(data.head())  
print(data.describe())  

# 3. Data preprocessing
data = data[['City', 'Room Type', 'Person Capacity', 'Cleanliness Rating', 'Guest Satisfaction', 'Bedrooms', 'City Center (km)', 'Metro Distance (km)', 'Price']]

label_encoder = LabelEncoder()
data['City'] = label_encoder.fit_transform(data['City'])
data['Room Type'] = label_encoder.fit_transform(data['Room Type'])

# Select numerical features for scaling
numerical_features = ['Person Capacity', 'Cleanliness Rating', 'Guest Satisfaction', 'Bedrooms', 'City Center (km)', 'Metro Distance (km)', 'Price']

# Standartize numeric features
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# 4. Regression modeling
X = data.drop('Price', axis=1)  
y = data['Price']  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement linear regression
# iesko tiesines funkcijos kuri labiausia tinka, tikrina tiesini rysi tarp nepriklausomu kintamuju
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Implement polynomial regression
# veikia kaip tiesine tik transformuoja nepriklausomus kintamuosius i polinominius terminus, gali imt daugyba tarp skaiciaus kvadratu 
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# Implement random forest regression
# Kiekvienas sprendimų medis mokomas atliekant atsitiktinę duomenų imtį ir nustatant, kuris kintamasis yra geriausias skirstymo kriterijus kiekvienam mazgui.
random_forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_reg.fit(X_train, y_train)

# Implement recurrent neural network (RNN)
# Jis naudoja atgalinio sklaidos algoritmą, kad galėtų "atminti" ankstesnes būsenas arba informaciją iš ankstesnių laiko žingsnių.
rnn_model = Sequential()
rnn_model.add(SimpleRNN(50, input_shape=(X_train.shape[1], 1)))
rnn_model.add(Dense(1))
rnn_model.compile(optimizer='adam', loss='mean_absolute_error')

# Reshape training data for RNN
X_train_rnn = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_rnn = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train the RNN model
rnn_model.fit(X_train_rnn, y_train, epochs=30, batch_size=32, verbose=1)

# 5. Model evaluation
models = {'Linear Regression': linear_reg, 'Polynomial Regression': poly_reg, 'Random Forest': random_forest_reg, 'Recurrent Neural Network': rnn_model}
for name, model in models.items():
    if name == 'Polynomial Regression':
        y_pred = model.predict(X_test_poly)
    elif name == 'Recurrent Neural Network':
        y_pred = rnn_model.predict(X_test_rnn)[:, 0]
    else:
        y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f'{name}: MAE = {mae}, MSE = {mse}')

# 6. Visualization
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

for name, model in models.items():
    if name == 'Polynomial Regression':
        y_pred = model.predict(X_test_poly)
    elif name == 'Recurrent Neural Network':
        y_pred = rnn_model.predict(X_test_rnn)[:, 0]
    else:
        y_pred = model.predict(X_test)
    axes[0].scatter(y_test, y_pred, label=name)

axes[0].set_xlabel('True Values')
axes[0].set_ylabel('Predicted Values')
axes[0].set_title('True vs. Predicted Values')
axes[0].legend()

for name, model in models.items():
    if name == 'Polynomial Regression':
        y_pred = model.predict(X_test_poly)
    elif name == 'Recurrent Neural Network':
        y_pred = rnn_model.predict(X_test_rnn)[:, 0]
    else:
        y_pred = model.predict(X_test)
    axes[1].scatter(y_test[y_test <= 10], y_pred[y_test <= 10], label=name)

axes[1].set_xlabel('True Values')
axes[1].set_ylabel('Predicted Values')
axes[1].set_title('True vs. Predicted Values (up to 10)')
axes[1].legend()

plt.tight_layout()
plt.show()
