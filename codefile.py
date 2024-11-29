import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

data = pd.read_csv('50_Startups.csv')

X = data.drop('Profit', axis=1)
y = data['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

predictor = LinearRegression(n_jobs =-1)
predictor.fit(X = X, y = y)

y_pred = predictor.predict(X_test)


mse_linearregression = mean_squared_error(y_test, y_pred)
r2_linearregression = r2_score(y_test, y_pred)
print(f"Linear Regression - MSE: {mse_linearregression}, R-squared: {r2_linearregression}")

dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(X_train, y_train)
y_pred_dt = dt_regressor.predict(X_test)
mse_decisiontree = mean_squared_error(y_test, y_pred_dt)
r2_decisiontree = r2_score(y_test, y_pred_dt)
print(f"Decision Tree Regression - MSE: {mse_decisiontree}, R-squared: {r2_decisiontree}")

rf_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
rf_regressor.fit(X_train, y_train)
y_pred_rf = rf_regressor.predict(X_test)
mse_randomforest = mean_squared_error(y_test, y_pred_rf)
r2_randomforest = r2_score(y_test, y_pred_rf)
print(f"Random Forest Regression - MSE: {mse_randomforest}, R-squared: {r2_randomforest}")


svr_regressor = SVR(kernel='rbf')
svr_regressor.fit(X_train, y_train)
y_pred_svr = svr_regressor.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)
print(f"Support Vector Regression - MSE: {mse_svr}, R-squared: {r2_svr}")


models = {
    "Linear Regression": (mse_linearregression, r2_linearregression),
    "Decision Tree Regression": (mse_decisiontree, r2_decisiontree),
    "Random Forest Regression": (mse_randomforest, r2_randomforest),
    "Support Vector Regression": (mse_svr, r2_svr)
}
best_model = max(models, key=lambda k: models[k][1])
print(f"\nBest Model based on R-squared: {best_model} (R-squared: {models[best_model][1]})")
