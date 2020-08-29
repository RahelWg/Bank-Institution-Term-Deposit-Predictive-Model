import pandas as pd
import Data as da
import Model as mo

input = da.my_data()
data = da.remove_outliers(input)
data = pd.get_dummies(data, drop_first=True)

# Oversampling minority class to overcome class imbalance
df_upsampled = da.Over_sampler(data)

x1 = df_upsampled
y1 = df_upsampled['y_yes']

#split data into 90% training, 10% test data
x_train, x_test, y_train, y_test = da.train_test_split(df_upsampled , y1, test_size=0.1)

y= mo.My_model(x_train, x_test, y_train, y_test)
print(y)

