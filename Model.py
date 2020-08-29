from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def My_model(x_train, x_test, y_train, y_test):
    classifiers = [
    LogisticRegression(random_state=0),
    xgb.XGBClassifier(random_state=0),
    MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    ]
    j=0
    
    accuracy = []
    for classifier in classifiers:
        j=j+1
        pipe = Pipeline(steps=[ ('scaler', MinMaxScaler()),
                                ('preprocessor', PCA(n_components=2)),
                                ('classifier', classifier)])
        pipe.fit(x_train, y_train)
        prediction = pipe.predict(x_test)
        cnf_matrix = metrics.confusion_matrix(y_test, prediction)
        accuracy.append(cnf_matrix)
        return accuracy
     
    