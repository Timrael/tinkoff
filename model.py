import pandas as pd
import pdb
from sklearn.pipeline import Pipeline
import sklearn.utils
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# from xgboost import cv, DMatrix, XGBClassifier

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='roc_auc')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def prepare_data(data, credit_count_data, over_credit_count_data):
    data['tariff_id'] = data['tariff_id'].astype(str)
    data['score_shk'] = data['score_shk'].str.replace(',','.').astype(float)
    data['credit_sum'] = data['credit_sum'].str.replace(',','.').astype(float)
    data.loc[data['monthly_income'].isnull(), 'monthly_income'] = 35000.0
    data.loc[data['monthly_income'] == 0, 'monthly_income'] = 35000.0
    data['perc_credit'] = data['credit_sum'] * 1.0 / data['monthly_income']
    data['living_region'] = data['living_region'].astype(str)
    from_replace = ['\s?(ОБЛАСТЬ|ОБЛ\.|ОБЛ|КРАЙ\.|КРАЙ|РЕСПУБЛИКА|РЕСП\.|РЕСП|Г\.\s|Г\s|\sГ|АО|Р-Н)\s?', '74', '98|САНКТ-ПЕТЕРБУРГ', 'ЕВРЕЙСКАЯБЛ', 'КАМЧАТСКАЯ|КАМЧАТС\?\?ИЙ', '(МОСКВА|МОСКВОСКАЯ|МЫТИЩИНСКИЙ)', '(САХА \(ЯКУТИЯ\)|САХА \/ЯКУТИЯ\/)', 'СЕВ\. ОСЕТИЯ - АЛАНИЯ', 'ХАНТЫ-МАНСИЙСКИЙ АВТОНОМНЫЙ ОКРУ- ЮГРА', 'ХАНТЫ-МАНСИЙСКИЙ АВТОНОМНЫЙ ОКРУ- Ю', '(ЧУВАШИЯ\sЧУВАШСКАЯ-|ЧУВАШСКАЯ\s?-\sЧУВАШИЯ)', 'БЛ ЕВРЕЙСКАЯ', 'БРЯНСКИЙ', 'ГОРЬКОВСКАЯ', 'ОРЁЛ', 'ПЕРМСКАЯ', 'ПРИВОЛЖСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'ЭВЕНКИЙСКИЙ', 'nan|ГУСЬ-ХРУСТАЛЬНЫЙ|МОСКОВСКИЙ\sП|РОССИЯ', 'ХАНТЫ-МАНСИЙСКИЙ-ЮГРА', 'ЧЕЛЯБИНСК$', 'ЧИТИНСКАЯ', 'ЧУКОТСКИЙ\sАO', 'Г.МОСКОВСКАЯ', 'Г.ОДИНЦОВО\sМОСКОВСКАЯ', 'ДАЛЬНИЙ\sВОСТОК']
    to_replace = ['', 'ЧЕЛЯБИНСКАЯ', 'ЛЕНИНГРАДСКАЯ', 'ЕВРЕЙСКАЯ АВТОНОМНАЯ', 'КАМЧАТСКИЙ', 'МОСКОВСКАЯ', 'САХА', 'СЕВЕРНАЯ ОСЕТИЯ - АЛАНИЯ', 'ХАНТЫ-МАНСИЙСКИЙ', 'ХАНТЫ-МАНСИЙСКИЙ', 'ЧУВАШСКАЯ', 'ЕВРЕЙСКАЯ АВТОНОМНАЯ', 'БРЯНСКАЯ', 'НИЖЕГОРОДСКАЯ', 'ОРЛОВСКАЯ', 'ПЕРМСКИЙ', 'МОСКОВСКАЯ', 'КРАСНОЯРСКИЙ', 'МОСКОВСКАЯ', 'ХАНТЫ-МАНСИЙСКИЙ', 'ЧЕЛЯБИНСКАЯ', 'ЗАБАЙКАЛЬСКИЙ', 'ЧУКОТСКИЙ', 'МОСКОВСКАЯ', 'МОСКОВСКАЯ', 'ПРИМОРСКИЙ']
    data['living_region'].replace(from_replace, to_replace, regex=True, inplace=True)

    data = pd.merge(data, credit_count_data, on='client_id', how='left')
    data = pd.merge(data, over_credit_count_data, on='client_id', how='left')
    repl_rows = data['credit_count'].isnull()
    repl_rows_occ = data['overdue_credit_count'].isnull()
    data.loc[repl_rows,'credit_count'] = data.loc[repl_rows,'credit_count_new']
    data.loc[repl_rows_occ,'overdue_credit_count'] = data.loc[repl_rows_occ,'overdue_credit_count_new']
    data = data.drop(['client_id', 'credit_count_new', 'overdue_credit_count_new'], axis=1)
    return data

def prepare_features(data, scaler, numeric_columns, categorical_columns):
    data = pd.get_dummies(data, columns=categorical_columns)
    data[numeric_columns] = scaler.transform(data[numeric_columns])
    return data

def prepare_sets(train_data, test_data, credit_count_data, over_credit_count_data):
    categorical_columns = ['gender', 'marital_status', 'job_position', 'tariff_id', 'education', 'living_region']
    numeric_columns = ['age', 'credit_sum', 'credit_month', 'monthly_income', 'credit_count', 'overdue_credit_count']

    test_client_id = test_data['client_id'].copy()
    train_data = prepare_data(train_data, credit_count_data, over_credit_count_data)
    test_data = prepare_data(test_data, credit_count_data, over_credit_count_data)

    super_data = pd.concat([train_data, test_data])
    scaler = preprocessing.MinMaxScaler()

    scaler.fit(super_data[numeric_columns].dropna())
    train_data = prepare_features(train_data, scaler, numeric_columns, categorical_columns)
    test_data = prepare_features(test_data, scaler, numeric_columns, categorical_columns)
    test_data.loc[:,'tariff_id_1.96'] = 0.0
    test_data.loc[:,'tariff_id_1.52'] = 0.0

    X_train = train_data.drop(['open_account_flg'], axis=1)
    y_train = train_data['open_account_flg']
    X_test = test_data
    return X_train, y_train, X_test, test_client_id

def tune_model(clf_name, clf, param_grid, X_train, y_train):
    CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='roc_auc')
    print('TUNNING ' + clf_name)
    CV_rfc.fit(X_train, y_train)
    print(CV_rfc.best_params_)
    print('Best auc = ' + str(CV_rfc.best_score_))

def fit(clf_name, clf, X_train, y_train):
    print('Fitting ' + clf_name)
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)

    if hasattr(clf, "predict_proba"):
        train_predprob = clf.predict_proba(X_train)[:, 1]
    else:  # use decision function
        train_predprob = clf.decision_function(X_train)
        train_predprob = \
            (train_predprob - train_predprob.min()) / (train_predprob.max() - train_predprob.min())

    print('CV ' + clf_name)
    cv_score = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc', verbose=2, n_jobs=1)

    if type(y_train) == pd.DataFrame:
        print('Accuracy: %f' % accuracy_score(y_train.values, train_pred))
    else:
        print('Accuracy: %f' % accuracy_score(y_train, train_pred))

    print('AUC Score: %f' % roc_auc_score(y_train, train_predprob))
    print('CV Score: min - %f, max - %f, mean - %f, std – %f' % (
        np.min(cv_score),
        np.max(cv_score),
        np.mean(cv_score),
        np.std(cv_score)
    ))

    return np.mean(cv_score)

train_data = pd.read_csv('credit_train.csv', encoding='cp1251', delimiter=';')
test_data = pd.read_csv('credit_test.csv', encoding='cp1251', delimiter=';')
credit_count_data = pd.read_csv('regres_credit_count.csv', encoding='cp1251', delimiter=',')
over_credit_count_data = pd.read_csv('regres_overdue_credit_count.csv', encoding='cp1251', delimiter=',')
X_train, y_train, X_test, test_client_id = prepare_sets(train_data, test_data, credit_count_data, over_credit_count_data)

# Create classifiers
lr = LogisticRegression()
gnb = GaussianNB()
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier(n_estimators=100, max_depth=19, min_samples_split=25, max_features=0.2, min_samples_leaf=2)
etc = ExtraTreesClassifier(n_estimators=100, max_depth=19, min_samples_split=25, max_features=0.2)
gbc = GradientBoostingClassifier()
# KerasClassifier
# ExtraTreesClassifier
# GradientBoostingClassifier
# XGBClassifier
# xgb = XGBClassifier(
#         learning_rate=0.05,
#         n_estimators=738,
#         max_depth=5,
#         subsample=0.9,
#         colsample_bytree=0.6,
#         nthread=4
#     )

mlpc = MLPClassifier(hidden_layer_sizes=(8,8), batch_size=200, activation='tanh', solver='lbfgs', alpha=1.6384)
ada = AdaBoostClassifier(n_estimators=300)

eclf = VotingClassifier(estimators=[
                  ('Logistic', lr),
                  ('Random Forest', rfc),
                  ('GradientBoostingClassifier', gbc),
                  ('AdaBoostClassifier', ada),
                  ('MLPClassifier', mlpc),
                  ('ExtraTreesClassifier', etc),
                  ], voting='soft', weights=[0.5,4,4,3,1,1])

for clf, name in [
                  (lr, 'Logistic'),
                  (rfc, 'Random Forest'),
                  (etc, 'ExtraTreesClassifier'),
                  (ada, 'AdaBoostClassifier'),
                  (mlpc, 'MLPClassifier'),
                  (svc, 'SVC'),
                  (eclf, 'Voting Classifier'),
                  ]:

    fit(name, clf, X_train, y_train)
    print(name)
    plot_learning_curve(clf, name, X_train, y_train)
    plt.show()
    print('done with ' + name)

y_predict = pd.DataFrame(pd.Series(eclf.predict_proba(X_test)[:, 1]))
result = pd.DataFrame(test_client_id)
result['_ID_'] = test_client_id
result = result.drop(['client_id'], axis=1)
result['_VAL_'] = y_predict

result.to_csv('predict.csv', encoding='cp1251', index=False)
