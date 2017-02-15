import pandas as pd
import pdb
from sklearn.pipeline import Pipeline
import sklearn.utils
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, roc_auc_score

data = pd.read_csv('credit_train.csv', encoding='cp1251', delimiter=';').dropna()
test_data = pd.read_csv('credit_test.csv', encoding='cp1251', delimiter=';')

# features for scaling = age, credit_sum, credit_month, score_shk, monthly_income, credit_count(?), overdue_credit_count(?)
# features for one_to_many = gender, marital_status, job_position, tariff_id, education, living_region
data['tariff_id'] = data['tariff_id'].astype(str)
data['score_shk'] = data['score_shk'].str.replace(',','.').astype(float)
data['credit_sum'] = data['credit_sum'].str.replace(',','.').astype(float)
data['living_region'] = data['living_region'].astype(str)
from_replace = ['\s?(ОБЛАСТЬ|ОБЛ\.|ОБЛ|КРАЙ\.|КРАЙ|РЕСПУБЛИКА|РЕСП\.|РЕСП|Г\.\s|Г\s|\sГ|АО|Р-Н)\s?', '74', '98|САНКТ-ПЕТЕРБУРГ', 'ЕВРЕЙСКАЯБЛ', 'КАМЧАТСКАЯ|КАМЧАТС\?\?ИЙ', '(МОСКВА|МОСКВОСКАЯ|МЫТИЩИНСКИЙ)', '(САХА \(ЯКУТИЯ\)|САХА \/ЯКУТИЯ\/)', 'СЕВ\. ОСЕТИЯ - АЛАНИЯ', 'ХАНТЫ-МАНСИЙСКИЙ АВТОНОМНЫЙ ОКРУ- ЮГРА', 'ХАНТЫ-МАНСИЙСКИЙ АВТОНОМНЫЙ ОКРУ- Ю', '(ЧУВАШИЯ\sЧУВАШСКАЯ-|ЧУВАШСКАЯ\s?-\sЧУВАШИЯ)', 'БЛ ЕВРЕЙСКАЯ', 'БРЯНСКИЙ', 'ГОРЬКОВСКАЯ', 'ОРЁЛ', 'ПЕРМСКАЯ', 'ПРИВОЛЖСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'ЭВЕНКИЙСКИЙ', 'nan|ГУСЬ-ХРУСТАЛЬНЫЙ|МОСКОВСКИЙ\sП|РОССИЯ', 'ХАНТЫ-МАНСИЙСКИЙ-ЮГРА', 'ЧЕЛЯБИНСК$', 'ЧИТИНСКАЯ', 'ЧУКОТСКИЙ\sАO']
to_replace = ['', 'ЧЕЛЯБИНСКАЯ', 'ЛЕНИНГРАДСКАЯ', 'ЕВРЕЙСКАЯ АВТОНОМНАЯ', 'КАМЧАТСКИЙ', 'МОСКОВСКАЯ', 'САХА', 'СЕВЕРНАЯ ОСЕТИЯ - АЛАНИЯ', 'ХАНТЫ-МАНСИЙСКИЙ', 'ХАНТЫ-МАНСИЙСКИЙ', 'ЧУВАШСКАЯ', 'ЕВРЕЙСКАЯ АВТОНОМНАЯ', 'БРЯНСКАЯ', 'НИЖЕГОРОДСКАЯ', 'ОРЛОВСКАЯ', 'ПЕРМСКИЙ', 'МОСКОВСКАЯ', 'КРАСНОЯРСКИЙ', 'МОСКОВСКАЯ', 'ХАНТЫ-МАНСИЙСКИЙ', 'ЧЕЛЯБИНСКАЯ', 'ЗАБАЙКАЛЬСКИЙ', 'ЧУКОТСКИЙ']
data['living_region'].replace(from_replace, to_replace, regex=True, inplace=True)
categorical_columns = ['gender', 'marital_status', 'job_position', 'tariff_id', 'education', 'living_region']
numeric_columns = ['age', 'credit_sum', 'credit_month', 'monthly_income', 'credit_count', 'overdue_credit_count']
data = pd.get_dummies(data, columns=categorical_columns)
scaler = preprocessing.MinMaxScaler()
# scaler = preprocessing.Normalizer()
# scaler = preprocessing.StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
data = sklearn.utils.shuffle(data)

positive_data = data[data['open_account_flg'] == 1]
negative_data = data[data['open_account_flg'] == 0]
p_train, p_validate, p_test = np.split(positive_data.sample(frac=1), [int(.6*len(positive_data)), int(.8*len(positive_data))])
n_train, n_validate, n_test = np.split(negative_data.sample(frac=1), [int(.6*len(negative_data)), int(.8*len(negative_data))])
train = sklearn.utils.shuffle(pd.concat([p_train, n_train]))
validate = sklearn.utils.shuffle(pd.concat([p_validate, n_validate]))
test = sklearn.utils.shuffle(pd.concat([p_test, n_test]))

X_train = train.drop(['open_account_flg', 'client_id'], axis=1)
y_train = train['open_account_flg']
X_validate = validate.drop(['open_account_flg', 'client_id'], axis=1)
y_validate = validate['open_account_flg']
X_test = test.drop(['open_account_flg', 'client_id'], axis=1)
y_test = test['open_account_flg']



# Create classifiers
lr = LogisticRegression()
gnb = GaussianNB()
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier(n_estimators=100)
eclf = VotingClassifier(estimators=[('Logistic', lr),
                  ('Random Forest', rfc),
                  ('Naive Bayes', gnb),
                  ('Support Vector Classification', svc)], voting='hard')

#plot
plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf, name in [
                  (lr, 'Logistic'),
                  (rfc, 'Random Forest'),
                  (gnb, 'Naive Bayes'),
                  (svc, 'Support Vector Classification'),
                  (eclf, 'Voting Classifier')
                  ]:
    print('Fitting model ' + name)
    clf.fit(X_train, y_train)
    print('AUC = ' + str(roc_auc_score(y_validate, clf.predict(X_validate))))
# pdb.set_trace()

# print(1)


# (Pdb) data[data['credit_count'].isnull()].shape[0]
# 9230
# (Pdb) data[data['monthly_income'].isnull()].shape[0]
# 1
# (Pdb) data[data['living_region'].isnull()].shape[0]
# 192
# (Pdb) data[data['education'].isnull()].shape[0]
