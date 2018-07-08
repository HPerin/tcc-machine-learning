import pandas as pd
import numpy as np
from sklearn import ensemble, linear_model, svm, tree
from sklearn import model_selection, feature_selection
from sklearn.base import clone
from statistics import mean
from sklearn import metrics
import preprocess
import sys


def train(estimator, param_grid, folds):
    # load data from csv and preprocess it
    df = pd.read_csv('dataset.csv', sep='#')
    df = preprocess.preprocess_global(df)

    # separate data into input and output
    attributes: pd.DataFrame = df.drop(['diabet'], axis=1)
    outcome = df[['diabet']]

    variables_df = pd.DataFrame(columns=list(attributes))

    class D(type(estimator)):
        def fit(self, x, y):
            est = clone(estimator)
            est.set_params(**self.get_params())
            fs = feature_selection.RFECV(
                estimator=est,
                cv=folds,
                scoring=metrics.make_scorer(metrics.roc_auc_score, average='weighted'),
                n_jobs=-1,
            )
            fs.fit(x, y)
            self.best_variables = list(x.iloc[:, fs.get_support(indices=True)])
            x = x[self.best_variables]
            super().fit(x, y)

            i = len(variables_df)
            variables_df.loc[i] = 0
            variables_df.loc[i, self.best_variables] = 1

        def score(self, x, y):
            x = x[self.best_variables]
            return metrics.make_scorer(metrics.roc_auc_score, average='weighted')(self, x, y)

    roc_auc_list = []
    f1_list = []
    acc_list = []
    variable_use_count = {}
    parameter_use_count = {}
    for i_train, i_test in model_selection.KFold(n_splits=folds).split(attributes, outcome):
        attributes_train = preprocess.preprocess_local(attributes.iloc[i_train])
        attributes_test = preprocess.preprocess_local(attributes).iloc[i_test]

        outcome_train = preprocess.preprocess_local(outcome.iloc[i_train])
        outcome_test = preprocess.preprocess_local(outcome).iloc[i_test]

        model = model_selection.GridSearchCV(estimator=D(), param_grid=param_grid, cv=folds)
        model.fit(attributes_train, np.ravel(outcome_train))
        local_best_parameters = model.best_params_
        # print('best parameters: %s' % (local_best_parameters,))

        best_variables_attributes_test = attributes_test[model.best_estimator_.best_variables]

        best_variables = model.best_estimator_.best_variables
        best_parameters = local_best_parameters

        result_model = clone(estimator)
        result_model.set_params(**best_parameters)
        result_model.fit(attributes_train[best_variables], np.ravel(outcome_train))

        pred = result_model.predict(best_variables_attributes_test)
        roc_auc = metrics.roc_auc_score(outcome_test, pred)
        f1 = metrics.f1_score(outcome_test, pred)
        acc = metrics.accuracy_score(outcome_test, pred)

        for variable in best_variables:
            if variable not in variable_use_count:
                variable_use_count[variable] = 0
            variable_use_count[variable] = variable_use_count[variable] + 1

        for key, value in best_parameters.items():
            if key not in parameter_use_count:
                parameter_use_count[key] = {}
            if value not in parameter_use_count[key]:
                parameter_use_count[key][value] = 0
            parameter_use_count[key][value] = parameter_use_count[key][value] + 1

        roc_auc_list.append(roc_auc)
        f1_list.append(f1)
        acc_list.append(acc)

        print('--------------iteration %2d---------------' % (len(f1_list)))
        print('roc_auc %f' % (roc_auc,))
        print('f1 %f' % (f1,))
        print('acc %f' % (acc,))
        print('best_variables %s' % (best_variables,))
        print('best_parameters %s' % (best_parameters,))
        print('------------------------------------------')

    print('average auc roc: %s' % str(mean(roc_auc_list)))
    print('average f1: %s' % str(mean(f1_list)))
    print('average acc: %s' % str(mean(acc_list)))
    print('variable use count: %s' % sorted(variable_use_count.items(), key=lambda x: x[1], reverse=True))
    print('parameter use count:')
    for key, value in parameter_use_count.items():
        print('%s: %s' % (key, sorted(value.items(), key=lambda x: x[1], reverse=True)))
    variables_df.to_csv(str(type(estimator).__name__) + '.csv')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception('need parameter: log, svm, tree, forest')
    elif sys.argv[1] == 'log':
        print('----------------- Logistic Regression -----------------------')
        train(linear_model.LogisticRegression(), {
            'penalty': ['l1', 'l2'],
            'C': np.linspace(0.001, 1, 10).tolist()
        }, 10)
    elif sys.argv[1] == 'svm':
        print('----------------- SVM -----------------------')
        train(svm.LinearSVC(), {
            'C': np.linspace(0.001, 1, 10).tolist()
        }, 10)
    elif sys.argv[1] == 'tree':
        print('----------------- Decision Tree -----------------------')
        train(tree.DecisionTreeClassifier(), {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random']
        }, 10)
    elif sys.argv[1] == 'forest':
        print('----------------- Random Forest -----------------------')
        train(ensemble.RandomForestClassifier(), {
            'n_estimators': np.arange(3, 33, 3).tolist(),
            'criterion': ['gini', 'entropy']
        }, 10)
