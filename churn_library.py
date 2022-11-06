# library doc string
"""Helper functions """

# import libraries
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
                    pth: a path to the csv
    output:
                    df: pandas dataframe
    '''
    df = pd.read_csv(pth, index_col=0)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
                    df: pandas dataframe

    output:
                    None
    '''
    pass


def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
                    df: pandas dataframe
                    category_lst: list of columns that contain categorical features
                    response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
                    df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        category_groups = df.groupby(category).mean()[response]
        new_feature = category + '_' + response
        df[new_feature] = df[category].apply(lambda x: category_groups.loc[x])

    return df


def perform_feature_engineering(df, response='Churn'):
    '''
    input:
                      df: pandas dataframe
                      response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
                      X_train: X training data
                      X_test: X testing data
                      y_train: y training data
                      y_test: y testing data
    '''
    y = df[response]
    X = df.drop(response, axis=1)
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def plot_classification_report(model_name,
                               y_train,
                               y_test,
                               y_train_preds,
                               y_test_preds):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
                    model_name: (str) name of the model, ie 'Random Forest'
                    y_train: training response values
                    y_test:  test response values
                    y_train_preds: training predictions from model_name
                    y_test_preds: test predictions from model_name

    output:
                     None
    '''

    plt.rc('figure', figsize=(5, 5))

    # Plot Classification report on Train dataset
    plt.text(0.01, 1.25,
             str(f'{model_name} Train'),
             {'fontsize': 10},
             fontproperties='monospace'
             )
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds)),
             {'fontsize': 10},
             fontproperties='monospace'
             )

    # Plot Classification report on Test dataset
    plt.text(0.01, 0.6,
             str(f'{model_name} Test'),
             {'fontsize': 10},
             fontproperties='monospace'
             )
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds)),
             {'fontsize': 10},
             fontproperties='monospace'
             )

    plt.axis('off')

    # Save figure to ./images folder
    fig_name = f'Classification_report_{model_name}.png'
    plt.savefig(
        os.path.join(
            "./images/results",
            fig_name),
        bbox_inches='tight')

    # Display figure
    plt.show()


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder using plot_classification_report helper function
    input:
                    y_train: training response values
                    y_test:  test response values
                    y_train_preds_lr: training predictions from logistic regression
                    y_train_preds_rf: training predictions from random forest
                    y_test_preds_lr: test predictions from logistic regression
                    y_test_preds_rf: test predictions from random forest

    output:
                     None
    '''

    plot_classification_report('Logistic Regression',
                               y_train,
                               y_test,
                               y_train_preds_lr,
                               y_test_preds_lr)

    plot_classification_report('Random Forest',
                               y_train,
                               y_test,
                               y_train_preds_rf,
                               y_test_preds_rf)


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
                    model: model object containing feature_importances_
                    X_data: pandas dataframe of X values
                    output_pth: path to store the figure

    output:
                     None
    '''

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save figure to output_pth
    fig_name = 'feature_importance.png'
    plt.savefig(os.path.join(output_pth, fig_name), bbox_inches='tight')

    # display feature importance figure
    plt.show()


"""def classification_score(dataset, y, preds):
	'''
	calculate classification score
	input:
			dataset: (str) train or test dataset type
			y: ground truth data
			preds: predictions
	output:
			None
	'''
	print(f'{dataset} results')
	print(classification_report(y, preds))"""


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
                      X_train: X training data
                      X_test: X testing data
                      y_train: y training data
                      y_test: y testing data
    output:
                      None
    '''
    # Initialize Random Forest model
    rfc = RandomForestClassifier(random_state=42)

    # Initialize Logistic Regression model
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    # grid search for random forest
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # Train Ramdom Forest using GridSearch
    cv_rfc.fit(X_train, y_train)

    # Train Logistic Regression
    lrc.fit(X_train, y_train)

    # get predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # calculate classification scores
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    '''print('random forest results')
	classification_score('train', y_train, y_train_preds_rf)
	classification_score('test', y_test, y_test_preds_rf)
	print('logistic regression results')
	classification_score('train', y_train, y_train_preds_lr)
	classification_score('test', y_test, y_test_preds_lr)'''

    # plot ROC-curves
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8
    )
    plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)

    # save ROC-curves to images directory
    plt.savefig(
        os.path.join(
            "./images/results",
            'ROC_curves.png'),
        bbox_inches='tight')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    pass
