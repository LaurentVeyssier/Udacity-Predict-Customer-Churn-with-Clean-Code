# library doc string
"""
Helper functions for Predicting customer Churn notebook
author: Laurent veyssier
Date: Nov. 9th 2022
"""

# import libraries
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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

    # Encode Churn dependent variable : 0 = Did not churned ; 1 = Churned
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 
                                            0 if val == "Existing Customer" 
                                            else 1)

    # Drop redudant Attrition_Flag variable (replaced by Churn response variable)
    df.drop('Attrition_Flag', axis=1, inplace=True)
    
    # Drop variable not relevant for the prediction model
    df.drop('CLIENTNUM', axis=1, inplace=True)

    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder

    input:
                    df: pandas dataframe

    output:
                    None
    '''

    # Analyze categorical features and plot distribution
    cat_columns = df.select_dtypes(include='object').columns.tolist()
    for cat_column in cat_columns:
        plt.figure(figsize=(7, 4))
        (df[cat_column]
            .value_counts('normalize')
            .plot(kind='bar',
                  rot=45,
                  title=f'{cat_column} - % Churn')

         )
        plt.savefig(os.path.join("./images/eda", f'{cat_column}.png'),
                    box_inches='tight')
        plt.show()

    # Analyze Numeric features
    plt.figure(figsize=(10, 5))
    (df['Customer_Age']
        .plot(kind='hist',
              title='Distribution of Customer Age')
     )
    plt.savefig(os.path.join("./images/eda", 'Customer_Age.png'),
                box_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 5))
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.show()

    # plot correlation matrix
    plt.figure(figsize=(15, 7))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join("./images/eda", 'correlation_matrix.png'),
                box_inches='tight')
    plt.show()

    plt.figure(figsize=(15, 7))
    (df[['Total_Trans_Amt', 'Total_Trans_Ct']]
        .plot(x='Total_Trans_Amt',
              y='Total_Trans_Ct',
              kind='scatter',
              title='Correlation analysis between 2 features')
     )
    plt.show()


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

    # Drop the obsolete categorical features of the category_lst 
    df.drop(category_lst, axis=1, inplace=True)

    return df


def perform_feature_engineering(df, response='Churn'):
    '''
    Converts remaining categorical using one-hot encoding adding the response 
    str prefix to new columns Then generate train and test datasets

    input:
                      df: pandas dataframe
                      response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
                      X_train: X training data
                      X_test: X testing data
                      y_train: y training data
                      y_test: y testing data
    '''

    # Collect categorical features to be encoded
    cat_columns = df.select_dtypes(include='object').columns.tolist()
    
    # Encode categorical features using mean of response variable on category
    df = encoder_helper(df, cat_columns, response='Churn')
    # Alternative to the encodign approach above - Not used here
    # convert categorical features to dummy variable
    #df = pd.get_dummies(df, columns=cat_columns, drop_first=True, prefix=response)

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


def confusion_matrix(model, model_name, X_test, y_test):
    '''
	Display confusion matrix of a model on test data

	input:
			model: trained model
            X_test: X testing data
			y_test: y testing data
	output:
			None
	'''
    class_names = ['Not Churned', 'Churned']
    plt.figure(figsize=(15, 5))
    ax = plt.gca()
    plot_confusion_matrix(model, 
                        X_test, 
                        y_test, 
                        display_labels=class_names, 
                        cmap=plt.cm.Blues, 
                        xticks_rotation='horizontal', 
                        colorbar=False, 
                        ax=ax)
    # Hide grid lines
    ax.grid(False)
    plt.title(f'{model_name} Confusion Matrix on test data')
    plt.savefig(
        os.path.join(
            "./images/results",
            f'{model_name}_Confusion_Matrix'),
        bbox_inches='tight')
    plt.show()



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

    # grid search for random forest parameters and instantiation
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

    # Display confusion matrix on test data
    confusion_matrix(cv_rfc.best_estimator_, 'Random Forest', X_test, y_test)
    confusion_matrix(lrc, 'Logistic Regression', X_test, y_test)



if __name__ == "__main__":
    pass
