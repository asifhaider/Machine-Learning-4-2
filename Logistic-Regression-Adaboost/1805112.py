"""
Adaboost implementation with Logistic Regression from scratch

Author: Md. Asif Haider (1805112)
Date: 7/12/2023
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
import time

np.random.seed(112)


def start_timer():
    start_time = time.time()
    return start_time


def calculate_time(start_time):
    end_time = time.time()
    return end_time - start_time


"""
Steps:
1. Imputation
2. Dropping
3. Modifying
4. One-hot encoding
5. Train-test split
6. Standardization
"""
def load_and_preprocess_dataset_one():
    # load the data into a dataframe
    df = pd.read_csv('Dataset-1/Telco-Customer-Churn.csv')

    # replace ' ' values with NaN
    df['TotalCharges'].replace(' ', np.nan, inplace=True)

    # imputation
    mean_imputer = SimpleImputer(strategy='mean', missing_values=np.nan)

    # Fit transform the imputer object on the columns with missing values
    df['TotalCharges'] = mean_imputer.fit_transform(df['TotalCharges'].values.reshape(-1,1))

    # drop the customerID column
    df.drop('customerID', axis=1, inplace=True)

    # modifying
    df['MultipleLines'].replace('No phone service', 'No', inplace=True)
    for i in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        df[i].replace('No internet service', 'No', inplace=True)
    
    # Modify the target column values
    df['Churn'].replace({'Yes':1, 'No':0}, inplace=True)

    # Separate the features and target
    y = df['Churn']
    X = df.drop('Churn', axis=1)

    # One-hot encoding using pandas
    X = pd.get_dummies(X, columns=['InternetService', 'Contract', 'PaymentMethod']) # these columns have more than 2 categories
    X = pd.get_dummies(X, drop_first=True).astype('float64') # these columns have only 2 categories

    # Split the data into train and test sets in 80-20 ratio

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345, stratify=y)

    # Scale the numerical features
    scaler = StandardScaler()   # z-score standardization: (x - mean) / std
    columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Fit and transform the training data, save the scaling parameters for future use in test data
    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns] = scaler.transform(X_test[columns])

    # print(type(X_train), type(X_test), type(y_train), type(y_test))
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test # dataframes and series


def load_and_preprocess_dataset_two():
    # column names for the data
    columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]

    # load the data into a dataframe
    df_train = pd.read_csv('Dataset-2/adult.data', names=columns)
    df_test = pd.read_csv('Dataset-2/adult.test', names=columns)
    df_test.drop(0, axis=0, inplace=True)

    # imputing the ' ?' values with the mode of the column
    for column in df_train.columns:
        df_train[column].replace(' ?', df_train[column].mode()[0], inplace=True)
        df_test[column].replace(' ?', df_test[column].mode()[0], inplace=True)
    
    # modify the output column values to 0 and 1
    df_train['income'].replace({' <=50K':0, ' >50K':1}, inplace=True)
    df_test['income'].replace({' <=50K.':0, ' >50K.':1}, inplace=True)

    # separate the features and target
    y_train = df_train['income']
    X_train = df_train.drop('income', axis=1)

    y_test = df_test['income']
    X_test = df_test.drop('income', axis=1)

    # find out the categorical columns
    categorical_columns = X_train.select_dtypes(include=['object'])
    categorical_columns.drop(['sex'], axis=1, inplace=True)
    # print(categorical_columns.columns)

    # one-hot encoding using pandas
    X_train = pd.get_dummies(X_train, drop_first=True, columns=['sex'])
    X_train = pd.get_dummies(X_train, columns=categorical_columns.columns.to_list()).astype('float64')

    X_test['age'] = X_test['age'].astype('float64')
    X_test = pd.get_dummies(X_test, drop_first=True, columns=['sex'])
    X_test = pd.get_dummies(X_test, columns=categorical_columns.columns.to_list()).astype('float64')
    
    # Not present in the test data, so drop them
    X_train.drop(['native-country_ Holand-Netherlands'], axis=1, inplace=True)

    # print(X_train.shape)  # (32561, 103)
    # print(X_test.shape)  # (16281, 103)

    # Scale the numerical features
    scaler = StandardScaler()
    numerical_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    # Fit and transform the training data, save the scaling parameters for future use in test data
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

    # print(type(X_train), type(X_test), type(y_train), type(y_test))
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test # dataframes


def load_and_preprocess_dataset_three():
   # load the data into a dataframe
    df = pd.read_csv('Dataset-3/creditcard.csv')

    y = df['Class'].astype('float64')
    X = df.drop('Class', axis=1)

    # Split into train and test sets in 80-20 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=112, stratify=y)

    # Scale the numerical features
    scaler = StandardScaler()
    columns = X_train.columns

    # Fit and transform the training data, save the scaling parameters for future use in test data
    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns] = scaler.transform(X_test[columns])

    # print(type(X_train), type(X_test), type(y_train), type(y_test))
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test # dataframes


"""
Information gain to select top k features
"""
def feature_selection_using_information_gain(X_train, y_train, top_features):
    # determine the mutual information on the training data
    mutual_info = mutual_info_classif(X_train, y_train)

    # sort the mutual information values in descending order
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X_train.columns
    mutual_info.sort_values(ascending=False)

    # # plot the mutual information values
    # mutual_info.sort_values(ascending=False).plot.bar(figsize=(20,8))

    # select the top k features manually
    selected_top_columns = mutual_info.sort_values(ascending=False).head(top_features).index
    # print(selected_top_columns)

    # return the selected columns
    return selected_top_columns



def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def calculate_hypothesis(X, theta):
    return sigmoid(np.dot(X, theta))    #  y = theta.T * X


def calculate_gradient(X, y, theta):
    h = calculate_hypothesis(X, theta)
    return np.dot(X.T, (h - y)) / y.shape[0]    # 1/m * (h-y) * X


def mean_squared_error(y, h):
    return np.mean((y - h) ** 2) 


def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std    # z-normalization: (x - mean) / std


def train(X, y, learning_rate, epochs, terminating_threshold):
    # convert the dataframes to numpy arrays and normalize the data
    # X = X.to_numpy()
    # y = y.to_numpy()
    X = normalize(X)

    # add a column of 1s as the first column
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    # print(f'X shape: {X.shape}')
    # print(type(X))

    # initialize the parameters
    theta = np.zeros(X.shape[1])
    # print(f'theta shape: {theta.shape}')
    # print(type(theta))

    # training loop
    for epoch in range(epochs):
        gradient = calculate_gradient(X, y, theta)
        theta -= learning_rate * gradient

        if epoch % 100 == 0:
            h = calculate_hypothesis(X, theta)
            error = mean_squared_error(y, h)
            print(f'Epoch: {epoch}, MSE: {error}')

        # terminate if the mean squared error is less than the terminating threshold
        if error < terminating_threshold:
            break
    
    return theta


def predict(X, theta):
    # normalize the data
    X = normalize(X)

    # add a column of 1s as the first column
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    # calculate the hypothesis
    h = calculate_hypothesis(X, theta)

    # calculate the predicted values
    y_pred = np.where(h >= 0.5, 1, 0)

    return y_pred


def weighted_majority_predict(X, hypotheses, hypothesis_weights):
    # normalize the data
    X = normalize(X)

    # add a column of 1s as the first column
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    # calculate the hypothesis
    y_preds = []
    for theta in hypotheses:
        h = calculate_hypothesis(X, theta)
        y_preds.append(np.where(h >= 0.5, 1, -1))

    y_preds = np.array(y_preds)

    # calculate the predicted values
    # What's happening here?
    weighted_hypotheses = np.dot(y_preds.T, hypothesis_weights)
    y_pred = np.where(weighted_hypotheses >= 0, 1, 0)

    return np.array(y_pred).reshape(-1,1)


def performance_metrics(y_gold, y_pred):

    # print(type(y_gold))
    # print(type(y_pred))

    # calculate the accuracy, (TP + TN) / (TP + TN + FP + FN)
    accuracy = np.mean(y_gold == y_pred)

    # calculate the sensitivity, TP / (TP + FN)
    sensitivity = np.sum((y_gold == 1) & (y_pred == 1)) / np.sum(y_gold == 1)

    # calculate the specificity, TN / (TN + FP)
    specificity = np.sum((y_gold == 0) & (y_pred == 0)) / np.sum(y_gold == 0)

    # calculate the precision, TP / (TP + FP)
    precision = np.sum((y_gold == 1) & (y_pred == 1)) / np.sum(y_pred == 1)

    # calculate the false discovery rate, FP / (TP + FP)
    false_discovery_rate = np.sum((y_gold == 0) & (y_pred == 1)) / np.sum(y_pred == 1)

    # calculate the F1 score, 2 * (precision * sensitivity) / (precision + sensitivity)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    return accuracy, sensitivity, specificity, precision, false_discovery_rate, f1_score


def print_performance_metrics(y_gold, y_pred, dataset, algo, learning_rate, epochs, terminating_threshold, top_k, boosting_count):
    accuracy, sensitivity, specificity, precision, false_discovery_rate, f1_score = performance_metrics(y_gold, y_pred)

    output_string = f'Accuracy: {accuracy}\nSensitivity: {sensitivity}\nSpecificity: {specificity}\nPrecision: {precision}\nFalse discovery rate: {false_discovery_rate}\nF1 score: {f1_score}'
    print(output_string)

    # write the output to a file
    file_name = f'output/Data-{dataset}-{algo}-alpha-{learning_rate}-epoch-{epochs}-thresh-{terminating_threshold}-top-{top_k}-boost-{boosting_count}.txt'
    # write or append
    with open(file_name, 'a') as f:
        f.write(output_string+'\n\n')
        f.close()


def evaluate_logistic_regression_model(dataset, algo, learning_rate = 0.01, epochs = 5000, terminating_threshold=0.2, top_k=105):
    # load and preprocess the dataset
    if dataset == "1":
        X_train, X_test, y_train, y_test = load_and_preprocess_dataset_one() # 26
    elif dataset == "2":
        X_train, X_test, y_train, y_test = load_and_preprocess_dataset_two() # 103
    elif dataset == "3":
        X_train, X_test, y_train, y_test = load_and_preprocess_dataset_three() # 30

    if top_k > X_train.shape[1]:
        top_k = X_train.shape[1]

    # select the top features using information gain
    top_features = feature_selection_using_information_gain(X_train, y_train, top_k)

    # select the top features from the top_features column names
    X_train = X_train[top_features]

    # if pandas, then convert to numpy
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()

    # print(type(X_train))
    # print(type(y_train))

    # exit()

    # train the model
    theta = train(X_train, y_train, learning_rate, epochs, terminating_threshold)

    # performance on training data
    y_pred = predict(X_train, theta)

    print(f'\033[94mPerformance on training data:\033[0m')
    print(f'Learning rate: {learning_rate}, Epochs: {epochs}, Terminating threshold: {terminating_threshold}, Top k Features: {top_k}')
    print_performance_metrics(y_train, y_pred, dataset, algo, learning_rate, epochs, terminating_threshold, top_k, "NA")

    # select the top features from the top_features column names
    X_test = X_test[top_features]

    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.to_numpy()
    if isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()

    # performance on test data
    y_pred = predict(X_test, theta)

    print(f'\033[94mPerformance on test data:\033[0m')
    print(f'Learning rate: {learning_rate}, Epochs: {epochs}, Terminating threshold: {terminating_threshold}, Top k Features: {top_k}')
    print_performance_metrics(y_test, y_pred, dataset, algo, learning_rate, epochs, terminating_threshold, top_k, "NA")


def adaboost(X, y, learning_rate, epochs, terminating_threshold, boosting_count):
    # initialize the weights, all the weights are equal
    weights = np.ones(X.shape[0]) / X.shape[0]

    # initialize the list of hypotheses
    hypotheses = []

    # initialize the list of hypothesis weights
    hypothesis_weights = []

    # loop for the number of boosting iterations
    for k in range(boosting_count):
        # concatenate the X and y to create the example
        example = np.concatenate((X, y), axis=1)
        # print(f'Example shape: {example.shape}')
        
        # resample the example with the weights
        resampled_example = example[np.random.choice(X.shape[0], size=X.shape[0], replace=True, p=weights)]
        # print(f'Resampled example shape: {resampled_example.shape}')

        # separate the resampled example into X and y
        resampled_X = resampled_example[:, :-1]
        resampled_y = resampled_example[:, -1]

        # print(f'Resampled X shape: {resampled_X.shape}')
        # print(f'Resampled y shape: {resampled_y.shape}')
        # exit()

        # train the model
        theta = train(resampled_X, resampled_y, learning_rate, epochs, terminating_threshold)

        # predict the hypothesis
        y_pred = predict(X, theta)

        # report the accuracy of the hypothesis
        # accuracy = np.mean(y == y_pred)
        # print(f'Accuracy of hypothesis {k+1}: {accuracy}')

        # initialize the error
        error = 0

        # calculate the error over original data
        for j in range(y.shape[0]):
            if y[j] != y_pred[j]:
                error += weights[j]

        if error > 0.5:
            continue
        else:
            hypotheses.append(theta)

        # calculate the hypothesis weight
        for j in range(y.shape[0]):
            if y[j] == y_pred[j]:
                weights[j] *= error / (1 - error)

        # normalize the weights
        weights /= np.sum(weights)

        # calculate and append the hypothesis weight
        hypothesis_weights.append(np.log2((1 - error) / error))

    return hypotheses, hypothesis_weights


def evaluate_adaboost_model(dataset, algo, learning_rate=0.005, epochs=5000, terminating_threshold=0, top_k=105, boosting_count=10):
    # load and preprocess the dataset
    if dataset == "1":
        X_train, X_test, y_train, y_test = load_and_preprocess_dataset_one()
    elif dataset == "2":
        X_train, X_test, y_train, y_test = load_and_preprocess_dataset_two()
    elif dataset == "3":
        X_train, X_test, y_train, y_test = load_and_preprocess_dataset_three()

    if top_k > X_train.shape[1]:
        top_k = X_train.shape[1]

    # select the top features using information gain, total 26 features after preprocessing
    top_features = feature_selection_using_information_gain(X_train, y_train, top_k)

    # select the top features from the top_features column names
    X_train = X_train[top_features]
    y_train = y_train.to_numpy().reshape(-1,1)

    # use the adaboost algorithm to train and predict
    hypotheses, hypotheses_weights = adaboost(X_train, y_train, learning_rate, epochs, terminating_threshold, boosting_count)

    print(f'\033[94mPerformance on training data:\033[0m')
    print(f'Learning rate: {learning_rate}, Epochs: {epochs}, Terminating threshold: {terminating_threshold}, Top k Features: {top_k}, Boosting count: {boosting_count}')
    print_performance_metrics(y_train, weighted_majority_predict(X_train, hypotheses, hypotheses_weights), dataset, algo, learning_rate, epochs, terminating_threshold, top_k, boosting_count)

    # select the top features from the top_features column names
    X_test = X_test[top_features]
    y_test = y_test.to_numpy().reshape(-1,1)

    print(f'\033[94mPerformance on test data:\033[0m')
    print(f'Learning rate: {learning_rate}, Epochs: {epochs}, Terminating threshold: {terminating_threshold}, Top k Features: {top_k}, Boosting count: {boosting_count}')
    print_performance_metrics(y_test, weighted_majority_predict(X_test, hypotheses, hypotheses_weights), dataset, algo, learning_rate, epochs, terminating_threshold, top_k, boosting_count)
    pass


def main():

    # Reproducing the results
    start = start_timer()
    evaluate_logistic_regression_model("1", "Logistic", 0.01, 10000, 0.1, 105)

    for i in range(5, 25, 5):
        evaluate_adaboost_model("1", "Adaboost", 0.01, 5000, 0, 15, i)
    total = calculate_time(start)
    print(f'\033[92mTime taken in minutes: {total/60}\033[0m')

    start = start_timer()
    evaluate_logistic_regression_model("2", "Logistic", 0.01, 10000, 0, 105)

    for i in range(5, 25, 5):
        evaluate_adaboost_model("2", "Adaboost", 0.01, 1000, 0, 75, i)
    total = calculate_time(start)
    print(f'\033[92mTime taken in minutes: {total/60}\033[0m')

    start = start_timer()
    evaluate_logistic_regression_model("3", "Logistic", 0.01, 10000, 0, 105)

    for i in range(5, 25, 5):
        evaluate_adaboost_model("3", "Adaboost", 0.01, 1000, 0, 15, i) 
    total = calculate_time(start)
    print(f'\033[92mTime taken in minutes: {total/60}\033[0m')
    
    return


if __name__ == '__main__':
    main()


    # # prompt user input to choose from 3 options for 3 datasets
    # print('Choose a dataset:')
    # print('1. Telco Customer Churn Prediction')
    # print('2. Adult Income Prediction')
    # print('3. Credit Card Fraud Detection')

    # dataset = int(input('Enter your choice: '))
    
    # # prompt user input to choose from 2 options for 2 algorithms
    # print('Choose an algorithm:')
    # print('1. Logistic Regression')
    # print('2. Adaboost (Logistic Regression as Weak Learner)')

    # algorithm = int(input('Enter your choice: '))

    # if dataset == 1:
    #     if algorithm == 1:
    #         # prompt user input for top k features or all features
    #         print('Choose the number of features to select:')
    #         print('Top k features: Input a number')
    #         print('All features: Press A')
    #         option = input('Enter your choice: ')
    #         try:
    #             if option == 'A':
    #                 evaluate_logistic_regression_model()
    #             elif option.isdigit():
    #                 evaluate_logistic_regression_model(top_k=int(option))  
    #             else:
    #                 print('Invalid input')
    #         except:
    #             print('Invalid input')
    #     elif algorithm == 2:
    #         # prompt user input for boosting round count
    #         print('Choose the number of boosting rounds:')
    #         print('Example: 5, 10, 15, 20')
    #         boosting_count = int(input('Enter your choice: '))

    #         print('Choose the number of features to select:')
    #         print('Top k features: Input a number')
    #         print('All features: Press A')
    #         option = input('Enter your choice: ')
    #         try:
    #             if option == 'A':
    #                 evaluate_adaboost_model(boosting_count=boosting_count)
    #             elif option.isdigit():
    #                 evaluate_adaboost_model(top_k=int(option), boosting_count=boosting_count)
    #             else:
    #                 print('Invalid input')
    #         except:
    #             print('Invalid input')
