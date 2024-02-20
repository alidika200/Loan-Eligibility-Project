#Import necessary libraries

#For data processing
import pandas as pd
import numpy as np

#For visualisation
import matplotlib.pyplot as plt
import seaborn as sb

# For splitting the dataset in training and testing sets
from sklearn.model_selection import train_test_split

#For checking the model accuracy
from sklearn.metrics import accuracy_score
from sklearn import metrics

#For using decision tree models in scikit-learn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree

#For using K-nn Model in scikit-learn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.datasets import tests

#For data cleaning and summrization
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from scipy import stats

#For unpurposed Error filteration
import warnings
warnings.filterwarnings("ignore")
df_lep = pd.read_csv(r"C:\Users\user\Downloads\Ali\Uni\AI\Project\Datasets\Datasets\Loan Eligibility Prediction\loan_eligibility.csv")  ## saving the dataset into a  dataframe
df_lep.head()
df_lep.describe
df_lep.info()
print("DataSet Size: ",df_lep.shape)

# dropping the Loan_ID from the data set because it is not necessary in the data and may create a biased data

df_lep.drop('Loan_ID', axis=1, inplace=True)
print(df_lep.head())

# this and the head argument is to check the shape of the data after droping a column

print("DataSet Size: ", df_lep.shape)
df_lep.info()
df_lep['Loan_Status'].value_counts()
df_lep['Loan_Status'].value_counts().plot.bar()

# function to print the columns of missing data if any exists

check_missing_values = lambda df: print(df.isnull().sum()) if df.isnull().values.any() else print("No missing Data")
check_missing_values(df_lep)

# Understand our data to know how to deal with it
df_lep.describe()
categorical_columns = ['Gender','Married','Dependents','Self_Employed']
for column in categorical_columns:
    imputer = SimpleImputer(strategy='most_frequent')
    df_lep[column] = imputer.fit_transform(df_lep[[column]])
check_missing_values(df_lep)

#By Using the mean we can imputate safely the needed data and by using check_missing_values() function we check if there is any

Dependents_median_value = df_lep['Dependents'].mode()[0]
df_lep['Dependents'].fillna(Dependents_median_value, inplace=True)
Loan_Amount_Term_mode_value = df_lep['Loan_Amount_Term'].mode()[0]
df_lep['Loan_Amount_Term'].fillna(Loan_Amount_Term_mode_value, inplace=True)
LoanAmount_mean_value = df_lep['LoanAmount'].mean()
df_lep['LoanAmount'].fillna(LoanAmount_mean_value, inplace=True)
Credit_History_mean_value = df_lep['Credit_History'].mean()
df_lep['Credit_History'].fillna(Credit_History_mean_value, inplace=True)

check_missing_values(df_lep)

categorical_columns = df_lep.select_dtypes(include=['object']).columns
le = LabelEncoder()
for column in categorical_columns:
  df_lep[column] = le.fit_transform(df_lep[column])

duplicate_rows = df_lep[df_lep.duplicated()]
print(duplicate_rows)

df_lep.notna()
sb.distplot(df_lep['LoanAmount'])
plt.show()
df_lep['LoanAmount'].plot.box(figsize=(16, 5))
plt.show()
df_lep['LoanAmount'].describe()
df_lep['LoanAmount_log']=np.log(df_lep['LoanAmount'])
df_lep['LoanAmount_log'].hist(bins=20)
columns_to_check = ['LoanAmount']
z_scores = stats.zscore(df_lep[columns_to_check])
threshold = 3
outliers = (abs(z_scores) > threshold).any(axis=1)
df_no_outliers = df_lep[~outliers]
df_lep['LoanAmount'].describe()

#Visualising bivaritate relationship between each pair of attributes
df_lep.hist(bins=50, figsize=(20,15))
plt.show()
correlation_matrix = df_lep.corr()
# Create a heatmap
plt.figure(figsize=(10, 8))
plt.title('Correlation Matrix Heatmap')
cx = sb.heatmap(correlation_matrix, square=True, annot = True, cmap='coolwarm', fmt='.2f',)
cx.set_xticklabels(cx.get_xticklabels(), rotation=90)
cx.set_yticklabels(cx.get_yticklabels(), rotation=30)

plt.show()
print(df_lep.info())
X = df_lep.drop('Loan_Status',axis = 1)
Y = df_lep['Loan_Status']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,random_state = 0)
print("train data: ",x_train.shape)
print("test data: ",x_test.shape)

#Check the shape of x_train
print("train data: ",x_train.shape)

#Check the shape of x_test
print("test data: ",x_test.shape)

model = DecisionTreeClassifier(criterion = 'entropy',random_state = 123)
model.fit(x_train,y_train)
predections = model.predict(x_test)
print('Training set accuracy: {:.4f}'.format(model.score(x_test, y_test)*100))

print('Accuracy:  {:0.4f}'. format(model.score(x_test, y_test)*100))

y_pred = model.predict(x_test)
print('Accuracy:  {:0.4f}'. format(metrics.accuracy_score(y_test, y_pred)*100))

text_representation = tree.export_text(model)
print(text_representation)

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=['Yes', 'No'],filled=True, rounded=True)
plt.title("Decision Tree for Loan Status Prediction")
plt.show()

def conf_matrix(predictions,Y_test):
    c= pd.DataFrame(confusion_matrix(Y_test,predictions), index=['1','0'], columns=['1','0'])
    return c
conf_matrix(y_pred,y_test)

X = df_lep.drop('Loan_Status',axis = 1)
Y = df_lep['Loan_Status']

x_train_knn, x_test_knn, y_train_knn, y_test_knn = train_test_split(X, Y, test_size = 0.2,random_state = 0)
print("train data: ",x_train_knn.shape)
print("test data: ",x_test_knn.shape)

cols = x_train_knn.columns
scaler = StandardScaler()
x_train_knn = scaler.fit_transform(x_train_knn)
x_test_knn = scaler.transform(x_test_knn)

x_train_knn = pd.DataFrame(x_train_knn, columns=[cols])
x_test_knn = pd.DataFrame(x_test_knn, columns=[cols])

x_train_knn.head()

#instantiate the model
knn = KNeighborsClassifier(n_neighbors=3)

#fit the model to the training set
knn.fit(x_train_knn, y_train_knn)

y_pred_knn = knn.predict(x_test_knn)
print(y_pred_knn)

print('Model accuracy score: ', accuracy_score(y_test_knn, y_pred_knn))
print('Model accuracy score: ', knn.score(x_test_knn, y_test_knn))

# This method is to automatically find the best value of k for this model

# Specify a range of k values
k_values = list(range(1, 21))

# Create a dictionary with the parameter values
param_grid = {'n_neighbors': k_values}

# Create the k-NN classifier
knn = KNeighborsClassifier()

# Use GridSearchCV to perform the grid search
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Get the best k value
best_k = grid_search.best_params_['n_neighbors']

print(best_k)

# instantiate the model with k=11
knn_11 = KNeighborsClassifier(n_neighbors=11)
# fit the model to the training set
knn_11.fit(x_train_knn, y_train_knn)
# predict on the test-set
y_pred_11 = knn_11.predict(x_test_knn)
print('Model accuracy score with k=11: {:0.4f}'. format(accuracy_score(y_test_knn, y_pred_11)))

#Print the Confusion Matrix with k =11 and slice it into four pieces
cm11 = confusion_matrix(y_test_knn, y_pred_11)
print('Confusion matrix\n\n', cm11)
print('\nTrue Positives(TP) = ', cm11[0,0])
print('\nTrue Negatives(TN) = ', cm11[1,1])
print('\nFalse Positives(FP) = ', cm11[0,1])
print('\nFalse Negatives(FN) = ', cm11[1,0])
#or
pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

