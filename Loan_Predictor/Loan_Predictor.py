# Importing multiple library to read,analysed and visualized the dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

df = pd.read_csv('LoanApprovalPrediction.csv')

#print(df.head(5))
#print(df.shape)
#print(df.dtypes)
#print(df.info())

#finding the missing values
#print(df.isnull().sum())

#Describe function gives the basic numerical info about data for each numeric feature..
#print(df.describe())

#To analyze the distribution of values within the Loan_Status column of the DataFrame
#print(df['Loan_Status'].value_counts())

'''
# Visualize the distribution of loan status (target variable)
sns.countplot(x='Loan_Status', data=df)
plt.xlabel('Loan_Status')
plt.ylabel('Number of Applicants')
plt.title('Loan_Status Distribution in Loan Applicants')
plt.show()
'''


#To analyze the distribution of values within the Gender column of the DataFrame
#print(df['Married'].value_counts())

'''
# Visualize the distribution of Married
sns.countplot(x='Married', data=df)
plt.xlabel('Married')
plt.ylabel('Number of Applicants')
plt.title('Married Distribution in Loan Applicants')
plt.show()
'''


#To analyze the distribution of values within the Gender column of the DataFrame
#print(df['Gender'].value_counts())

'''
# Visualize the distribution of Gender
sns.countplot(x='Gender', data=df)
plt.xlabel('Gender')
plt.ylabel('Number of Applicants')
plt.title('Gender Distribution in Loan Applicants')
plt.show()
'''

#To analyze the distribution of values within the Education column of the DataFrame
#print(df['Education'].value_counts())

'''
# Visualize the distribution of Education
sns.countplot(x='Education', data=df)
plt.xlabel('Education')
plt.ylabel('Number of Applicants')
plt.title('Education Distribution in Loan Applicants')
plt.show()
'''
#To analyze the distribution of values within the Self_Employed column of the DataFrame
#print(df['Self_Employed'].value_counts())

'''
# Visualize the distribution of Self_Employed
sns.countplot(x='Self_Employed', data=df)
plt.xlabel('Self_Employed')
plt.ylabel('Number of Applicants')
plt.title('Self_Employed Distribution in Loan Applicants')
plt.show()
'''

#To analyze the distribution of values within the Property_Area column of the DataFrame
#print(df['Property_Area'].value_counts())

'''
# Visualize the distribution of Property_Area
sns.countplot(x='Property_Area', data=df)
plt.xlabel('Property_Area')
plt.ylabel('Number of Applicants')
plt.title('Property_Area in Loan Applicants')
plt.show()


#Create correlation matrix to assess relationships between features
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True)
plt.show()
'''

#display the columns of the dataset
#print(df.columns)

#This list contain all the "object" columns
categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area','Loan_Status','Loan_Amount_Term']
print(categorical_columns)

#This list contain all the numeric columns
numerical_columns = [ 'Dependents','ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
print(numerical_columns)

#To performs a transformation on the Loan_Status column in the DataFrame (df)
#Make sure the original categories ('Y' and 'N') in the data actually map to a meaningful interpretation of 1 (positive) and 0 (negative).
df['Loan_Status'] = df['Loan_Status'].map({'Y' : 1, 'N' : 0})

#x: This represents your entire feature set.
#y: This represents your entire target variable set.
x= df.drop(["Loan_ID","Loan_Status","Gender"],axis=1)
y= df[["Loan_Status"]]

#To iterates through each column in the DataFrame x and checks its data type and if the data type is "object",
#the column name is added to the cat_cols list.
#This list will then contain the names of all categorical columns in the dataset.
cat_cols = [col for col in x.columns if x.dtypes[col]=="object"]
print(cat_cols)

#To iterates through each column in the DataFrame x and checks its data type and if the data type is "numeric",
#the column name is added to the num_cols list.
#This list will then contain the names of all categorical columns in the dataset.
num_cols = [col for col in x.columns if x.dtypes[col] !="object"]
print(num_cols)

x_cat = x[cat_cols] #x_cat is assign to categorical values
x_num = x[num_cols] #x_num is also assign to numeric values

# filling in the null value with median in the numeric columns while most frequent is for the categorical columns.
from sklearn.impute import SimpleImputer

cat_impu = SimpleImputer(strategy="most_frequent")
num_impu = SimpleImputer(strategy="median")
x_cat = pd.DataFrame(cat_impu.fit_transform(x_cat), columns=cat_cols)
x_num = pd.DataFrame(num_impu.fit_transform(x_num), columns=num_cols)


x_cat[cat_cols]=x_cat[cat_cols].astype("category")

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

for col in cat_cols:
    x_cat[col]=encoder.fit_transform(x_cat[col])

#print("Printing:", x_cat.head())

x=pd.concat([x_num,x_cat],axis=1)

columns_x = x.columns
print(columns_x)

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(x)


x= pd.DataFrame(scale.transform(x),columns=columns_x)

#print(x.head())

# Split data into training and testing sets
#This allocated 20% (0.2) of the data to the testing set. and The remaining 80% also used for training.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#To check the dimensions (shape) of the data used for training and testing the model
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# Define the algorithms
from sklearn.svm import SVC
SVM= SVC(kernel = 'rbf', random_state = 0)
SVM.fit(x_train, y_train)

from sklearn.tree import DecisionTreeClassifier
DCT= DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DCT.fit(x_train, y_train)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state=0)
LR.fit(x_train,y_train)

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(x_train, y_train)

# Make predictions on the testing set
y_pred_SVM = SVM.predict(
    x_test)  # predicts testing data (x_test) using the trained SVM model (SVM). The results are stored in the variable y_pred_SVM.
y_pred_LR = LR.predict(
    x_test)  # predicts testing data (x_test) using the trained LR model (LR). The results are stored in the variable y_pred_LR.
y_pred_DCT = DCT.predict(
    x_test)  # predicts testing data (x_test) using the trained DCT model (DCT). The results are stored in the variable y_pred_DCT.
y_pred_KNN = KNN.predict(
    x_test)  # predicts testing data (x_test) using the trained KNN model (KNN). The results are stored in the variable y_pred_KNN.


# Evaluate the performance
from sklearn.metrics import accuracy_score
SVM_accuracy = accuracy_score(y_test, y_pred_SVM)
LR_accuracy = accuracy_score(y_test, y_pred_LR)
DCT_accuracy = accuracy_score(y_test, y_pred_DCT)
KNN_accuracy = accuracy_score(y_test, y_pred_KNN)

# Print the results
print("SVM Accuracy:", SVM_accuracy)
print("LR Accuracy:", LR_accuracy)
print("DCT Accuracy:", DCT_accuracy)
print("KNN Accuracy:", KNN_accuracy)

from sklearn.metrics import confusion_matrix
SVM_CM = confusion_matrix(y_test, y_pred_SVM)
LR_CM = confusion_matrix(y_test, y_pred_LR)
DCT_CM = confusion_matrix(y_test, y_pred_DCT)
KNN_CM = confusion_matrix(y_test, y_pred_KNN)

# Print the results
print("SNM confusion_matrix:", SVM_CM)
print("LR confusion_matrix:", LR_CM)
print("DCT confusion_matrix:", DCT_CM)
print("KNN confusion_matrix:", KNN_CM)

import tensorflow as tf

# Define the ANN model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=6, activation='relu', input_dim=x_train.shape[1]))
model.add(tf.keras.layers.Dense(units=3, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))  # Sigmoid for binary classification
# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32)

# Evaluate the model on test data
loss, accuracy = model.evaluate(x_test, y_test)
print('Test accuracy:', accuracy)





