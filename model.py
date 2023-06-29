
def lines():
    print()
    print("-----------------------------------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------------------------------")
    print()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from pandas.core.frame import DataFrame
import seaborn as sns

# DISPLAYING THE DATASET
result_all_grade = pd.read_csv('Data/MOODLE DATASET_MAJOR PROJECT - Sheet1.csv')
print("DATASET - RESULT OBTAINED FROM LMS")
print(result_all_grade.head())
lines()


# VIEWING THE COLUMNS ( PARAMETERS ) OF THE DATASET
print("VIEWING THE COLUMNS OF DATASET")
print(result_all_grade.columns)
lines()


# VIEWING THE QUIZ MARKS OF THE STUDENTS USING THE INDIVIDUAL DATASET OBTAINED FROM MOODLE
result_quiz_ind = pd.read_csv('Data/Quiz Time - Sheet1.csv')
print("DATASET - QUIZ MARKS OBTAINED ( INDIVIDUAL RECORD )")
print(result_quiz_ind.head())
lines()


# CONVERTING THE TIME FORMAT AND USING ONLY THE MINUTES AND ADDING THIS TIME PARAMETER TO THE MAIN FUNCTION
dataF_allGrade = DataFrame(result_all_grade)
dataF_quizIndividual = DataFrame(result_quiz_ind)
result_quiz_ind['Time taken'] = dataF_quizIndividual['Time taken'].str.replace('mins....sec', '')
result_quiz_ind['Time taken'] = result_quiz_ind['Time taken'].astype(dtype="int32")
result_all_grade['Quiz Time'] = result_quiz_ind['Time taken']

# VIEWING THE CGPA DATASET AND ADDING THE CGPA COLUMN TO THE DATASET
result_CGPA = pd.read_csv("Data/CGPA of Students - Sheet1.csv")
print("DATASET - CGPA OF THE STUDENTS")
print(result_CGPA.head())
lines()

result_all_grade['CGPA'] = result_CGPA['CGPA']


# VIEWING THE ENTIRE DATASET
print(result_all_grade)


# VISUALIZING THE DATA USING SEABORN
disp_data = result_all_grade.drop(["First name", "ID number", "Institution", "Department", "Email address", "PRN", "Last downloaded from this course"], axis=1)
sns.pairplot(disp_data, hue="Outcome")
plt.show()


# VISUALIZING THE COMPARISION OF FIRST TEST( ASSSIGNMENT ) AND LAST TEST( IN-CLASS ASSESSMENT )
x = result_all_grade.loc[0:50, "Roll No"]
y1 = result_all_grade.loc[0:50, 'Inclass Assessment(20)']
y2 = result_all_grade.loc[0:50, 'Assignment(20)']
fig, ax = plt.subplots(figsize=(8, 10), layout='constrained')
ax.plot(x, y1, 'r', label=' MARKS - ASSIGNMENT')
ax.plot(x, y2, 'y', label='MARKS - INCLASS ASSESSMENT')
plt.title("KNOWLEDGE RETENTION ( 0 - 50 STUDENTS)")
plt.xlabel("STUDENTS ( 0 - 50 )")
plt.ylabel("MARKS")
plt.legend()
plt.legend()
plt.show()
lines()

# ANALYZING THE QUIZ PATTERN
x = result_all_grade.loc[0:50, "Roll No"]
y1 = result_all_grade.loc[0:50, 'Quiz Unit 1(20)']
y2 = result_all_grade.loc[0:50, 'Quiz Time']
fig, ax = plt.subplots(figsize=(8, 10), layout='constrained')
ax.plot(x, y1, 'r', label=' QUIZ MARKS')
ax.plot(x, y2, 'y', label='TIME TAKEN')
plt.title("QUIZ ANALYSIS ( 0 - 50 STUDENTS)")
plt.xlabel("STUDENTS ( 0 - 50 )")
plt.ylabel("QUIZ")
plt.legend()
plt.show()
lines()

'''

x = result_all_grade.loc[0:50, "Roll No"]
y1 = result_all_grade.loc[0:50, 'Assignment(20)']
y2 = result_all_grade.loc[0:50, 'Forum(5)']
y3 = result_all_grade.loc[0:50, 'Quiz Unit 1(20)']
y4 = result_all_grade.loc[0:50, 'Inclass Assessment(20)']


#PLOTTING THE DOTTED GRAPH FOR STUDENTS
fig, ax = plt.subplots(figsize=(8, 10), layout='constrained')
ax.plot(x, y1, 'ro', label='Assignment(20)')
ax.plot(x, y2, 'yo', label='Forum(5)')
ax.plot(x, y3, 'go', label='Quiz Unit 1(20)')
ax.plot(x, y4, 'bo', label='Inclass Assessment(20)')
plt.xlabel("STUDENTS ( 0 - 50 )")
plt.ylabel("QUIZ")
plt.legend()
plt.show()

'''


# GETTING THE MEAN, MEDIAN AND MODE OF THE QUIZ FOR FURTHER ANALYSIS
print("QUIZ TIME TAKEN AVERAGE : ")
meanTime = result_all_grade['Quiz Time'].mean()
print("Mean : ", meanTime)
lines()

# GETTING THE INDEX OF ALL COLUMNS
print(result_all_grade.columns)
lines()

# FILTERING AND REDUCING THE MARKS IF AVERAGE TIME OF A STUDENT IS LESS THAN THE
# AVERAGE TIME OF EVERY STUDENT TO COVER PLAGARISM IN QUIZ SUBMISSION
for i in result_all_grade['Quiz Time']:
    if i < meanTime:
        result_all_grade['Quiz Unit 1(20)'] -= 5

# CREATING TRAIN AND TEST DATA FOR MODEL TRAINING
X = result_all_grade.iloc[:, 8:13]
X['Quiz Time'] = result_all_grade['Quiz Time']
X['CGPA'] = result_all_grade['CGPA']

y = result_all_grade['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
print("Training Size")
print("X : ", X_train.shape)
print("Y : ", y_train.shape)
print("Testing Size")
print("X : ", X_test.shape)
print("Y : ", y_test.shape)
lines()




print("----->   K-NEAREST NEIGHBORS")
# USING K-NEAREST NEIGHBORS
knn = KNeighborsClassifier()
clf = knn.fit(X_train, y_train)
predict = knn.predict(X_test)

# GETTING THE INFO. OF KNN RESULTS
print('****************** KNN Classification ******************')
print('Classes: ', clf.classes_)
print('Effective Metric: ', clf.effective_metric_)
print('Effective Metric Params: ', clf.effective_metric_params_)
print('No. of Samples Fit: ', clf.n_samples_fit_)
lines()

testScore = knn.score(X_test, y_test)
print('Accuracy Score: ', testScore)
# Look at classification report to evaluate the model
print(classification_report(y_test, predict))
lines()


cMatrix = confusion_matrix(y_test, predict)
print('')
print("Confusion Matrix : ", cMatrix)

knn_Score = accuracy_score(y_test, predict)
print("Accuracy Score : ", knn_Score)
lines()




print("----->   NAIVE BAYES")
# USING NAIVE BAYES NEIGHBORS
nb = GaussianNB()
clf = nb.fit(X_train, y_train)
predict = nb.predict(X_test)
print(X_test.columns)

# GETTING THE INFO. OF NAIVE BAYES RESULTS
print('****************** Naive bayes Classification ******************')
print('Classes: ', clf.classes_)
lines()

testScore = nb.score(X_test, y_test)
print('Accuracy Score: ', testScore)
# Look at classification report to evaluate the model
print(classification_report(y_test, predict))
lines()

cMatrix = confusion_matrix(y_test, predict)
print('')
print("Confusion Matrix : ", cMatrix)

nb_Score = accuracy_score(y_test, predict)
print("Accuracy Score : ", knn_Score)
lines()
