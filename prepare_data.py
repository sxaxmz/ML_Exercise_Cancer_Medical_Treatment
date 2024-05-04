from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from collections import Counter
import matplotlib.pyplot as plt
from scipy.sparse import hstack
import seaborn as sns
import pandas as pd
import numpy as np
from functions_ import get_gv_fea_dict, get_gv_feature, apply_response_coding, extract_dictionary_paddle, get_text_responsecoding, get_intersec_text, get_impfeature_names, predict_and_plot_confusion_matrix, plot_confusion_matrix, report_log_loss # response coding with Laplace smoothing
import math


result = pd.read_csv('CLeaned_Merged.csv')
y_true = result['Class'].values

# Create train, test, and validation datasets #

# Splitting the data into train and test set 
X_train, test_df, y_train, y_test = train_test_split(result, y_true, stratify=y_true, test_size=0.2)
# split the train data now into train validation and cross validation
train_df, cv_df, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)

train_class_distribution = train_df['Class'].value_counts().sortlevel()
test_class_distribution = test_df['Class'].value_counts().sortlevel()
cv_class_distribution = cv_df['Class'].value_counts().sortlevel()

sorted_yi = np.argsort(-train_class_distribution.values)
for i in sorted_yi:
    print('Data points in train data: {} | Class distribution: {} ({}%)'.format(train_df.shape[0],train_class_distribution, np.round((train_class_distribution.values[i]/train_df.shape[0]*100), 3)))

my_colors = 'rgbkymc'
train_class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel(' Number of Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()

sorted_yi = np.argsort(-test_class_distribution.values)
for i in sorted_yi:
    print('Data points in test data: {} | Class distribution: {} ({}%)'.format(test_df.shape[0], test_class_distribution), np.round((test_class_distribution.values[i]/test_df.shape[0]*100), 3))

my_colors = 'rgbkymc'
test_class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel(' Number of Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()

sorted_yi = np.argsort(-cv_class_distribution.values)
for i in sorted_yi:
    print('Data points in cross validation data: {} | Class distribution: {} ({}%)'.format(cv_df.shape[0], cv_class_distribution), np.round((cv_class_distribution.values[i]/cv_df.shape[0]*100), 3))

my_colors = 'rgbkymc'
cv_class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel(' Number of Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()

# Build a random model to evaluate log loss when testing on a good model #

test_data_len = test_df.shape[0]
cv_data_len = cv_df.shape[0]

# create an output array that has exactly same size as the CV data
cv_predicted_y = np.zeros((cv_data_len,9))
for i in range(cv_data_len):
    rand_probs = np.random.rand(1,9)
    cv_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])
print('Log loss on Cross Validation Data using Random Model',log_loss(y_cv,cv_predicted_y, eps=1e-15))

# create an output array that has exactly same as the test data
test_predicted_y = np.zeros((test_data_len,9))
for i in range(test_data_len):
    rand_probs = np.random.rand(1,9)
    test_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])
print('Log loss on Test Data using Random Model',log_loss(y_test,test_predicted_y, eps=1e-15))

# index of the highest probability
predicted_y =np.argmax(test_predicted_y, axis=1)

# adjust prediction classes (start from 1 instead of 0)
predicted_y = predicted_y + 1

labels = [i for i in range(1,9)]

# Confusion Matrix
cm = confusion_matrix(y_test, predicted_y)

plt.figure(figsize=(20,7))
sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
plt.title('Percision Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()

# Percision Matrix
percision_mat =(cm/cm.sum(axis=0))

plt.figure(figsize=(20,7))
sns.heatmap(percision_mat, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
plt.title('Prediction Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()

# Recall Matrix
recall_mat =(((cm.T)/(cm.sum(axis=1))).T)

plt.figure(figsize=(20,7))
sns.heatmap(recall_mat, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
plt.title('Recall Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()

# Encode categorical victorizer #
gene_vectorizer = CountVectorizer()
variation_vectorizer = CountVectorizer()
text_vectorizer = CountVectorizer(min_df=3)


# Evaluate column relevancy for the prediction column # 

# Gene column
unique_genes = train_df['Gene'].value_counts()
print('Number of Unique Genes :', unique_genes.shape[0])
print(unique_genes.head(10))

sum_ = sum(unique_genes.values);
vos = unique_genes.values/sum_;
cumulative_sum = np.cumsum(vos)
plt.plot(cumulative_sum,label='Cumulative distribution of Genes')
plt.grid()
plt.legend()
plt.show()

train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(train_df['Gene'])
test_gene_feature_onehotCoding = gene_vectorizer.transform(test_df['Gene'])
cv_gene_feature_onehotCoding = gene_vectorizer.transform(cv_df['Gene'])

# column names after one-hot encoding for Gene column
print(gene_vectorizer.get_feature_names())

train_gene_feature_responseCoding, test_gene_feature_responseCoding, cv_gene_feature_responseCoding = apply_response_coding(train_df, test_df, cv_df, y_train, y_cv, 'Gene', train_gene_feature_onehotCoding, test_gene_feature_onehotCoding, cv_gene_feature_onehotCoding)

# Variation column

unique_variations = train_df['Variation'].value_counts()
print('Number of Unique Variations :', unique_variations.shape[0])
# the top 10 variations that occured most
print(unique_variations.head(10))

sum_ = sum(unique_variations.values)
vos = unique_variations.values/sum_
cumulative_sum = np.cumsum(vos)
print(cumulative_sum)
plt.plot(cumulative_sum,label='Cumulative distribution of Variations')
plt.grid()
plt.legend()
plt.show()

train_variation_feature_onehotCoding = variation_vectorizer.fit_transform(train_df['Variation'])
test_variation_feature_onehotCoding = variation_vectorizer.transform(test_df['Variation'])
cv_variation_feature_onehotCoding = variation_vectorizer.transform(cv_df['Variation'])

train_variation_feature_responseCoding, test_variation_feature_responseCoding, cv_variation_feature_responseCoding = apply_response_coding(train_df, test_df, cv_df, y_train, y_cv, 'Variation', train_variation_feature_onehotCoding, test_variation_feature_onehotCoding, cv_variation_feature_onehotCoding)

# Text column

train_text_feature_onehotCoding = text_vectorizer.fit_transform(train_df['TEXT'])
train_text_features= text_vectorizer.get_feature_names() # getting all the feature names (words)

# train_text_feature_onehotCoding.sum(axis=0).A1 will sum every row and returns (1*number of features) vector
train_text_fea_counts = train_text_feature_onehotCoding.sum(axis=0).A1

# zip(list(text_features),text_fea_counts) will zip a word with its number of times it occured
text_fea_dict = dict(zip(list(train_text_features),train_text_fea_counts))

print("Total number of unique words in train data :", len(train_text_features))

dict_list = []
# dict_list =[] contains 9 dictoinaries each corresponds to a class
for i in range(1,10):
    cls_text = train_df[train_df['Class']==i]
    # build a word dict based on the words in that class
    dict_list.append(extract_dictionary_paddle(cls_text))
    # append it to dict_list

# dict_list[i] is build on i'th  class text data
# total_dict is buid on whole training text data
total_dict = extract_dictionary_paddle(train_df)

confuse_array = []
for i in train_text_features:
    ratios = []
    max_val = -1
    for j in range(0,9):
        ratios.append((dict_list[j][i]+10 )/(total_dict[i]+90))
    confuse_array.append(ratios)
confuse_array = np.array(confuse_array)

train_text_feature_responseCoding  = get_text_responsecoding(train_df)
test_text_feature_responseCoding  = get_text_responsecoding(test_df)
cv_text_feature_responseCoding  = get_text_responsecoding(cv_df)

# we convert each row values such that they sum to 1  
train_text_feature_responseCoding = (train_text_feature_responseCoding.T/train_text_feature_responseCoding.sum(axis=1)).T
test_text_feature_responseCoding = (test_text_feature_responseCoding.T/test_text_feature_responseCoding.sum(axis=1)).T
cv_text_feature_responseCoding = (cv_text_feature_responseCoding.T/cv_text_feature_responseCoding.sum(axis=1)).T

train_text_feature_onehotCoding = normalize(train_text_feature_onehotCoding, axis=0)

test_text_feature_onehotCoding = text_vectorizer.transform(test_df['TEXT'])
test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)

cv_text_feature_onehotCoding = text_vectorizer.transform(cv_df['TEXT'])
cv_text_feature_onehotCoding = normalize(cv_text_feature_onehotCoding, axis=0)

#https://stackoverflow.com/a/2258273/4084039
sorted_text_fea_dict = dict(sorted(text_fea_dict.items(), key=lambda x: x[1] , reverse=True))
sorted_text_occur = np.array(list(sorted_text_fea_dict.values()))

# Number of words for a given frequency.
print(Counter(sorted_text_occur))

alpha =  0.001

cv_log_error_array=[]
for i in alpha:
    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
    clf.fit(train_text_feature_onehotCoding, y_train)
    
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_text_feature_onehotCoding, y_train)
    predict_y = sig_clf.predict_proba(cv_text_feature_onehotCoding)
    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
    print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))


fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array,c='g')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))
plt.grid()
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error measure")
plt.show()

best_alpha = np.argmin(cv_log_error_array)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
clf.fit(train_text_feature_onehotCoding, y_train)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_text_feature_onehotCoding, y_train)

predict_y = sig_clf.predict_proba(train_text_feature_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(cv_text_feature_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(test_text_feature_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

len1,len2 = get_intersec_text(test_df, train_text_features)
print(np.round((len2/len1)*100, 3), "% of word of test data appeared in train data")
len1,len2 = get_intersec_text(cv_df, train_text_features)
print(np.round((len2/len1)*100, 3), "% of word of Cross Validation appeared in train data")


# Combine all features #

# building train, test and cross validation data sets
# a = [[1, 2], 
#      [3, 4]]
# b = [[4, 5], 
#      [6, 7]]
# hstack(a, b) = [[1, 2, 4, 5],
#                [ 3, 4, 6, 7]]

train_gene_var_onehotCoding = hstack((train_gene_feature_onehotCoding,train_variation_feature_onehotCoding))
test_gene_var_onehotCoding = hstack((test_gene_feature_onehotCoding,test_variation_feature_onehotCoding))
cv_gene_var_onehotCoding = hstack((cv_gene_feature_onehotCoding,cv_variation_feature_onehotCoding))

train_x_onehotCoding = hstack((train_gene_var_onehotCoding, train_text_feature_onehotCoding)).tocsr()
train_y = np.array(list(train_df['Class']))

test_x_onehotCoding = hstack((test_gene_var_onehotCoding, test_text_feature_onehotCoding)).tocsr()
test_y = np.array(list(test_df['Class']))

cv_x_onehotCoding = hstack((cv_gene_var_onehotCoding, cv_text_feature_onehotCoding)).tocsr()
cv_y = np.array(list(cv_df['Class']))

print("One hot encoding features :")
print("(number of data points * number of features) in train data = ", train_x_onehotCoding.shape)
print("(number of data points * number of features) in test data = ", test_x_onehotCoding.shape)
print("(number of data points * number of features) in cross validation data =", cv_x_onehotCoding.shape)

train_gene_var_responseCoding = np.hstack((train_gene_feature_responseCoding,train_variation_feature_responseCoding))
test_gene_var_responseCoding = np.hstack((test_gene_feature_responseCoding,test_variation_feature_responseCoding))
cv_gene_var_responseCoding = np.hstack((cv_gene_feature_responseCoding,cv_variation_feature_responseCoding))

train_x_responseCoding = np.hstack((train_gene_var_responseCoding, train_text_feature_responseCoding))
test_x_responseCoding = np.hstack((test_gene_var_responseCoding, test_text_feature_responseCoding))
cv_x_responseCoding = np.hstack((cv_gene_var_responseCoding, cv_text_feature_responseCoding))

print(" Response encoding features :")
print("(number of data points * number of features) in train data = ", train_x_responseCoding.shape)
print("(number of data points * number of features) in test data = ", test_x_responseCoding.shape)
print("(number of data points * number of features) in cross validation data =", cv_x_responseCoding.shape)


f_1 = open("train_x_onehotCoding.txt", "w")
f_1.write(train_x_onehotCoding)
f_1.close()

f_2 = open("test_x_onehotCoding.txt", "w")
f_2.write(test_x_onehotCoding)
f_2.close()

f_3 = open("cv_x_onehotCoding.txt", "w")
f_3.write(cv_x_onehotCoding)
f_3.close()

f_4 = open("train_x_responseCoding.txt", "w")
f_4.write(train_x_responseCoding)
f_4.close()

f_5 = open("test_x_responseCoding.txt", "w")
f_5.write(test_x_responseCoding)
f_5.close()

f_6 = open("cv_x_responseCoding.txt", "w")
f_6.write(cv_x_responseCoding)
f_6.close()

f_7 = open('train_y.txt', 'w')
f_7.write(train_y)
f_7.close()

f_8 = open('test_y.txt', 'w')
f_8.write(test_y)
f_8.close()

f_9  = open('cv_y.txt', 'w')
f_9.write(cv_y)
f_9.close()

f_10  = open('y_cv.txt', 'w')
f_10.write(y_cv)
f_10.close()

f_11  = open('y_test.txt', 'w')
f_11.write(y_test)
f_11.close()

f_12  = open('y_train.txt', 'w')
f_12.write(y_train)
f_12.close()

f_13  = open('train_gene_feature_onehotCoding.txt', 'w')
f_13.write(train_gene_feature_onehotCoding)
f_13.close()

f_14  = open('train_text_features.txt', 'w')
f_14.write(train_text_features)
f_14.close()