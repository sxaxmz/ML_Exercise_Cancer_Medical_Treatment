from nltk.corpus import stopwords
import pandas as pd
import re



train_variants = pd.read_csv('training/training_variants')
train_text = pd.read_csv('training/training_text', sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)

def text_preprocess(df, total_text, ind, col):
    # Remove int values from text data as that might not be imp
    if type(total_text) is not int:
        string = ""
        # replacing all special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))
        # replacing multiple spaces with single space
        total_text = re.sub('\s+',' ', str(total_text))
        # bring whole text to same lower-case scale.
        total_text = total_text.lower()
        
        for word in total_text.split():
        # if the word is a not a stop word then retain that word from text
            if not word in stop_words:
                string += word + " "
        
        df[col][ind] = string

# Training Text (ID, text) #
print(train_text.columns)
# ID and Text column. 
# We can also observe column ID which is common in both the dataset.
print(train_text.head(5))

print(train_text.info())

print(train_text.describe())

print('The training text shape is ', train_text.shape)
# Training Variants (ID, Gene, Variations, Class) #

print(train_variants.columns)
# ID : row id used to link the mutation to the clinical evidence
# Gene : the gene where this genetic mutation is located
# Variation : the aminoacid change for this mutations
# Class : class value 1-9, this genetic mutation has been classified on
print(train_variants.head(5))

print(train_variants.info())

print(train_variants.describe())

print('The train variant shape is ', train_variants.shape)

# Prediction classes
print('Total Number of Prediction Classes: ', train_variants['Class'].unique())

# Text cleaning #
stop_words = set(stopwords.words('english'))

for index, row in train_text.iterrows():
    if type(row['TEXT']) is str:
        text_preprocess(train_text, row['TEXT'], index, 'TEXT')

result = pd.merge(train_variants, train_text,on='ID', how='left')
print(result.head())

# Handle missing
result[result.isnull().any(axis=1)]
result.loc[result['TEXT'].isnull(),'TEXT'] = result['Gene'] +' '+result['Variation']

# Remove spaces
result.Gene      = result.Gene.str.replace('\s+', '_')
result.Variation = result.Variation.str.replace('\s+', '_')

result.to_csv('Cleaned_Merged_df.csv')