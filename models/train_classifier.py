import sys
import sqlite3
import nltk

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
import pickle
from sklearn.metrics import confusion_matrix

def load_data(database_filepath):
    conn = sqlite3.connect(database_filepath)
    df =  pd.read_sql('SELECT * from table1', con = conn)
    
    X, y = np.asarray(df['message']), np.asarray(df.drop(['message', 'genre', 'original'], axis=1))
    categories = df.drop(['message', 'genre', 'original'], axis=1).columns
    #simple code to ensure that everything greater than zero is conerted to one
    #should refactor later
    new_y = []
    for row in y:
        new_row = []
        for x in row:
            if int(x) > 0:
                new_row.append(1)
            else:
                new_row.append(0)
        new_y.append(new_row)
    y = np.array(new_y)
    return X, y, categories



def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
    


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(XGBClassifier()))
    ])
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 1.0),
         'features__text_pipeline__vect__max_features': (None, 5000, 10000),
         'features__text_pipeline__tfidf__use_idf': (True, False),
        # 'clf__estimator': [XGBClassifier(), RandomForestClassifier()],
        # 'clf__max_depth': [2, 5]#,
        # 'features__transformer_weights': (
        # {'text_pipeline': 1, 'starting_verb': 0.5},
        # {'text_pipeline': 0.5, 'starting_verb': 1},
        # {'text_pipeline': 0.8, 'starting_verb': 1},
        # )
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def display_results(y_test, y_pred):
    conf_1 = confusion_matrix(y_true= pd.DataFrame(y_test)[0], y_pred= pd.DataFrame(y_pred)[0])
    for i in range(y_pred.shape[1]-1):
        conf_1 += confusion_matrix(y_true = pd.DataFrame(y_test)[i+1], y_pred= pd.DataFrame(y_pred)[i+1])
    labels = np.unique(y_pred)
    #confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()


    print("Confusion Matrix:\n", conf_1)
    print("Accuracy:", accuracy)

    print(
        classification_report(y_true=np.asarray(y_test).reshape(1, -1)[0], y_pred=np.asarray(y_pred).reshape(1, -1)[0],
                              labels=np.unique(y_pred)))
def display_report(y_test, y_pred1, cetegories):
    for real, pred, cat in zip(np.asarray(y_test).T , np.asarray(y_pred1).T, cetegories):
        print(cat)
        print(classification_report(real, pred))
        
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def evaluate_model(model, X_test, Y_test, categories):
    y_pred = model.predict(X_test)
    display_report(y_test, y_pred, categories)
    

def save_model(model, model_filepath):
    filename = 'trained_model.sav'
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, categories = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, categories)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
