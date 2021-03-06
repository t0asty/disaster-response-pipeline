import sys
import string
import pickle

import nltk
nltk.download('punkt')
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
        load data from database in database_filepath
        return 
            X - pd.DataFrame with messages
            Y - pd.DataFrame with categories
            category_names - names of categories in Y
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('Messages', engine)
    X = df['message']
    Y = df.drop(columns=['message', 'original', 'id', 'genre'])
    
    return X, Y, Y.columns


def tokenize(text):
    """ remove punctuation from text, make lowercase and return list of words"""
    table = str.maketrans(dict.fromkeys(string.punctuation))
    text.translate(table)
    return word_tokenize(text.lower())


def build_model():
    """ return model to train """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
    'clf__estimator__criterion': ['gini', 'entropy'],
    'clf__estimator__min_samples_split': [2, 5],
    'clf__estimator__min_samples_leaf': [1, 3],
    'vect__ngram_range': [(1,1), (1,2)]
}

    cv = GridSearchCV(pipeline, parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """evaluate model on X_test against Y_test"""
    Y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)
    
    for col in category_names:
        print(col)
        print(classification_report(Y_test[col], Y_pred[col]))


def save_model(model, model_filepath):
    """ save model to pickle file in model_filepath"""
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """ load data, train model and save model """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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