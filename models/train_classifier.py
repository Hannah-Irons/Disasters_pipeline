import sys
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine;
from nltk import word_tokenize, pos_tag;
from nltk.corpus import stopwords;
from nltk.stem.wordnet import WordNetLemmatizer;
from sklearn.model_selection import GridSearchCV;
from sklearn.pipeline import Pipeline;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.multioutput import MultiOutputClassifier;
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix;
from sklearn.model_selection import train_test_split;
import pickle
import nltk

nltk.download(['wordnet', 'punkt', 'stopwords'])

"""
    Function Objective: Bring in merged data from database
    INPUT: path to database
    OUTPUT: 
"""

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    conn = engine.connect();
    
    df = pd.read_sql_table('disaster_messages_tbl', engine)
    X = df['message']
    Y = df.select_dtypes('int64').drop('id', axis=1)
    
    conn.close()
    return X, Y, df.columns

"""
    Function Objective: Apply string manipulation and clean up text.
    INPUT: path to text to clean
    OUTPUT: the processed text
"""

def tokenize(text):
    
    normalised_text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    applied_tokenize = word_tokenize(normalised_text)
    
    stop_words = set(stopwords.words("english"))
    removed_stop_words = [s for s in applied_tokenize if s not in stop_words]
    
    lemmed_words = [WordNetLemmatizer().lemmatize(w) for w in removed_stop_words]
    
    return lemmed_words

"""
    Function Objective: Build the model pipeline
    INPUT: Nothing, this is basically set parametrs for a random forrest model and it's parameters for tuning.
    OUTPUT: The grid search to feed into the model fitting
"""

def build_model():
    classifier = RandomForestClassifier(n_estimators = 10, random_state = 42, verbose = 10)
    pipe = Pipeline(
                    [('count', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('model', MultiOutputClassifier(estimator = classifier, n_jobs = 1))
                    ])
    
#     print(pipe.get_params().keys())
                     
    parameter_grid = {
                      'model__estimator__max_depth' : [3,6,9],
                      'model__estimator__max_features': [3,6,9],
                      'model__estimator__criterion' : ['gini', 'entropy']
                    }
                     
    cv = GridSearchCV(pipe, param_grid = parameter_grid)
                     
    return cv
                     
"""
    Function Objective: Analyse fitted model using predictionson test data and print out classification report.
    INPUT: Model: the fitted model
           X_Test: the previously split out test data.
           Y_test: the correspondingly split out response variable
    OUTPUT: No retunr, but Classification report gets printed to logs.
"""
                     
def evaluate_model(model, X_test, Y_test):
    
    Prediction_on_test = model.predict(X_test);
    df_predictions = pd.DataFrame(Prediction_on_test);
    df_predictions.columns = Y_test.columns;
    df_predictions.index = Y_test.index;
    
    for column in Y_test.columns:
        print('Column: ' , column)
        print(classification_report(Y_test[column], df_predictions[column]))

"""
    Function Objective: Save the model and pickle dump it.
    INPUT: Model: the fitted model
           model_filepath: where to save the model
    OUTPUT: No return, but model dumped in location specified.
"""
                     
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'));


def main():
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
        evaluate_model(model, X_test, Y_test)

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