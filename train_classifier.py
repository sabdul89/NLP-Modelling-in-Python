import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle

def load_data(database_filepath):
    """
    This function loads in data from the database filepath specifed
    and splits them into independent and target variables
    
    Input:
    database_filepath(.db):this is used to create a dataframe from the database filepath
    
    Output:
    X                : independent features:
    y                : target variable
    category_names   : categories of the dataset
    
    """
    
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('emergencies',con=engine)
    target=[col for col in df.columns if col not in ['id','message','original','genre']]
    X = df['message']
    Y=df[target]
    category_names=target   
    
    return X,Y,category_names
    

def tokenize(text):
    
    """
    This function takes in a text dataset,splits them into word tokens
    extracts proper words from these tokens using lemmatizer
    and returns a list of words for the text data
    Input:
        text (str): str text field that has messages that needs to be classified
    
    Output:
        returns word tokeds for these messages which can then be fed into a machine learning 
        model
    """
    
    
    tokens=word_tokenize(text)
    lemmatizer=WordNetLemmatizer()
    clean_tokens=[]
    
    for tok in tokens:
        clean_tok=lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """
    This function builds the model by using Pipeline and 
    finds the best parameters using sklearn.crossvalidation
    
    Output:
        Returns the model that was built
    
    """
    
    
    pipeline=Pipeline([
                       ('vect',CountVectorizer(tokenizer=tokenize)),
                       ('tfidf',TfidfTransformer()),
                       ('clf',MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
                      ])
    
    parameters = { 
               # 'vect__ngram_range':((1,1),(1,2)),
               #'vect__max_df':(0.5,0.75,1.0),
               #'vect__max_features':(None,5000,10000),
               #'tfidf__use_idf':(True,False),
               'clf__estimator__estimator__C' : list(range(1,10,2)),
               'clf__estimator__estimator__max_iter':[100,200,500]
                              
                }

    model = GridSearchCV(pipeline,param_grid=parameters,cv=5,scoring='accuracy',n_jobs=-1)
      
    return model
    


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    This function evaluates the model and prints out the classification
    report and accuracy score for each category in the dataset
    
    Input:
        model          :the model that was trained
        X_test         :the test data 
        y_test         :actual data for the test dataset
        category_names :Various categories in the dataset
    """
    
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    print ("Model final accuracy score is {}".format(accuracy_score(Y_test,y_pred)))
    
    return

def save_model(model, model_filepath):
    
    """ 
    This function saves the model in the specified filepath as a pkl file
     Input:-
            Model : Final tuned model
            model_filepath : Location for saving the model
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    return
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = ["data/DisasterResponse.db","models/classifier.pkl" ]
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
