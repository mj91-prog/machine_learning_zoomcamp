import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, accuracy_score

NUMERIC = ['annual_income', 'number_of_courses_viewed', 'interaction_count', 'lead_score']
CATEGORICAL = ['lead_source', 'industry', 'employment_status', 'location']

GLOBAL_CONVERSION_RATE = 0

def Q1():
    df = pd.read_csv("course_lead_scoring_2026.csv")
    f = df['industry'].mode()[0]
    print("most frequent value:", f )

def Q2():
    df = pd.read_csv("course_lead_scoring_2026.csv")
    correlation_matrix = df.corr(numeric_only=True)
    plt.figure ( figsize=(8,6))
    sns.heatmap( correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.show()

def remove_null_values(  df ):
    df[ NUMERIC ] = df[ NUMERIC ].fillna(0.0)
    df[ CATEGORICAL ] = df[ CATEGORICAL ].fillna("NA")

def calc_global_conversion_rate( df ):
    GLOBAL_CONVERSION_RATE = df.converted.mean()

def prepare_dataset( df ):
    remove_null_values( df )
    df_full_train, df_test = train_test_split( df, test_size=0.2, random_state=42 )
    df_train, df_val = train_test_split( df_full_train, test_size=0.25, random_state=42 )

    # reset the index 
    df_train.reset_index( drop=True )
    df_val.reset_index( drop=True )
    df_test.reset_index( drop=True )

    y_train = df_train['converted']
    y_test = df_test['converted']
    y_val = df_val['converted']

    del df_train['converted']
    del df_test['converted']
    del df_val['converted']

    return df_train, df_val, df_test, y_train.values, y_val.values, y_test.values, df_full_train

def analyze_diff_risk( df ):
    calc_global_conversion_rate(df)
    for c in CATEGORICAL:
        df_group = df.groupby(c).converted.agg(['mean','count'])
        df_group['diff'] = df_group['mean'] - GLOBAL_CONVERSION_RATE
        df_group['risk'] = df_group['mean'] / GLOBAL_CONVERSION_RATE
        print( df_group )

def Q3():
    df = pd.read_csv("course_lead_scoring_2026.csv")
    _,_,_,_,_,_,dft = prepare_dataset(df)
    # print( dft )

    for category in CATEGORICAL:
        print( f"{category}:{mutual_info_score( dft[category], dft.converted )}")

def Q4():
    df = pd.read_csv("course_lead_scoring_2026.csv")
    df_train, df_val, df_test, y_train, y_val, y_test, _ = prepare_dataset(df)
    dv = DictVectorizer( sparse=False )
    train_dicts = df_train[ CATEGORICAL + NUMERIC ].to_dict( orient='records' )
    x_train = dv.fit_transform( train_dicts )

    val_dicts = df_val[ CATEGORICAL + NUMERIC ].to_dict( orient='records' )
    x_val = dv.transform( val_dicts )

    model = LogisticRegression( solver='liblinear', C=1.0, max_iter=1000, random_state=42 )
    model.fit( x_train, y_train )
    y_pred = model.predict_proba( x_val )[:,1]

    conv_decision = ( y_pred > 0.5 )
    # accuracy = ( y_val == conv_decision ).mean()
    accuracy = accuracy_score( y_val, conv_decision )
    print("accuracy:", accuracy )
    return accuracy

def train_and_solve( df_train, df_val, y_train, y_val, categories, c ):
    dv = DictVectorizer( sparse=False )
    x_train = dv.fit_transform( df_train[categories].to_dict( orient='records' ) )
    x_val = dv.transform( df_val[categories].to_dict( orient='records' ) )
    model = LogisticRegression(solver='liblinear', C=c, max_iter=1000, random_state=42)
    model.fit( x_train, y_train )
    y_pred = model.predict_proba(x_val)[:,1]
    accuracy = accuracy_score( y_val, y_pred > 0.5 )
    return accuracy


def Q5():
    df = pd.read_csv("course_lead_scoring_2026.csv")
    df_train, df_val, df_test, y_train, y_val, y_test, _ = prepare_dataset(df)
    full_acc = train_and_solve( df_train, df_val, y_train, y_val, CATEGORICAL + NUMERIC )

    NEW_NUMERIC = ['annual_income', 'number_of_courses_viewed', 'interaction_count', 'lead_score']
    NEW_CATEGORICAL = ['industry', 'employment_status', 'location']
    acc1 = train_and_solve( df_train, df_val, y_train, y_val, NEW_CATEGORICAL + NEW_NUMERIC )

    NEW_NUMERIC = ['annual_income', 'interaction_count', 'lead_score']
    NEW_CATEGORICAL = ['lead_source', 'industry', 'employment_status', 'location']
    acc2 = train_and_solve( df_train, df_val, y_train, y_val, NEW_CATEGORICAL + NEW_NUMERIC )

    NEW_NUMERIC = ['annual_income', 'number_of_courses_viewed', 'lead_score']
    NEW_CATEGORICAL = ['lead_source', 'industry', 'employment_status', 'location']
    acc3 = train_and_solve( df_train, df_val, y_train, y_val, NEW_CATEGORICAL + NEW_NUMERIC )

    print( "[lead_source] full-acc1 = ", full_acc - acc1 )
    print( "[number_of_courses_viewed] full-acc2 = ", full_acc - acc2 )
    print( "[interaction_count] full-acc3 = ", full_acc - acc3 )



def Q6():
    df = pd.read_csv("course_lead_scoring_2026.csv")
    df_train, df_val, df_test, y_train, y_val, y_test, _ = prepare_dataset(df)
    C = [0.000001, 0.00001, 0.0001, 0.001]

    for c in C:
        acc = train_and_solve( df_train, df_val, y_train, y_val, CATEGORICAL + NUMERIC, c )
        print(f"c:{c} - accuracy:{round(acc,3)}")


if __name__ == "__main__":
    #Q1() # technology
    #Q2() # interaction_count and lead_score
    #Q3() # lead_source:0.0347609795100606
    #Q4() # accuracy: 0.645
    #Q5() # number_of_courses_viewed
    Q6() # 0.001 - accuracy:0.645