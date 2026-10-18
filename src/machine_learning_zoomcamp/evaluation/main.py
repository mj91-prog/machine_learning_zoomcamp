from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NUMERIC = ['annual_income', 'number_of_courses_viewed', 'interaction_count', 'lead_score']
CATEGORICAL = ['lead_source', 'industry', 'employment_status', 'location']

def prepare_data( df ):
    df[ NUMERIC ] = df[ NUMERIC ].fillna(0.0)
    df[ CATEGORICAL ] = df[ CATEGORICAL ].fillna("NA")

def split_dataset( df ):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
                                        
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

def Q1():
    df = pd.read_csv("course_lead_scoring_2026.csv")
    print( df.head() )
    prepare_data( df )
    df_train, df_val, df_test, y_train, y_val, y_test, df_full_train = split_dataset( df )

    print("len df_train:", len(df_train))
    print("len y_train:", len(y_train))

    for category in NUMERIC:
        x, y, _ = roc_curve( y_train, df_train[category] )
        score = auc(x,y)
        if score < 0.5:
            score = -score
        print(f"{category} -> {score}")

def train_and_solve( df_train, df_val, y_train, y_val, categories, c ):
    dv = DictVectorizer( sparse=False )
    x_train = dv.fit_transform( df_train[categories].to_dict( orient='records' ) )
    x_val = dv.transform( df_val[categories].to_dict( orient='records' ) )
    model = LogisticRegression(solver='liblinear', C=c, max_iter=1000 )
    model.fit( x_train, y_train )
    y_pred = model.predict_proba(x_val)[:,1]
    return y_pred

def calc_aur( y_val, y_pred ):
    fpr, tpr, _ = roc_curve( y_val, y_pred )
    score = auc(fpr,tpr)
    print( "AUC:", round(score,3) )

def Q2():
    df = pd.read_csv("course_lead_scoring_2026.csv")
    print( df.head() )
    prepare_data( df )
    df_train, df_val, df_test, y_train, y_val, y_test, df_full_train = split_dataset( df )

    y_pred = train_and_solve( df_train, df_val, y_train, y_val, CATEGORICAL + NUMERIC, 1.0 )
    calc_aur( y_val, y_pred )

def calc_confusion_matrix( y_val, y_pred, t ):
    actual_positive = ( y_val == 1 )
    actual_negative = ( y_val == 0 )
    predict_positive = ( y_pred >= t )
    predict_negative = ( y_pred < t )
    tp = ( predict_positive & actual_positive).sum()
    tn = ( predict_negative & actual_negative ).sum()
    fp = ( predict_positive & actual_negative ).sum()
    fn = ( predict_negative & actual_positive ).sum()

    return tp, tn, fp, fn

def Q3_4():
    df = pd.read_csv("course_lead_scoring_2026.csv")
    prepare_data( df )
    df_train, df_val, df_test, y_train, y_val, y_test, df_full_train = split_dataset( df )
    y_pred = train_and_solve( df_train, df_val, y_train, y_val, CATEGORICAL + NUMERIC, 1.0 )

    precisions = []
    recalls = []
    thresholds = np.linspace(0,1,101)
    scores = []
    c_rs = []
    for t in thresholds:
        tp, tn, fp, fn = calc_confusion_matrix( y_val, y_pred, t )
        # print(f"{tp}, {tn}, {fp}, {fn}")
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        score = calc_f1_score(p,r)
        c_rs.append((t,score))
        precisions.append(p)
        recalls.append(r)
        scores.append( score )

    max_score = max( scores )
    for c in c_rs:
        if c[1] == max_score:
            print("t:", c[0])

    plt.plot( thresholds, precisions, label="Precision" )
    plt.plot( thresholds, recalls, label="Recall" )
    plt.plot( thresholds, scores, label="F1")
    plt.legend()
    plt.show()

def calc_f1_score( p, r ):
    return 2*((p*r)/(p+r))

def train( df_train, y_train, C=1.0 ):
    dicts = df_train[ CATEGORICAL + NUMERIC ].to_dict( orient='records' )
    dv = DictVectorizer( sparse=False )
    x_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(x_train, y_train)

    return dv, model

def predict( df, dv, model ):
    dicts = df[ CATEGORICAL + NUMERIC ].to_dict( orient='records' )
    x = dv.transform(dicts)
    y_pred = model.predict_proba(x)[:,1]

    return y_pred

def Q5_6():
    df = pd.read_csv("course_lead_scoring_2026.csv")
    prepare_data( df )
    df_train, df_val, df_test, y_train, y_val, y_test, df_full_train = split_dataset( df )
    # dv, model = train( df_train, y_train )
    # y_pred = predict( df_val, dv, model )
    for C in [0.000001, 0.001, 1]:
        kfold = KFold(n_splits=5, shuffle=True, random_state=1)
        scores = []
        for train_idx, val_idx in kfold.split(df_full_train):
            df_train = df_full_train.iloc[train_idx]
            df_val = df_full_train.iloc[val_idx]

            y_train = df_train.converted.values
            y_val = df_val.converted.values
            dv, model = train( df_train, y_train, C )
            y_pred = predict( df_val, dv, model )

            auc = roc_auc_score( y_val, y_pred )
            scores.append(auc)

        print("scores:", scores )
        print("C:", C )
        print("mean:", np.mean(scores))
        print("std:", np.std(scores))


if __name__ == "__main__":
    # Q1() # lead_score -> 0.7882254810906102
    # Q2() # AUC: 0.732
    # Q3_4() # intersection at 0.73, t: 0.41000000000000003
    Q5_6() # std: 0.006853233151022138 ,C: 0.001 -mean: 0.7491721776030379