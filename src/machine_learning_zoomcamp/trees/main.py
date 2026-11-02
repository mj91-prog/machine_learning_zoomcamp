from sklearn.metrics import roc_auc_score, auc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

CATEGORICAL = [ 'origin','fuel_type','drivetrain' ]
NUMERICAL = [ 'num_doors', 'engine_displacement', 'num_cylinders', 'horsepower', 'vehicle_weight', 'acceleration', 'fuel_efficiency_mpg' ]

def split_dataset( df ):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
                                        
    # reset the index 
    df_train.reset_index( drop=True )
    df_val.reset_index( drop=True )
    df_test.reset_index( drop=True )

    y_train = df_train['fuel_efficiency_mpg']
    y_test = df_test['fuel_efficiency_mpg']
    y_val = df_val['fuel_efficiency_mpg']

    del df_train['fuel_efficiency_mpg']
    del df_test['fuel_efficiency_mpg']
    del df_val['fuel_efficiency_mpg']

    return df_train, df_val, df_test, y_train.values, y_val.values, y_test.values, df_full_train

def remove_null_values(  df ):
    df[NUMERICAL] = df[NUMERICAL].fillna(0.0)
    df[CATEGORICAL] = df[CATEGORICAL].fillna("NA")


def Q1():
    df = pd.read_csv('car_fuel_efficiency_2026.csv')
    remove_null_values( df )

    # print( df.describe().round() )

    df_train, df_val, df_test, y_train, y_val, y_test, df_full_train = split_dataset(df)

    train_dicts = df_train.to_dict( orient='records' )
    dv = DictVectorizer( sparse=True )

    x_train = dv.fit_transform( train_dicts )
    dt = DecisionTreeRegressor( max_depth=1 )

    dt.fit( x_train, y_train )

    feature_names = dv.get_feature_names_out()
    rfi = dt.tree_.feature[0]
    split_feature = feature_names[rfi]
    print("Split feature:", split_feature)

def Q2():

    df = pd.read_csv('car_fuel_efficiency_2026.csv')
    remove_null_values( df )

    df_train, df_val, df_test, y_train, y_val, y_test, df_full_train = split_dataset(df)

    train_dicts = df_train.to_dict( orient='records' )
    dv = DictVectorizer( sparse=True )
    x_train = dv.fit_transform( train_dicts )

    rf = RandomForestRegressor( n_estimators=10, random_state=1, n_jobs=1 )
    rf.fit( x_train, y_train )

    val_dicts = df_val.to_dict( orient='records' )
    x_val = dv.transform( val_dicts )

    y_pred = rf.predict( x_val )

    mse = mean_squared_error( y_val, y_pred ) 
    rmse = round( np.sqrt(mse), 3 )

    print("rmse:", rmse )

def Q3():
    df = pd.read_csv('car_fuel_efficiency_2026.csv')
    remove_null_values( df )

    df_train, df_val, df_test, y_train, y_val, y_test, df_full_train = split_dataset(df)

    train_dicts = df_train.to_dict( orient='records' )
    dv = DictVectorizer( sparse=True )
    x_train = dv.fit_transform( train_dicts )

    for n in [ 10, 50, 100, 150 ]:

        rf = RandomForestRegressor( n_estimators=n, random_state=1, n_jobs=1 )
        rf.fit( x_train, y_train )
    
        val_dicts = df_val.to_dict( orient='records' )
        x_val = dv.transform( val_dicts )
    
        y_pred = rf.predict( x_val )
    
        mse = mean_squared_error( y_val, y_pred ) 
        rmse = round( np.sqrt(mse), 3 )
    
        print(f"n={n} - rmse:", rmse )

def Q4():
    df = pd.read_csv('car_fuel_efficiency_2026.csv')
    remove_null_values( df )

    df_train, df_val, df_test, y_train, y_val, y_test, df_full_train = split_dataset(df)

    train_dicts = df_train.to_dict( orient='records' )
    dv = DictVectorizer( sparse=True )
    x_train = dv.fit_transform( train_dicts )

    for m in [ 10, 15, 20, 25]:
        print("max_depth=",m)
        mean_rmse = []
        for n in [ 10, 50, 100, 150 ]:
    
            rf = RandomForestRegressor( n_estimators=n, max_depth=m, random_state=1, n_jobs=1 )
            rf.fit( x_train, y_train )
        
            val_dicts = df_val.to_dict( orient='records' )
            x_val = dv.transform( val_dicts )
        
            y_pred = rf.predict( x_val )
        
            mse = mean_squared_error( y_val, y_pred ) 
            rmse = round( np.sqrt(mse), 3 )
        
            print(f"n_estimators={n} - rmse:", rmse )
            mean_rmse.append(rmse)
        print( f"mean=", np.mean(mean_rmse) )

def Q5():
    df = pd.read_csv('car_fuel_efficiency_2026.csv')
    remove_null_values( df )

    df_train, df_val, df_test, y_train, y_val, y_test, df_full_train = split_dataset(df)

    train_dicts = df_train.to_dict( orient='records' )
    dv = DictVectorizer( sparse=True )
    x_train = dv.fit_transform( train_dicts )
    

    n_estimators=10
    max_depth=20
    random_state=1
    n_jobs=-1 

    rf = RandomForestRegressor( n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=n_jobs )
    rf.fit( x_train, y_train )

    feature_names = dv.get_feature_names_out()
    df_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    })

    df_importances = df_importances.sort_values(by='importance', ascending=False)

    print(df_importances.head(10))

def Q6():
    df = pd.read_csv('car_fuel_efficiency_2026.csv')
    remove_null_values( df )

    df_train, df_val, df_test, y_train, y_val, y_test, df_full_train = split_dataset(df)

    train_dicts = df_train.to_dict( orient='records' )
    dv = DictVectorizer( sparse=True )
    x_train = dv.fit_transform( train_dicts )
    val_dicts = df_val.to_dict( orient='records' )
    x_val = dv.transform( val_dicts )

    features = list(dv.get_feature_names_out())
    print("features")
    print(features)
    dtrain = xgb.DMatrix(x_train,label=y_train, feature_names=features )
    dval = xgb.DMatrix(x_val, label=y_val, feature_names=features )

    watchlist = [ (dtrain, 'train'), (dval,'val') ]

    for e in [ 0.3, 0.1 ]:
        xgb_params = {
            'eta': e, 
            'max_depth': 6,
            'min_child_weight': 1,
            
            'objective': 'reg:squarederror',
            'nthread': 8,
            
            'seed': 1,
            'verbosity': 1,
        }   

        model = xgb.train( xgb_params, dtrain, num_boost_round=100, verbose_eval=5, evals=watchlist )
        y_pred = model.predict(dval)
        mse = mean_squared_error( y_val, y_pred ) 
        rmse = round( np.sqrt(mse), 3 )
        print(f" eta={e} - rmse={rmse}")


if __name__ == "__main__":
    #Q1()
    #Q2()
    #Q3()
    #Q4() # max_depth=10 - mean rmse=1.754
    #Q5() # vehicle_weight    0.203313
    Q6()  # eta=0.3 - rmse=1.826 | eta=0.1 - rmse=1.725