import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def median( df ):
    print("median:", df[ 'horsepower' ].median() )

def dot( xi, w ):
    n = len(xi)
    res = 0.0
    for j in range(n):
        res = res + xi[j] * w[j]

    return res

def _linear_regression( xi ):
    xi = [1]+xi
    return dot( x1, xnew=[7.17, 0.01, 0.04, 0.002] )

def train_linear_regression( X, y ):
    ones = np.ones( X.shape[0] )
    X = np.column_stack( [ones,X] )
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]

def prepare_dataset( df, seed=42 ):
    n = len( df )
    n_val = int( n * 0.2 )
    n_test = int( n * 0.2 )
    n_train = n - n_val - n_test

    np.random.seed( seed )
    idx = np.arange(n)
    print("idx:", idx )
    np.random.shuffle(idx)
    print("idx:", idx )

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train+n_val]]
    df_test = df.iloc[idx[n_train + n_val:]]

    # print("df_train:", df_train )
    # print("df_val:", df_val )
    # print("df_test:", df_test )

    # y_pred = linear_regression([453,11,86])
    # print("y_pred:", y_pred )

    y_train = np.log1p( df_train.fuel_efficiency_mpg.values )
    y_val = np.log1p( df_val.fuel_efficiency_mpg.values )
    y_test = np.log1p( df_test.fuel_efficiency_mpg.values )

    return df_train, df_val, df_test, y_train, y_val, y_test

def linear_regression( xi ):
    n = len(xi)
    w0 = 7.17
    w = [ 0.01, 0.04, 0.002 ]
    pred = w0
    for j in range(n):
        pred += w[j] * xi[j]

    return pred

def rmse( y, y_pred ):
    se = ( y - y_pred ) ** 2
    mse = se.mean()
    return np.sqrt( mse )

def prepare_X( df ):
    features = [ 'engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg' ]
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

def train_linear_regression_reg( X, y, r=0.001 ):
    ones = np.ones( X.shape[0] )
    X = np.column_stack( [ones,X])
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye( XTX.shape[0] )

    XTX_inv = np.linalg.inv( XTX )
    w_full = XTX_inv.dot( X.T ).dot( y )

    return w_full[0], w_full[1:]

def main():
    dataset = pd.read_csv("car_fuel_efficiency_2026.csv")
    features = [ 'engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg' ]
    df = dataset[ features ]
    print("head:", df.head() )
    # sns.histplot( df.fuel_efficiency_mpg )

    print( df[ features ].isnull().sum() )

    df_z = df[features].fillna(0)
    df_a = df[features].fillna(0)

    df_train, df_val, df_test, y_train, y_val, y_test = prepare_dataset( df_z )

    median(df)

    x_train = df_train[ features ].values

    w0, w = train_linear_regression( x_train , y_train )
    print("w0:", w0 )
    print("w:", w)
    y_pred = w0 + x_train.dot(w)

    # sns.histplot( y_pred, color='red', alpha=0.5, bins=50 )
    # sns.histplot( y_train, color='blue', alpha=0.5, bins=50 )
    plt.show()

    print("rmse:", rmse( y_train, y_pred ) )
    x_val = prepare_X( df_val )
    y_pred = w0 + x_val.dot(w).round(3)

    print( "score:",  rmse( y_val, y_pred) )

    rs = [ 0, 0.01, 0.1, 1, 5, 10, 100 ]

    for r in rs:
        x_train = prepare_X( df_train )
        w0, w = train_linear_regression_reg( x_train, y_train, r )

        x_val = prepare_X( df_val )
        y_pred = w0 + x_val.dot(w)
        # print("ypred", y_pred )
        print("r:", r )
        print( "rmse:", np.round( rmse( y_val, y_pred ), 4) )



def new_main( seed ):
    dataset = pd.read_csv("car_fuel_efficiency_2026.csv")
    df_train, df_val, df_test, y_train, y_val, y_test = prepare_dataset( dataset, seed )
    x_train = prepare_X( df_train )
    w0, w = train_linear_regression( x_train , y_train )
    x_val = prepare_X( df_val )
    y_pred = w0 + x_val.dot(w)
    return rmse( y_val, y_pred )

def q6( seed=9, r=0.001 ):
    dataset = pd.read_csv("car_fuel_efficiency_2026.csv")
    df_train, df_val, df_test, y_train, y_val, y_test = prepare_dataset( dataset, seed )
    x_combined = np.concatenate( [df_train, df_val ] )
    y_combined = np.concatenate( [y_train, y_val ] )
    w0, w = train_linear_regression_reg( x_combined, y_combined, r )
    x_test = prepare_X( df_test )
    y_pred = w0 + x_test.dot(w)
    error = rmse( y_combined, y_pred )
    print("error:", error ) 



if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    rmses = []
    for s in seeds:
        rmses.append( new_main(s) )

    print("std:", np.std(rmses) )
    q6()