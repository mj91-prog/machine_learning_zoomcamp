import pandas as pd

def main():
    print(pd.__version__)
    df = pd.read_csv("car_fuel_efficiency_2026.csv")
    print( df.head() )
    print( "len:", len(df))

    print( "fuel_types:", df['fuel_type'].unique() )
    print("is_null:", df.isnull().sum() )
    print( "MAX: ", df[ df['origin'] == "Asia" ].max() )

    print("median horsepowa: ", df['horsepower'].median() )

    most_frequent_value = df['horsepower'].value_counts().idxmax();
    print("most frequent value:", most_frequent_value )
    print("how many are null? ", df['horsepower'].isnull().sum() )
    df['horsepower'] = df['horsepower'].fillna( most_frequent_value )
    print("how many are null after? ", df['horsepower'].isnull().sum() )
    print("median horsepowa[double]: ", df['horsepower'].median() )

    # Linear Regression


    print( "HEAD: ", df[ ['vehicle_weight', 'model_year'] ].head(7) )

    import numpy as np
    df = df[ df['origin'] == "Asia" ]
    X = df[ ['vehicle_weight', 'model_year'] ].head(7).values
    print("X:")
    print(X)

    XTX = X.T.dot(X)
    print("XTX:")
    print(XTX)

    y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

    inverse_XTX = np.linalg.inv( XTX )
    I = inverse_XTX.dot(X.T)
    w = I.dot(y.T)

    print("w:",w)
    print("sum:", w.sum() )

if __name__ == "__main__":
    main()