import pandas as pd
import numpy as np 

import seaborn as sns
import matplotlib.pyplot as plt


from scipy import stats
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import f_regression, SelectKBest, RFE 
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt 


def clustering(train, f1, f2):
    '''
    This function is creating unscaled and scaled clusters and adding columns to the dataset
    '''
    
    seed = 22
    
    X = train[[f1, f2]]
    
    kmeans = KMeans(n_clusters = 4, random_state= seed)
    kmeans.fit(X)
    kmeans.predict(X)

    X['unscaled_clusters'] = kmeans.predict(X)
    
    mm_scaler = MinMaxScaler()
    X[[f1, f2]] = mm_scaler.fit_transform(X[[f1, f2]])
    
    kmeans_scale = KMeans(n_clusters = 4, random_state = 22)
    kmeans_scale.fit(X[[f1, f2]])
    kmeans_scale.predict(X[[f1, f2]])
    
    X['scaled_clusters'] = kmeans_scale.predict(X[[f1, f2]])
    
    return X  


def cluster_relplot(df, f1, f2):
    '''
    This functions creates a relplot of the clusters
    '''
    
    sns.set(style = "whitegrid")
    
    X = clustering(df, f1, f2)
    
    sns.relplot(data = X, x = f1, y = f2, hue = 'unscaled_clusters')
    
    plt.title('Clusters')
    
    return plt.show() 


def best_cluster(df, f1, f2):
    '''
    This function makes a graph to show the most optimal cluster number
    '''
    
    X = clustering(df, f1, f2)
    
    inertia = []
    seed = 22 

    for n in range(1,11):

        kmeans = KMeans(n_clusters = n, random_state = seed)

        kmeans.fit(X[[f1, f2]])

        inertia.append(kmeans.inertia_)
        
        
    results_df = pd.DataFrame({'n_clusters': list(range(1,11)),
                               'inertia': inertia})   
    
    sns.set_style("whitegrid")
    sns.relplot(data = results_df, x='n_clusters', y = 'inertia', kind = 'line')
    plt.xticks(np.arange(0, 11, step=1))
    point = (3, 107) # specify the x and y values of the point to annotate
   # plt.annotate("optimal cluster", xy=point, xytext=(5, 125), 
                 #arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.title('The Best Cluster')
    
    return plt.show()


def combined_df(df, f1, f2):
    
    X = clustering(df, f1, f2)
    
    scaled_clusters = X['scaled_clusters']
    df3 = pd.merge(df, scaled_clusters, left_index=True, right_index=True)
    
    return df3


def mvp_scaled_data(train, val, test, return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    train.drop(columns= ['wisdom'], inplace= True)
    columns_scale = train.iloc[:, :16]
    columns_to_scale = columns_scale.columns
    
    train, val, test= the_split(df)
    
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()
    #     make the thing
    mms = MinMaxScaler()
    #     fit the thing
    mms.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(mms.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    val_scaled[columns_to_scale] = pd.DataFrame(mms.transform(val[columns_to_scale]), 
                                                     columns=val[columns_to_scale].columns.values).set_index([val.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(mms.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, val_scaled, test_scaled
    
    
def six_split(train, val, test, target):
    """
    this functions splits the data into 6 different datasets. We will use them for 
    our modeling 
    """
    # split into X and y train dataset 
    X_train= train.drop(columns=[target])
    y_train= train[target]

    # split into X and y val dataset 
    X_val= val.drop(columns=[target])
    y_val= val[target]

    # split into X and y test dataset 
    X_test= test.drop(columns=[target])
    y_test= test[target]

    X_train= pd.DataFrame(X_train)
    X_val= pd.DataFrame(X_val)
    X_test= pd.DataFrame(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def splitting_subsets(train_scaled, val_scaled, test_scaled, target):
    '''
    This function splits our train, validate, and test scaled datasets into X/y train,
    validate, and test subsets
    '''
    
    
    X_train_scaled= train_scaled.drop(columns= [target])
    X_train_scaled= pd.get_dummies(X_train_scaled, columns= ['scaled_clusters'])
    y_train_unscaled= train_scaled[target]


    X_val_scaled= val_scaled.drop(columns= [target])
    X_val_scaled= pd.get_dummies(X_val_scaled, columns= ['scaled_clusters'])
    y_val_unscaled= val_scaled[target]


    X_test_scaled= train_scaled.drop(columns= [target])
    X_test_scaled= pd.get_dummies(X_test_scaled, columns= ['scaled_clusters'])
    y_test_unscaled= train_scaled[target]
    
    
    return X_train_scaled, y_train_unscaled, X_val_scaled, y_val_unscaled, X_test_scaled, y_test_unscaled


def baseline(y_train_unscaled):
    '''
    This function takes in y_train to calculate the baseline rmse
    '''
    
    preds_df = pd.DataFrame({'actual': y_train_unscaled})
    
    preds_df['baseline'] = y_train_unscaled.mean()
    
    baseline_rmse = sqrt(mean_squared_error(preds_df.actual, preds_df.baseline))

    return baseline_rmse


def lasso_lars(X_train_scaled, y_train_unscaled):
    metrics = []

    for i in np.arange(0.05, 1, .05):
    
        lasso= LassoLars(alpha = i )
    
        lasso.fit(X_train_scaled, y_train_unscaled)
    
        lasso_preds= lasso.predict(X_train_scaled)
        
        preds_df= pd.DataFrame({'actual': y_train_unscaled})
    
        preds_df['lasso_preds'] = lasso_preds

        lasso_rmse= sqrt(mean_squared_error(preds_df['actual'], preds_df['lasso_preds']))
    
        output = {
                'alpha': i,
                'lasso_rmse': lasso_rmse
                 }
    
        metrics.append(output)

    df = pd.DataFrame(metrics)    
    return df.sort_values('lasso_rmse')


def linear_model(X_train_scaled, y_train_unscaled):
    
    lm= LinearRegression()

    lm.fit(X_train_scaled, y_train_unscaled)
    
    lm_preds= lm.predict(X_train_scaled)
    
    preds_df= pd.DataFrame({'actual': y_train_unscaled,'lm_preds': lm_preds})
    
    lm_rmse= sqrt(mean_squared_error(preds_df['lm_preds'], preds_df['actual']))
    
    df= pd.DataFrame({'model': 'linear', 'linear_rmse': lm_rmse},index=['0']) 
                      
    return df

def tweedie_models(X_train_scaled, y_train_unscaled):
    metrics = []

    for i in range(0, 4, 1):
    
        tweedie = TweedieRegressor(power = i)
    
        tweedie.fit(X_train_scaled, y_train_unscaled)
    
        tweedie_preds = tweedie.predict(X_train_scaled)
        
        preds_df = pd.DataFrame({'actual': y_train_unscaled})
    
        preds_df['tweedie_preds'] = tweedie_preds
    
        tweedie_rmse = sqrt(mean_squared_error(preds_df.actual, preds_df.tweedie_preds))
    
        output = {
                'power': i,
                'tweedie_rmse': tweedie_rmse
                 }
    
        metrics.append(output)

    df = pd.DataFrame(metrics)    
    return df.sort_values('tweedie_rmse') 


def linear_poly(X_train, y_train):
    metrics = []

    for i in range(2,4):

        pf = PolynomialFeatures(degree = i)

        pf.fit(X_train, y_train)

        X_polynomial = pf.transform(X_train)

        lm2 = LinearRegression()

        lm2.fit(X_polynomial, y_train)
        
        preds_df = pd.DataFrame({'actual': y_train})

        preds_df['poly_preds'] = lm2.predict(X_polynomial)

        poly_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['poly_preds']))

        output = {
                'degree': i,
                'poly_rmse': poly_rmse
                 }

        metrics.append(output)

    df = pd.DataFrame(metrics)    
    return df.sort_values('poly_rmse') 


def validate_models(X_train_scaled, y_train_unscaled, X_val_scaled, y_val_unscaled):
   
    lm = LinearRegression()

    lm.fit(X_train_scaled, y_train_unscaled)
    
    lm_val = lm.predict(X_val_scaled)
    
    val_preds_df = pd.DataFrame({'actual_val': y_val_unscaled})
    
    val_preds_df['lm_preds'] = lm_val

    lm_rmse_val = sqrt(mean_squared_error(val_preds_df['actual_val'], val_preds_df['lm_preds']))

    #tweedie model
    
    tweedie = TweedieRegressor(power = 1)
    
    tweedie.fit(X_train_scaled, y_train_unscaled)
    
    tweedie_val = tweedie.predict(X_val_scaled)
    
    val_preds_df['tweedie_preds']= tweedie_val
    
    tweedie_rmse_val= sqrt(mean_squared_error(val_preds_df.actual_val, val_preds_df.tweedie_preds))
    
    #polynomial model
    
    pf = PolynomialFeatures(degree = 2)
    
    pf.fit(X_train_scaled, y_train_unscaled)
    
    X_train = pf.transform(X_train_scaled)
    X_val = pf.transform(X_val_scaled)
    
    lm2 = LinearRegression()
    
    lm2.fit(X_train_scaled, y_train_unscaled)
    
    val_preds_df['poly_vals']= lm2.predict(X_val_scaled)
    
    poly_validate_rmse= sqrt(mean_squared_error(val_preds_df.actual_val, val_preds_df['poly_vals']))

    #lasso_lars model
    
    lasso = LassoLars(alpha = .05 )
    
    lasso.fit(X_train_scaled, y_train_unscaled)
    
    lasso_val = lasso.predict(X_val_scaled)
    
    val_preds_df['lasso_preds'] = lasso_val

    lasso_rmse_val = sqrt(mean_squared_error(val_preds_df.actual_val, val_preds_df['lasso_preds']))
    
    
    return lm_rmse_val,lasso_rmse_val, poly_validate_rmse


def test_model(X_train_scaled, y_train_unscaled, X_test_scaled, y_test_unscaled):

    lm = LinearRegression()

    lm.fit(X_train_scaled, y_train_unscaled)
    
    lm_preds = lm.predict(X_test_scaled)

    test_preds_df = pd.DataFrame({'actual_test': y_test_unscaled})

    test_preds_df['linear_test'] = lm.predict(X_test_scaled)

    linear_test_rmse = sqrt(mean_squared_error(test_preds_df.actual_test, test_preds_df['linear_test']))
    
    return linear_test_rmse


def best_models(X_train_scaled, y_train_unscaled, X_val_scaled, y_val_unscaled):
    
    lm_rmse = linear_model(X_train_scaled, y_train_unscaled).iloc[0,1]
    
    lasso_rmse = lasso_lars(X_train_scaled, y_train_unscaled).iloc[0,1]
    
    #tweedie_rmse = tweedie_models(X_train, y_train).iloc[0,1]
        
    poly_rmse = linear_poly(X_train, y_train_unscaled).iloc[1,1]
    
    baseline_rmse = baseline(y_train_unscaled)
    
    lm_rmse_val, lasso_rmse_val, poly_validate_rmse = validate_models(X_train_scaled, y_train_unscaled,
                                                                      X_val_scaled, y_val_unscaled)
    
    df = pd.DataFrame({'model': ['linear', 'lasso_lars','linear_poly', 'baseline'],
                      'train_rmse': [lm_rmse, lasso_rmse, poly_rmse,  baseline_rmse],
                      'validate_rmse': [lm_rmse_val, lasso_rmse_val, poly_validate_rmse, baseline_rmse]})
    
    df['difference'] = df['train_rmse'] - df['validate_rmse']
    
    return df.sort_values('difference').reset_index().drop(columns = ('index'))


def plot_model(best_mods):
        
    color= ['grey', 'grey', 'grey', 'firebrick']
    ax = sns.barplot(x='model', y='validate_rmse',
                 data= best_mods,
                 palette= color,
                 errwidth=0, ec= 'black')
    for i in ax.containers:
        ax.bar_label(i,)
plot_model()


def best_model(X_train, y_train, X_val, y_val, X_test, y_test):
    
    df = best_models(X_train, y_train, X_val, y_val).iloc[1]
    
    df['test_rmse'] = test_model(X_train, y_train, X_test, y_test)
    
    df = pd.DataFrame(df).T
    
    #df = df.drop(columns = ['difference'])

    return df

