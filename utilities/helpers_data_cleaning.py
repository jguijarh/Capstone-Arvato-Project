import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
from pylab import *
from pathlib import Path

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels

def cast_categorical_h2o(hf, h2o_dict):
    '''Create a custom dictionary to be used in the H2O frames'''
    
    feat_cols = []
    for a, b in h2o_dict.items(): 
        if b == 'enum':
            feat_cols.append(str(a))

    feat_cols.append('CLUSTER')
    
    for feat in feat_cols:
        hf[feat] = hf[feat].asfactor()
    return hf

def read_feats(path_read: Path)-> dict:
    '''Read the information from attributes file'''
    
    df_feats = pd.read_excel(path_read,
                                header = 1,
                                usecols=['Attribute', 'Meaning']).dropna()

    df_feats['numeric'] = df_feats['Meaning'].str.startswith('numeric')

    dict_attributes = (df_feats[['Attribute', 'numeric']]
                       .set_index('Attribute')
                       .to_dict()
                       .get('numeric'))

    feat_dict = {feat: float if numeric else 'category'
                 for feat, numeric in dict_attributes.items()}
    
    return feat_dict
    
    
def convert_categorical(df_cat:pd.DataFrame)->pd.DataFrame:
    '''Convert the 0  and 0.0 that we have in the dataframe 
    in categorical and object column data to NaN values'''
    
    df = df_cat.copy()
    X_cat = df.select_dtypes(include=['category', 'object']).columns

    for feat in X_cat:
        df[feat] = df[feat].replace('0.0',np.nan)
        df[feat] = df[feat].replace('0',np.nan)

    return df
    
def distincts_in_col(df: pd.DataFrame):
    '''
    count the distinct values in each column and take the one 
    
    '''
    df_count = (pd.DataFrame(df.apply(lambda x: x.nunique()),
                             columns = ['count'])
                .sort_values('count',ascending=False)
                .reset_index())
    
    df_count.colu mns = ['feat','count']
    
    return df_count

def nan_in_col(df: pd.DataFrame):
    '''
    count the NaN values in each column and take the one 
    
    '''
    df_nans = (pd.DataFrame(df.apply(lambda x: np.round(x.isna().sum()/df.shape[0],3)),
                             columns = ['count'])
               .sort_values('count',ascending=False)
               .reset_index())
    
    df_nans.columns = ['feat','count']
    
    return df_nans

def generate_synthetic_train_data(df_customers: pd.DataFrame,
                                  df_popu: pd.DataFrame)-> pd.DataFrame:
    '''Create a synthetic dataframe with the population and customer data'''
    df_c = df_customers.copy()
    df_p = df_popu.copy()
    
    df_p['target_customer'] = 0
    df_c['target_customer'] = 1
    df_synthetic = pd.concat([df_p,df_c])
    
    return df_synthetic

def obtain_features_label(df:pd.DataFrame)-> [list,list]:
    '''obtain the data features and label in a df before train a model.'''
    label = [col for col in df if col.startswith('target')]
    feats = list(df.columns.drop(label))
    
    #remove the feat of ID that in this data is LNR
    try:
        feats.remove('LNR')
    except:
        pass           
    
    return feats, label

def mailout_features_label(df: pd.DataFrame):
    '''obtain the data features and label in a mailout df before train a model.'''
    label = [col for col in df if col.startswith('RESPONSE')]
    feats = list(df.columns.drop(label))
    
    #remove the feat of ID that in this data is LNR
    try:
        feats.remove('LNR')
    except:
        pass           
    
    return feats, label

def fill_categorical(df: pd.DataFrame) -> pd.DataFrame:
    
    '''Fill categorical columns with a value for the missing values. 
    If the column is of type object i just fill and i don't add a new category'''
    
    X_cat = df.select_dtypes(include=['category', 'object']).columns
    df_cat = df.copy()
    
    for feat in X_cat:
        # if column is category
        try:
            df_cat[feat] = df_cat[feat].cat.add_categories('MISS_VALUE').fillna('MISS_VALUE')

        except AttributeError:
        # if column is object
            df_cat[feat] = df_cat[feat].fillna('MISS_VALUE')

    return df_cat
    
def split_data(df:pd.DataFrame, perc_v_t: list):
    '''take a dataframe and preprocess to be able to use in a model. the step are:
    - select the features and label.
    - fill nan values in the categorical vairbles.
    - select train, val and test data sets.
    '''
    df_func = df
    val_perc = perc_v_t[0]
    test_perc = perc_v_t[1]

    features, label = obtain_features_label(df_func)
    df_func = fill_categorical(df_func)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            df_func.drop(columns=label),
            df_func[label],
            test_size=val_perc + test_perc,
            random_state=1993,
            stratify=df_func[label]))
    
    X_val, X_test, y_val, y_test = (
        train_test_split(
            X_test,
            y_test,
            test_size=val_perc/(val_perc + test_perc),
            random_state=1993,
            stratify=y_test))
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_pca(pca_model):
    ''' Plot the explained variances'''
    
    fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    plt.subplot(2,1,1)

    pca_features = range(pca_model.n_components_)
    plt.bar(pca_features, pca_model.explained_variance_ratio_, color='blue')
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.xticks(pca_features)

    plt.subplot(2,1,2)
    plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs Number of Components SS')
    plt.grid(b=True)


    plot = plt.show()


def pca_features(df, pca, dimensions):
    '''
    This function displays interesting 
    features of the selected dimension
    '''
    
    features = df.columns.values
    components = pca.components_
    feature_weights = dict(zip(features, components[dimensions]))
    sorted_weights = sorted(feature_weights.items(), key = lambda kv: kv[1])
    
    print('Highest: ')
    for feature, weight in sorted_weights[-dimensions:]:
        print('\t{:20} {:.3f}'.format(feature, weight))
    
    print('Lowest: ')
    for feature, weight, in sorted_weights[:dimensions]:
        print('\t{:20} {:.3f}'.format(feature, weight))
        

        