import pandas as pd
import numpy as np
import sklearn.cross_validation as CV
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.feature_extraction import DictVectorizer as DV
from datetime import date
import itertools

def myScore(y, y_pred):
    res = np.ndarray([1,2]);
    u = (np.power((y-y_pred),2)).sum();
    v = (np.power((y-y.mean()),2)).sum();
    res[0,0] = 1-(u/v);

    yLog = np.log(y);
    ind = np.isinf(yLog);
    yLog[ind == True] = 0; 
    y_predLog = np.log(y_pred);
    ind = np.isinf(y_predLog);
    y_predLog[ind == True] = 0;   
    summ = np.sum(np.power(yLog-y_predLog,2))
    res[0,1] = np.sqrt(summ/y.shape[0]);
    return res

# replace qualitative estimation to number
def replaceQualVal(dataSet,fNameList):
    # dictionary of values
    vocab = {
        'Ex': 5, 'EX': 5, # excellent    
        'Gd': 4, 'GD': 4, # good
        'TA': 3, 'Ta': 3, # normal
        'FA': 2, 'Fa': 2, # fair
        'PO': 1, 'Po': 1  # 
        }
    
    for fName in fNameList:
    # replace stings to numbers
        for word in vocab:
            searchDict = {fName:[word]};
            X = dataSet.isin(searchDict);
            dataSet.loc[X[fName],fName] = vocab[word];
        
        # convert to numeric type
        dataSet[[fName]] = dataSet[[fName]].apply(pd.to_numeric);
    return dataSet;

# replace YN estimation to number
def replaceYNVal(dataSet,fNameList):
    # dictionary of values
    vocab = {
        'Yes': 1, 'Y': 5, # yes
        'No': 1, 'N': 1  # no
        }
    
    for fName in fNameList:
        
        # replace stings to numbers
        for word in vocab:
            searchDict = {fName:[word]};
            X = dataSet.isin(searchDict);
            dataSet.loc[X[fName],fName] = vocab[word];
        
        # convert to numeric type
        dataSet[[fName]] = dataSet[[fName]].apply(pd.to_numeric);
    return dataSet;

# main fit function
def fitData(folds,regressor,features):
    
    num_features = features.select_dtypes(exclude=['object']);
    num_features.fillna(0,inplace=True);
    obj_features = features.select_dtypes(include=['object']);
    obj_features.fillna('empty',inplace=True)
    
    encoder = DV(sparse = False);
    encoded_data = encoder.fit_transform(obj_features.T.to_dict().values());
    newFeatures = np.hstack([num_features, encoded_data]);
    score = np.empty([1,2]);
    
    for [trainInds, testInds] in folds:
        regressor.fit(newFeatures[trainInds,:],price[trainInds]);
        y_pr = regressor.predict(newFeatures[testInds,:]);
        pr = myScore(price[testInds],y_pr);
        print pr
        score =  np.append(score,pr,axis = 0);
    
    score = np.delete(score,0,0);
    return score
    
data = pd.read_csv('train.csv',index_col='Id')
features = data.drop('SalePrice',axis = 1)
price = data.SalePrice;

# define regressor
trees = GBR(verbose = 0, n_estimators = 1000, max_depth = 3);

# define cross-validation folds
nFolds = 10;
folds = CV.KFold(price.size, n_folds=nFolds, random_state = 43);

features = data.drop('SalePrice',axis = 1)
price = data['SalePrice'].get_values()


#features["OverallQualLog"]=np.log(features["OverallQual"])
#features.drop(['OverallQual'],axis = 1,inplace = True);

# drop heating type
#features.drop('Heating',axis = 1,inplace = True);
#features.drop(['MoSold', 'YrSold'],axis = 1,inplace = True);
features = replaceYNVal(features,{'CentralAir'});

# process GarageYrBlt
empty = pd.isnull(features['GarageYrBlt']);
ind = empty[empty == True].index;
features.loc[ind.values,'GarageYrBlt'] = features.loc[ind.values,'YearBuilt'];

# add new feature
features.loc[:,'houseAgeLog'] = np.log(date.today().year - features.loc[:,'YearBuilt']);
features.loc[:,'garageAgeLog'] = np.log(date.today().year - features.loc[:,'GarageYrBlt']);
features.loc[:,'remodeAge'] = features.loc[:,'YearRemodAdd'] - features.loc[:,'YearBuilt'];


featureList = {'ExterQual','ExterCond','BsmtQual','BsmtCond','PoolQC','HeatingQC','KitchenQual','GarageQual',
               'GarageCond','FireplaceQu','PoolQC'};
res = list();

for fNames in itertools.combinations(featureList, 1):
    print fNames
    newFeatures = replaceQualVal(features,fNames);
    
    num_features = newFeatures.select_dtypes(exclude=['object']);
    num_features.fillna(0,inplace=True);
    
    obj_features = newFeatures.select_dtypes(include=['object']);
    obj_features.fillna('empty',inplace=True)
    encoder = DV(sparse = False);
    encoded_data = encoder.fit_transform(obj_features.T.to_dict().values());
    arrFeatures = np.hstack([num_features, encoded_data]);
       
    y_pr = CV.cross_val_predict(trees, arrFeatures, y=price, cv=folds, n_jobs=1, verbose=0)
    sc = myScore(price, y_pr);
    res.append([fNames,sc]);
    print sc
    
for fNames in itertools.combinations(featureList, 2):
    print fNames
    newFeatures = replaceQualVal(features,fNames);
    
    num_features = newFeatures.select_dtypes(exclude=['object']);
    num_features.fillna(0,inplace=True);
    
    obj_features = newFeatures.select_dtypes(include=['object']);
    obj_features.fillna('empty',inplace=True)
    encoder = DV(sparse = False);
    encoded_data = encoder.fit_transform(obj_features.T.to_dict().values());
    arrFeatures = np.hstack([num_features, encoded_data]);
       
    y_pr = CV.cross_val_predict(trees, arrFeatures, y=price, cv=folds, n_jobs=8, verbose=0)
    sc = myScore(price, y_pr);
    res.append([fNames,sc]);
    print sc
    
for fNames in itertools.combinations(featureList, 3):
    print fNames
    newFeatures = replaceQualVal(features,fNames);
    
    num_features = newFeatures.select_dtypes(exclude=['object']);
    num_features.fillna(0,inplace=True);
    
    obj_features = newFeatures.select_dtypes(include=['object']);
    obj_features.fillna('empty',inplace=True)
    encoder = DV(sparse = False);
    encoded_data = encoder.fit_transform(obj_features.T.to_dict().values());
    arrFeatures = np.hstack([num_features, encoded_data]);
       
    y_pr = CV.cross_val_predict(trees, arrFeatures, y=price, cv=folds, n_jobs=8, verbose=0)
    sc = myScore(price, y_pr);
    res.append([fNames,sc]);
    print sc
    
for fNames in itertools.combinations(featureList, 4):
    print fNames
    newFeatures = replaceQualVal(features,fNames);
    
    num_features = newFeatures.select_dtypes(exclude=['object']);
    num_features.fillna(0,inplace=True);
    
    obj_features = newFeatures.select_dtypes(include=['object']);
    obj_features.fillna('empty',inplace=True)
    encoder = DV(sparse = False);
    encoded_data = encoder.fit_transform(obj_features.T.to_dict().values());
    arrFeatures = np.hstack([num_features, encoded_data]);
       
    y_pr = CV.cross_val_predict(trees, arrFeatures, y=price, cv=folds, n_jobs=8, verbose=0)
    sc = myScore(price, y_pr);
    res.append([fNames,sc]);
    print sc
    
for fNames in itertools.combinations(featureList, 5):
    print fNames
    newFeatures = replaceQualVal(features,fNames);
    
    num_features = newFeatures.select_dtypes(exclude=['object']);
    num_features.fillna(0,inplace=True);
    
    obj_features = newFeatures.select_dtypes(include=['object']);
    obj_features.fillna('empty',inplace=True)
    encoder = DV(sparse = False);
    encoded_data = encoder.fit_transform(obj_features.T.to_dict().values());
    arrFeatures = np.hstack([num_features, encoded_data]);
       
    y_pr = CV.cross_val_predict(trees, arrFeatures, y=price, cv=folds, n_jobs=8, verbose=0)
    sc = myScore(price, y_pr);
    res.append([fNames,sc]);
    print sc