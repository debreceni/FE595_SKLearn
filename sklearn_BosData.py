# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:40:01 2019

@author: dave
"""

from sklearn.linear_model import LinearRegression as LR
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np


def getCoef():
    #load the dataset
    bData = load_boston()
    X= pd.DataFrame(data=bData['data'], columns=bData['feature_names'])
    X = X.drop('CHAS', axis=1) 
    Y = pd.DataFrame(data=bData['target'], columns=['MEDV'])
    
    cols = list(X.columns)
    model = LR().fit(X, Y)
    coef = model.coef_
    return coef, cols
    

    
def main():
    
    #get the regression from the coef
    coefs, cols = getCoef()
    idx = np.argmax(abs(coefs))
    
    print('Most influential element is %s with a slope of %.2f'% (cols[idx],coefs[0,idx]))



if __name__ == '__main__':
    main()
