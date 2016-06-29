from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
import csv
    
# imports csv file
f = open("../../DBTT_Data.csv")
title = f.readline().split(',')
data =  np.loadtxt(f, delimiter = ',')

dataTemp = data[:, np.newaxis]
# converts each column in the excell file into an array
XCu = (dataTemp[:, :, title.index("N(Cu)")])
XNi = (dataTemp[:, :, title.index("N(Ni)")])
XMn = (dataTemp[:, :, title.index("N(Mn)")])
XP  = (dataTemp[:, :, title.index("N(P)")])
XSi = (dataTemp[:, :, title.index("N(Si)")])
XC  = (dataTemp[:, :, title.index("N( C )")])
XFl = (dataTemp[:, :, title.index("N(log(fluence)")])
XFx = (dataTemp[:, :, title.index("N(log(flux)")])
XT  = (dataTemp[:, :, title.index("N(Temp)")])

Ydata = (dataTemp[:, :, title.index("delta sigma")])
Xdata = np.concatenate((XCu, XNi, XMn, XP, XSi, XC, XFl, XFx, XT), axis=1)


Y_predicted_best = []
Y_predicted_worst = []

L = []
maxRMS = 10
minRMS = 40
for a in range (9,18):
    for b in range (1,10):
        for c in range (10,110,10):
            RMS_List = []
            num_folds = 5
            l = []
            for n in range(200):
                kf = cross_validation.KFold(len(Xdata),n_folds = num_folds, shuffle = True)
                K_fold_rms_list = []
                Overall_Y_Pred = np.zeros(len(Xdata))
                #split into testing and training sets
                for train_index, test_index in kf:
                    X_train, X_test = Xdata[train_index], Xdata[test_index]
                    Y_train, Y_test = Ydata[train_index], Ydata[test_index]
                    #train on training sets
                    model = RandomForestRegressor(n_estimators=c,
                                                                max_depth=a,
                                                                min_samples_split=b,
                                                                min_samples_leaf=1,
                                                                min_weight_fraction_leaf=0,
                                                                max_features='auto',
                                                                max_leaf_nodes=None,
                                                                random_state = 3,
                                                                n_jobs=-1)
                    #model= tree.DecisionTreeRegressor(max_depth = a, min_samples_split = b, min_samples_leaf = c, random_state = 3)
                    model.fit(X_train, Y_train)
                    Y_test_Pred= model.predict(X_test)
                    rms = np.sqrt(mean_squared_error(Y_test,Y_test_Pred))
                    K_fold_rms_list.append(rms)
                    Overall_Y_Pred[test_index]= Y_test_Pred 
                    
                RMS_List.append(np.mean(K_fold_rms_list))
                
            
        
            avgRMS = np.mean(RMS_List)   
            A=str(a)
            B = str(b)
            C = str(c)
            l.append(A)
            l.append(B)
            l.append(C)
            l.append(avgRMS)
            L.append(l)

with open("rf_gs_5fold.csv",'w') as f: 
    writer = csv.writer(f,lineterminator='\n')
    x= ["max_depth","min_sample_split", "estimators","mean_RMSE"]
    writer.writerow(x)
    for i in L:
        writer.writerow(i)
                                               


f.close() 