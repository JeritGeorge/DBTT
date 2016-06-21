import numpy as np
import csv
import data_parser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from matplotlib import cm as cm


def kfold_cv(model,X,Y, num_folds=5, num_runs=200):

    Xdata = X
    Ydata = Y

    RMS_List = []
    for n in range(num_runs):
        kf = cross_validation.KFold(len(Xdata), n_folds=num_folds, shuffle=True)
        K_fold_rms_list = []
        # split into testing and training sets
        for train_index, test_index in kf:
            X_train, X_test = Xdata[train_index], Xdata[test_index]
            Y_train, Y_test = Ydata[train_index], Ydata[test_index]
            # train on training sets
            model = model
            model.fit(X_train, Y_train)
            Y_test_Pred = model.predict(X_test)
            rms = np.sqrt(mean_squared_error(Y_test, Y_test_Pred))
            K_fold_rms_list.append(rms)

        RMS_List.append(np.mean(K_fold_rms_list))

    avgRMS = np.mean(RMS_List)
    #print (avgRMS)
    return avgRMS

def leaveout_cv(model, X, Y, testing_size = .95, num_runs=200):

    Ydata = Y
    Xdata = X

    RMS_List = []
    for n in range(num_runs):
        # split into testing and training sets
        X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(Xdata, Ydata, test_size= testing_size)
        model.fit(X_train, Y_train)
        Y_test_Pred = model.predict(X_test)
        rms = np.sqrt(mean_squared_error(Y_test, Y_test_Pred))
        RMS_List.append(rms)

    avgRMS = np.mean(RMS_List)

    #print (avgRMS)
    return avgRMS

def alloy_cv(model, data):

    rms_list = []
    for alloy in range(1, 60):
        model = model  # creates a new model

        # fit model to all alloys except the one to be removed
        data.remove_all_filters()
        data.add_exclusive_filter("Alloy", '=', alloy)
        model.fit(data.get_x_data(), data.get_y_data().ravel())

        # predict removed alloy
        data.remove_all_filters()
        data.add_inclusive_filter("Alloy", '=', alloy)
        if len(data.get_x_data()) == 0: continue  # if alloy doesn't exist(x data is empty), then continue
        Ypredict = model.predict(data.get_x_data())

        rms = np.sqrt(mean_squared_error(Ypredict, data.get_y_data().ravel()))
        rms_list.append(rms)

    return np.mean(rms_list)

datapath = "../../DBTT_Data.csv"
X=["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)", "N( C )", "N(log(fluence)", "N(log(flux)", "N(Temp)"]
Y="delta sigma"

# get data
data = data_parser.parse(datapath)
data.set_x_features(X)
data.set_y_feature(Y)
Ydata = data.get_y_data().ravel()
Xdata = data.get_x_data()


parameters = {'alpha' : np.logspace(-3,-1,15),
                'gamma': np.logspace(-2,0,15)}
grid_scores = []

for a in parameters['alpha']:
    for y in parameters['gamma']:
        model = KernelRidge(alpha= a, gamma= y, kernel='laplacian')
        rms = leaveout_cv(model, X = Xdata, Y = Ydata, num_runs = 200)
        #rms = kfold_cv(model, X = Xdata, Y = Ydata, num_folds = 2, num_runs=100)
        #rms = alloy_cv(model, data)
        grid_scores.append((a, y, rms))

grid_scores = np.asarray(grid_scores)

with open("grid_scores.csv",'w') as f:
    writer = csv.writer(f, lineterminator = '\n')
    x = ["alpha", "gamma", "rms"]
    writer.writerow(x)
    for i in grid_scores:
        writer.writerow(i)

#Heatmap of RMS scores vs the alpha and gamma
print('hello')
plt.figure(1)
plt.hexbin(np.log10(grid_scores[:,0]), np.log10(grid_scores[:,1]), C = grid_scores[:,2], gridsize=10, cmap=cm.plasma, bins=None)
plt.xlabel('log alpha') 
plt.ylabel('log gamma')
cb = plt.colorbar()
cb.set_label('rms')
plt.savefig("../../{}.png".format("alphagammahex"), dpi=200, bbox_inches='tight')
plt.show() 

