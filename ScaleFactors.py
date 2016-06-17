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


def kfold_cv(model, X, Y, num_folds=5, num_runs=200):
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
            model.fit(X_train, Y_train)
            Y_test_Pred = model.predict(X_test)
            rms = np.sqrt(mean_squared_error(Y_test, Y_test_Pred))
            K_fold_rms_list.append(rms)

        RMS_List.append(np.mean(K_fold_rms_list))

    avgRMS = np.mean(RMS_List)
    # print (avgRMS)
    return avgRMS


def alloy_cv(model, X, Y, AlloyList):
    rms_list = []
    for alloy in range(1, 60):
        train_index = []
        test_index = []
        for x in range(len(AlloyList)):
            if AlloyList[x] == alloy:
                test_index.append(x)
            else:
                train_index.append(x)
        if len(train_index) == 0 or len(test_index) == 0: continue  # if alloy doesn't exist(x data is empty), then continue
        # fit model to all alloys except the one to be removed
        model.fit(Xdata[train_index], Ydata[train_index])

        Ypredict = model.predict(Xdata[test_index])

        rms = np.sqrt(mean_squared_error(Ypredict, Ydata[test_index]))
        rms_list.append(rms)

    return np.mean(rms_list)


datapath = "../../DBTT_Data.csv"
X = ["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)", "N( C )", "N(log(fluence)", "N(log(flux)", "N(Temp)"]
Y = "delta sigma"

# get data
data = data_parser.parse(datapath)
data.set_x_features(X)
data.set_y_feature(Y)
Ydata = data.get_y_data().ravel()
Xdata = data.get_x_data()
Alloys = data.get_data("Alloy")

for x in range(len(Xdata)):
    rms_list = []
    for y in np.arange(0, 5.5, .5):
        Xdata[:, x] * y
        model = KernelRidge(alpha=.00518, gamma=.518, kernel='laplacian')
        rms1 = kfold_cv(model, X=Xdata, Y=Ydata, num_folds=5, num_runs=200)
        rms2 = alloy_cv(model, Xdata, Ydata, Alloys)
        rms = rms1 + rms2
        rms_list.append(rms)
    plt.scatter(np.arange(0, 5.5, .5), rms_list)
    plt.xlabel("Scale factor")
    plt.ylabel("5-fold + LO Alloy RMSE")
    plt.title(X[x])
    plt.savefig("../../bardeengraphs/{}.png".format(plt.gca().get_title()), dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()