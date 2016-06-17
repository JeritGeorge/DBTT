import numpy as np
import data_parser
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge


def cv(model, datapath = "../../DBTT_Data.csv", num_folds=5, num_runs=200,
       X=["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)", "N( C )", "N(log(fluence)", "N(log(flux)", "N(Temp)"],
       Y="delta sigma"):

    # get data
    data = data_parser.parse(datapath)
    data.set_x_features(X)
    data.set_y_feature(Y)
    Ydata = data.get_y_data().ravel()
    Xdata = data.get_x_data()

    RMS_List = []
    for n in range(num_runs):
        kf = cross_validation.KFold(len(Xdata), n_folds=num_folds, shuffle=True)
        K_fold_rms_list = []
        Overall_Y_Pred = np.zeros(len(Xdata))
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
            Overall_Y_Pred[test_index] = Y_test_Pred

        RMS_List.append(np.mean(K_fold_rms_list))

    avgRMS = np.mean(RMS_List)
    print (avgRMS)
    return avgRMS

degrees = range(2,7)
RMS_List = []
model = KernelRidge(alpha= 0.00139, gamma=0.518, kernel='polynomial', degree = 4)
cv(model)
for degree in degrees:
    model = KernelRidge(alpha= 0.00139, gamma=0.518, kernel='polynomial', degree = degree)
    RMS_List.append(cv(model))

plt.scatter(degrees, RMS_List)
plt.xlabel("degree of polynomial kernel")
plt.ylabel("200x 2-fold Mean RMSE")
plt.savefig("../../polynomial_kernel_degree.png", dpi=200, bbox_inches='tight')
plt.show()
