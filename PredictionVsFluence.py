import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error


def execute(model, data, savepath):
    Xdata = np.asarray(data.get_x_data())
    Ydata = np.asarray(data.get_y_data()).ravel()

    model.fit(Xdata, Ydata)

    Temp = 290 * np.ones((500,1))
    Flux = 3e10 * np.ones((500,1))
    Time = np.reshape(np.linspace(3.154e+7/2, 3.154e+7 * 100, 500),(500,1))
    aFluence = (Flux * Time)
    #aFluence = np.reshape(np.logspace(16,21,100),(100,1))
    #normalize
    Temp = (Temp - np.min(np.asarray(data.get_data("Temp (C)"))))/(np.max(np.asarray(data.get_data("Temp (C)"))) - np.min(np.asarray(data.get_data("Temp (C)"))))
    Flux = (np.log10(Flux) - np.min(np.asarray(data.get_data("log(flux)"))))/(np.max(np.asarray(data.get_data("log(flux)"))) - np.min(np.asarray(data.get_data("log(flux)"))))
    Fluence  = (np.log10(aFluence) - np.min(np.asarray(data.get_data("log(fluence)"))))/(np.max(np.asarray(data.get_data("log(fluence)"))) - np.min(np.asarray(data.get_data("log(fluence)"))))

    for alloy in range(1, 60):
        data.remove_all_filters()
        data.add_inclusive_filter("Alloy", '=', alloy)

        if len(data.get_x_data()) == 0: continue  # if alloy doesn't exist(x data is empty), then continue

        Alloy = np.reshape(np.asarray(data.get_x_data())[0,0:6],(1,6)) * np.ones((500, 6))

        Xdata = np.concatenate([Alloy, Fluence, Flux, Temp], 1)

        ypredict = model.predict(Xdata)


        fig, ax = plt.subplots()
        ax.plot(np.log10(aFluence), ypredict)
        ax.set_title(alloy)
        ax.set_xlabel("log(Fluence(n/cm^2))")
        ax.set_ylabel("predicted ∆sigma (MPa)")
        fig.savefig(savepath.format(ax.get_title()), dpi=200, bbox_inches='tight')
        plt.close()


from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

#model = KernelRidge(alpha=.00139, gamma=.518, kernel='laplacian')
model = SVR(C = 400, gamma = 5)
#model = RandomForestRegressor(n_estimators = 100, max_depth= 10, min_samples_split=1, min_samples_leaf=1)
X = ["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)", "N( C )", "N(log(fluence)", "N(log(flux)", "N(Temp)"]
Y = "delta sigma"
datapath = "../../DBTT_Data.csv"
savepath = '../../graphs/fluencegraphs/{}.png'
data = data_parser.parse(datapath)
data.set_x_features(X)
data.set_y_feature(Y)

execute(model, data, savepath)
