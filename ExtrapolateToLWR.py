import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error


def lwr(model=KernelRidge(alpha=.00139, gamma=.518, kernel='laplacian'),
            datapath="../../DBTT_Data.csv", lwr_datapath = "../../CD_LWR_clean.csv", savepath='../../{}.png',
            X=["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)", "N( C )", "N(log(fluence)", "N(Temp)","N(log(flux)"],
            Y=" CD ∆σ"):

    data = data_parser.parse(datapath)
    data.set_x_features(X)
    data.set_y_feature(Y)

    trainX = np.asarray(data.get_x_data())
    trainY = np.asarray(data.get_y_data()).ravel()

    '''
    for x in range(len(trainX[:, 0])):
        if trainX[x, 0] > .25:
            trainX[x, 0] = .25'''



    lwr_data = data_parser.parse(lwr_datapath)
    lwr_data.set_y_feature("CD predicted delta sigma (Mpa)")
    lwr_data.set_x_features(X)
    #lwr_data.add_exclusive_filter('Fluence(n/cm^2)', '<=', 1e18)
    #lwr_data.add_exclusive_filter('Si (At%)', '<', .1)
    testX = np.asarray(lwr_data.get_x_data())
    '''
    for x in range(len(testX[:,0])):
        if testX[x,0] > .25:
            testX[x,0] = .25'''

    model.fit(trainX, trainY)
    Ypredict = model.predict(testX)
    rms = np.sqrt(mean_squared_error(Ypredict, lwr_data.get_y_data()))
    print("RMS: ", rms)

    plt.figure(1)
    plt.scatter(lwr_data.get_y_data(), Ypredict, s=10, color='black', label='IVAR')
    #plt.scatter(data.get_y_data().ravel(), model.predict(data.get_x_data()), s = 10, color = 'red')
    plt.plot(plt.gca().get_ylim(), plt.gca().get_ylim(), ls="--", c=".3")
    plt.xlabel('CD Predicted (MPa)')
    plt.ylabel('Model Predicted (MPa)')
    plt.title('Extrapolate to LWR')
    plt.figtext(.15, .83, 'RMS: %.4f' % (rms), fontsize=14)
    plt.show()
    plt.savefig(savepath.format(plt.gca().get_title()), dpi=200, bbox_inches='tight')

from sklearn.ensemble import RandomForestRegressor
lwr(model = RandomForestRegressor(n_estimators = 50, max_depth=10, min_samples_leaf=1, min_samples_split=3))
#lwr()
