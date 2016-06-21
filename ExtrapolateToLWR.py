import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error


def lwr(model=KernelRidge(alpha=.00139, gamma=.518, kernel='rbf'),
            datapath="../../DBTT_Data.csv", lwr_datapath = "../../CD_LWR_clean3.csv", savepath='../../{}.png',
            X=["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)", "N( C )", "N(log(fluence)", "N(log(flux)", "N(Temp)"],
            Y=" CD ∆σ"):

    data = data_parser.parse(datapath)
    data.set_x_features(X)
    data.add_exclusive_filter("Alloy", '=', 1)
    data.add_exclusive_filter("Alloy", '=', 2)
    data.add_exclusive_filter("Alloy", '=', 8)
    data.set_y_feature(Y)
    #cdY = (data.get_y_data().ravel() - np.mean(data.get_y_data().ravel()))/np.std(data.get_y_data().ravel())
    Y = data.get_y_data().ravel()
    model.fit(data.get_x_data(), data.get_y_data().ravel())

    lwr_data = data_parser.parse(lwr_datapath)
    lwr_data.set_y_feature("CD predicted delta sigma (Mpa)")
    lwr_data.set_x_features(X)

    XLWR = lwr_data.get_x_data()
    YLWR = lwr_data.get_y_data()
    #Ypredict = model.predict(lwr_data.get_x_data())*np.std(data.get_y_data().ravel()) + np.mean(data.get_y_data().ravel())
    Ypredict = model.predict(lwr_data.get_x_data())
    rms = np.sqrt(mean_squared_error(Ypredict, lwr_data.get_y_data().ravel()))
    print("RMS: ", rms)

    plt.figure(1)
    plt.scatter(lwr_data.get_y_data(), Ypredict, s=10, color='black', label='IVAR')
    #plt.scatter(data.get_y_data().ravel(), model.predict(data.get_x_data()), s = 10, color = 'red')
    plt.plot(plt.gca().get_ylim(), plt.gca().get_ylim(), ls="--", c=".3")
    plt.xlabel('CD Predicted (MPa)')
    plt.ylabel('Model Predicted (MPa)')
    plt.title('Extrapolate to LWR')
    plt.figtext(.15, .83, 'RMS: %.4f' % (rms), fontsize=14)
    plt.savefig(savepath.format(plt.gca().get_title()), dpi=200, bbox_inches='tight')

lwr()
