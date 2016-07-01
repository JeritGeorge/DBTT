import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error


def lwr(model=KernelRidge(alpha=.00139, gamma=.518, kernel='laplacian'),
            datapath="../../DBTT_Data.csv", lwr_datapath = "../../CD_LWR_clean.csv", savepath='../../{}.png',
            X=["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)", "N( C )", "N(log(fluence)", "N(Temp)","N(log(flux)"],
            Y=" CD delta sigma"):

    data = data_parser.parse(datapath)
    data.set_x_features(X)
    data.set_y_feature(Y)

    trainX = np.asarray(data.get_x_data())
    trainY = np.asarray(data.get_y_data()).ravel()

    lwr_data = data_parser.parse(lwr_datapath)
    lwr_data.set_y_feature(Y)
    lwr_data.set_x_features(X)
    
    testX = np.asarray(lwr_data.get_x_data())
    
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

