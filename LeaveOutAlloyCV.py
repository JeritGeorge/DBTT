import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error


def loacv(model=KernelRidge(alpha=.00518, coef0=1, degree=3, gamma=.518, kernel='laplacian', kernel_params=None),
          datapath="../../DBTT_Data.csv", savepath='../../{}.png',
          X=["N(Cu)", "N(Ni)", "N(Mn)", "N(P)","N(Si)", "N( C )", "N(log(fluence)", "N(log(flux)", "N(Temp)"],
          Y="delta sigma"):

    data = data_parser.parse(datapath)
    data.set_x_features(X)
    data.set_y_feature(Y)

    rms_list = []
    alloy_list = []

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
        alloy_list.append(alloy)

    print('Mean RMSE: ', np.mean(rms_list))

    # graph rmse vs alloy 
    ax = plt.gca()
    plt.figure(figsize=(10, 4))
    plt.xticks(np.arange(0, max(alloy_list) + 1, 5))
    plt.scatter(alloy_list, rms_list, color='black', s=10)
    plt.plot((0, 59), (0, 0), ls="--", c=".3")
    plt.xlabel('Alloy Number')
    plt.ylabel('RMSE (Mpa)')
    plt.title('Leave out Alloy RMSE')
    plt.figtext(.15, .83, 'Mean RMSE: {:.2f}'.format(np.mean(rms_list)), fontsize=14)
    plt.savefig(savepath.format(plt.gca().get_title()), dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()

loacv()