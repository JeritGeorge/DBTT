import matplotlib
import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error


def loalwr(model=KernelRidge(alpha=.00518, coef0=1, degree=3, gamma=.518, kernel='laplacian', kernel_params=None),
          datapath="../../DBTT_Data.csv", lwr_datapath = "../../CD_LWR_clean.csv", savepath='../../{}.png',
          X=["N(Cu)", "N(Ni)", "N(Mn)", "N(P)","N(Si)", "N( C )", "N(log(fluence)", "N(log(flux)", "N(Temp)"],
          Y="delta sigma"):

    data = data_parser.parse(datapath)
    data.set_x_features(X)
    data.set_y_feature(Y)

    lwr_data = data_parser.parse(lwr_datapath)
    lwr_data.set_x_features(X)
    lwr_data.set_y_feature(Y)

    rms_list = []
    alloy_list = []

    for alloy in range(1, 60):
        model = model  # creates a new model

        # fit model to all alloys except the one to be removed
        data.remove_all_filters()
        data.add_exclusive_filter("Alloy", '=', alloy)
        model.fit(data.get_x_data(), np.asarray(data.get_y_data()).ravel())

        # predict removed alloy
        lwr_data.remove_all_filters()
        lwr_data.add_inclusive_filter("Alloy", '=', alloy)
        if len(lwr_data.get_x_data()) == 0: continue  # if alloy doesn't exist(x data is empty), then continue
        Ypredict = model.predict(lwr_data.get_x_data())

        rms = np.sqrt(mean_squared_error(Ypredict, np.asarray(lwr_data.get_y_data()).ravel()))
        rms_list.append(rms)
        alloy_list.append(alloy)

    print('Mean RMSE: ', np.mean(rms_list))


    # graph rmse vs alloy
    matplotlib.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.xticks(np.arange(0, max(alloy_list) + 1, 5))
    ax.scatter(alloy_list, rms_list, color='black', s=10)
    ax.plot((0, 59), (0, 0), ls="--", c=".3")
    ax.set_xlabel('Alloy Number')
    ax.set_ylabel('RMSE (Mpa)')
    ax.set_title('Leave out Alloy LWR')
    ax.text(.05, .88, 'Mean RMSE: {:.2f}'.format(np.mean(rms_list)), fontsize=14, transform=ax.transAxes)
    for x in np.argsort(rms_list)[-5:]:
        ax.annotate(s = alloy_list[x],xy = (alloy_list[x], rms_list[x]))
    fig.savefig(savepath.format(ax.get_title()), dpi=200, bbox_inches='tight')
    fig.clf()
    plt.close()