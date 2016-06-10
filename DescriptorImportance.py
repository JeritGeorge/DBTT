import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation

def getModel():
    return KernelRidge(alpha= .00139, coef0=1, degree=3, gamma=.518, kernel='rbf', kernel_params=None)

data = data_parser.parse("../DBTT_data.csv")
data.set_x_features(["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)","N( C )", "N(log(fluence)", "N(log(flux)", "N(Temp)"])
data.set_y_feature("delta sigma")

overall_rms_list = []
sd_list = []
descriptorList = ['Cu','Ni','Mn','P', 'Si', 'C', 'Fl', 'Fx', 'Temp']

numFolds = 5
numIter = 200
model = getModel()

Xdata = np.asarray(data.get_x_data())
Ydata = np.asarray(data.get_y_data())

print ("Testing descriptor importance using {}x {} - Fold CV".format(numIter,numFolds)) 
print ("")

for x in range(len(data.get_x_data()[0])): 
    RMS_List = []
    newX = np.delete(Xdata, x, 1)
    for n in range(numIter):
        kf = cross_validation.KFold(len(Xdata),n_folds = numFolds, shuffle = True)
        K_fold_rms_list = [];
        #split into testing and training sets
        for train_index, test_index in kf:
            X_train, X_test = newX[train_index], newX[test_index]
            Y_train, Y_test = Ydata[train_index], Ydata[test_index]
            #train on training sets
            model.fit(X_train, Y_train)
            YTP = model.predict(X_test)
            rms = np.sqrt(mean_squared_error(Y_test,YTP))
            K_fold_rms_list.append(rms)
        RMS_List.append(np.mean(K_fold_rms_list))    
        #calculate rms
    
    maxRMS = np.amax(RMS_List)
    minRMS = np.amin(RMS_List)
    avgRMS = np.mean(RMS_List)
    medRMS = np.median(RMS_List)
    sd = np.sqrt(np.mean((RMS_List - np.mean(RMS_List)) ** 2))
    
    print ("Removing {}:".format(descriptorList[x]))
    print ("The average RMSE was " + str(avgRMS))
    print ("The median RMSE was " + str(medRMS))
    print ("The max RMSE was " + str(maxRMS))
    print ("The min RMSE was " + str(minRMS))
    print ("The std deviation of the RMSE values was " + str(sd))
    print ("")

    overall_rms_list.append(avgRMS)
    sd_list.append(sd)

ax = plt.gca()
width = .8
rects = ax.bar(np.arange(9), overall_rms_list, width, color = 'red', yerr = sd_list)
ax.set_xlabel('Descriptor Removed')
ax.set_ylabel('200x 5-fold RMSE')
ax.set_ylim(top = 1.2*max(overall_rms_list))
ax.set_title('Descriptor Importance')
ax.set_xticks(np.arange(9) + .5*width)
ax.set_xticklabels(descriptorList)
plt.savefig('../%s.png' %(plt.gca().get_title()), dpi = 200, bbox_inches='tight')

#attach text with RMSE
for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, 
                '%.1f' % (height),
                ha='center', va='bottom', fontsize = 12)
                
plt.show()


    
