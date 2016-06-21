import KFold_CV
import LeaveOutAlloyCV
import FullFit
import FluenceFluxExtrapolation
import DescriptorImportance
import ErrorBias

# things to change before running the codes

# model to use
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from pyearth import Earth
model = KernelRidge(alpha= .00193, gamma = .518, kernel='laplacian')
#model = Earth()
#model = LinearRegression()
#model = SVR(verbose=False, C=400, gamma=5)

# file paths
datapath = "../../DBTT_Data.csv"  # path to your data
savepath = "../../graphs/{}.png"  # where you want output graphs to be saved

#data
Ydata = "delta sigma"

print("K-Fold CV:")
KFold_CV.cv(model, datapath, savepath, Y = Ydata)  # also has parameters num_folds (default is 5) and num_runs (default is 200)

print("\nLeave out alloy CV:")
LeaveOutAlloyCV.loacv(model, datapath, savepath, Y = Ydata)

print("\nFull Fit:")
FullFit.fullfit(model, datapath, savepath, Y = Ydata)

print("\nFluence and Flux Extrapolation:")
FluenceFluxExtrapolation.flfxex(model, datapath, savepath, Y = Ydata)

print("\nError Bias:")
ErrorBias.errbias(model, datapath, savepath, Y = Ydata)


print("\nDescriptor Importance:")
DescriptorImportance.desimp(model, datapath, savepath, Y = Ydata)