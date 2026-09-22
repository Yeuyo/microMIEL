import sys, os, re, math, time, warnings, copy
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
from scipy.spatial import distance
# from sklearn.manifold import TSNE
# from umap import UMAP #umap-learn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from scipy.stats import spearmanr
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
# import seaborn as sns
import plotly.graph_objects as go
import multiprocessing as mp
from dfply import X, group_by, summarize, summary_functions
import plotly.io as pio

pio.renderers.default = "browser"

from epilands.generic_read_write import (
    read_dataframe_from_h5_file
)
from epilands.image_read_write import (
    read_segmentation_data
)
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

runParallel = True
condenseCells = False
condenseNumber = 20
removeLeft = False
removeLeftBoundary = -5000
histPlotRange = 20000 #np.inf
histRemove = True
# histRange = [3000, 10500] #KSHV vs LEC
# histRange = [4000, 10500] #HUVEC DMSO
# histRange = [1250, 3250] # Shear Stress no drug
# histRange = [500, 3000] # Shear Stress drug
# histRange = [0, 2000] # Shear Stress drug
# histRange = [5000, 12000] #HUVEC sen
# histRange = [3000, 7000] #HUVEC
# histRange = [5000, 9500] #HUVEC DMSO
# histRange = [1000, 4500] #HUVEC DMSO
# histRange = [3000, 9500] #Hela Susav
histRange = [0, 100] #Hela Susav
n_bootstraps = 1
realBS = False
optimiseCondense = False
# condenseCellsTry = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
# condenseCellsTry = [ 1,2,3,4,5,6,7,8,9,10, 20,30,40,50, 100]
# condenseCellsTry = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
# condenseCellsTry = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,110,120,130, 140,150]
condenseCellsTry = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200]
# condenseCellsTry = [50, 60]
# condenseCellsTry = [ 1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,125,150,175,200 ]
# condenseCellsTry = [9, 8, 7, 6, 5, 4, 3, 2, 1]
optimiseBootstrap = False
bootstrapsTry = [1,2,3,4,5]
project_name = "microMIEL-NoSox"
saveFormat = ".svg"
# saveDir = "C:/Users/yeww/Downloads/LEC_DMSO vs KLEC_DMSO/"
# saveDir = "/media/gic/b053d09e-e3ac-4356-848a-0ef98d9db14e/Susav-Miel-data/TIFF/Result/Visual/"/media/gic/b053d09e-e3ac-4356-848a-0ef98d9db14e/Susav-Miel-data/TIFF/Result/Visual/
# saveDir = r"//shared.sydney.edu.au/research-data/PRJ-DataT/Bitong's data/Bitong HiLo 405/Cell 1/Result/Plots/"
saveDir = r"//shared.sydney.edu.au/research-data/PRJ-DataT/Bitong's data/Bitong HiLo 405/Data for HILO405 analysis/Result/Plots/"
n_test = 10 # number of test to run to check for confidence
fontSize = 20
bin_number_for_plot = 200

manualOrder = False
# manualOrderList = ["0hr", "0hr GsMT4x", "6hr", "6hr GsMT4x"]
# condList = ["6hr","6hr_C1", "0hr_C1","60min1", "60min2", "60min3"]
# groupList = ["6.0hr", "6.0hr", "0.0hrs", "1.0hr", "1.0hr","1.0hr"]

folDir = r"//shared.sydney.edu.au/research-data/PRJ-DataT/Bitong's data/Bitong HiLo 405/Data for HILO405 analysis/Result" #AA82-AA90
condList = os.listdir(folDir)
condList = condList[:-1]
groupList = copy.deepcopy(condList)
# condList = ["C317_Pre", "C317_Post"]
# groupList = ["C317_Pre", "C317_Post"]
# condList = ["C2T1_Pre", "C2T1_Post"]
# groupList = ["C2T1_Pre", "C2T1_Post"]
# condList = ["C317T4_Pre", "C317T4_Post", "C317T4_Others", "C317T5_Pre", "C317T5_Post", "C317T5_Others",
#                "C617T4_Pre", "C617T4_Post", "C617T4_Others", "C617T5_Pre", "C617T5_Post", "C617T5_Others"]
# groupList = ["C317T4_Pre", "C317T4_Post", "C317T4_Others", "C317T5_Pre", "C317T5_Post", "C317T5_Others",
#                "C617T4_Pre", "C617T4_Post", "C617T4_Others", "C617T5_Pre", "C617T5_Post", "C617T5_Others"]

# condList = ["C317T4_Pre", "C317T4_Post"]
# groupList = ["C317T4_Pre", "C317T4_Post"]
# condList = ["C317T5_Pre", "C317T5_Post"]
# groupList = ["C317T5_Pre", "C317T5_Post"]
# condList = ["C617T4_Pre", "C617T4_Post"]
# groupList = ["C617T4_Pre", "C617T4_Post"]
# condList = ["C617T5_Pre", "C617T5_Post"]
# groupList = ["C617T5_Pre", "C617T5_Post"]

# condList = ["C612T6_Pre", "C612T6_Post"]
# groupList = ["C612T6_Pre", "C612T6_Post"]
# condList = ["C317T6_Pre", "C317T6_Post"]
# groupList = ["C317T6_Pre", "C317T6_Post"]
condList = ["C617T6_Pre", "C617T6_Post"]
groupList = ["C617T6_Pre", "C617T6_Post"]

condList = ["C112_Pre", "C112_Post", "C112_Others",
            "C117_Pre", "C117_Post", "C117_Others",
            "C212_Pre", "C212_Post", "C212_Others",
            "C217_Pre", "C217_Post", "C217_Others",
            "C312_Pre", "C312_Post", "C312_Others",
            "C412_Pre", "C412_Post", "C412_Others",
            "C417_Pre", "C417_Post", "C417_Others",
            "C512_Pre", "C512_Post", "C512_Others",
            "C517_Pre", "C517_Post", "C517_Others",
            "C610_Pre", "C610_Post", "C610_Others",
            "C910_Pre", "C910_Post", "C910_Others",
            "C617T6_Pre", "C617T6_Post", "C617T6_Others",
            "C224T1_Pre", "C224T1_Post", "C224T1_Others",
            "C224T2_Pre", "C224T2_Post", "C224T2_Others",
            "C719T1_Pre", "C719T1_Post", "C719T1_Others",
            "C719T2_Pre", "C719T2_Post", "C719T2_Others",
            "C401T1_Pre", "C401T1_Post", "C401T1_Others",
            "C401T2_Pre", "C401T2_Post", "C401T2_Others",
            "C101_Pre", "C101_Post", "C101_Others",
            "C215_Pre", "C215_Post", "C215_Others",
            "C319_Pre", "C319_Post", "C319_Others",
            "C315_Pre", "C315_Post", "C315_Others",
            "C424_Pre", "C424_Post", "C424_Others",
            "C415_Pre", "C415_Post", "C415_Others",
            "C501_Pre", "C501_Post", "C501_Others",
            "C515_Pre", "C515_Post", "C515_Others",
            "C615_Pre", "C615_Post", "C615_Others",
            "C724_Pre", "C724_Post", "C724_Others",
            "C715_Pre", "C715_Post", "C715_Others",
            "C815_Pre", "C815_Post", "C815_Others",
            "C915_Pre", "C915_Post", "C915_Others",
            "C1015_Pre", "C1015_Post", "C1015_Others",
            "C617T6_Pre", "C617T6_Post", "C617T6_Others",
            "C317T4_Pre", "C317T4_Post", "C317T4_Others",
            "C317T5_Pre", "C317T5_Post", "C317T5_Others"]
            # "C2T1_Pre", "C2T1_Post", "C2T1_Others"]
# groupList = ["No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "C617T6_Pre", "C617T6_Post", "C617T6_Others",
#              "C224T1_Pre", "C224T1_Post", "C224T1_Others",
#              "C224T2_Pre", "C224T2_Post", "C224T2_Others",
#              "C719T1_Pre", "C719T1_Post", "C719T1_Others",
#              "C719T2_Pre", "C719T2_Post", "C719T2_Others",
#              "C401T1_Pre", "C401T1_Post", "C401T1_Others",
#              "C401T2_Pre", "C401T2_Post", "C401T2_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "No Sox_Pre", "No Sox_Post", "No Sox_Others",
#              "C617T6_Pre", "C617T6_Post", "C617T6_Others",
#              "C317T4_Pre", "C317T4_Post", "C317T4_Others",
#              "C317T5_Pre", "C317T5_Post", "C317T5_Others",
#              "C2T1_Pre", "C2T1_Post", "C2T1_Others"]
groupList = ["No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "Sox_Pre", "Sox_Post", "Sox_Others",
             "Sox_Pre", "Sox_Post", "Sox_Others",
             "Sox_Pre", "Sox_Post", "Sox_Others",
             "Sox_Pre", "Sox_Post", "Sox_Others",
             "Sox_Pre", "Sox_Post", "Sox_Others",
             "Sox_Pre", "Sox_Post", "Sox_Others",
             "Sox_Pre", "Sox_Post", "Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "No Sox_Pre", "No Sox_Post", "No Sox_Others",
             "Sox_Pre", "Sox_Post", "Sox_Others",
             "Sox_Pre", "Sox_Post", "Sox_Others",
             "Sox_Pre", "Sox_Post", "Sox_Others"]

# No Others
condList = ["C112_Pre", "C112_Post",
            "C117_Pre", "C117_Post",
            "C212_Pre", "C212_Post",
            "C217_Pre", "C217_Post",
            "C312_Pre", "C312_Post",
            "C412_Pre", "C412_Post",
            "C417_Pre", "C417_Post",
            "C512_Pre", "C512_Post",
            "C517_Pre", "C517_Post",
            "C610_Pre", "C610_Post",
            "C910_Pre", "C910_Post",
            "C617T6_Pre", "C617T6_Post",
            "C224T1_Pre", "C224T1_Post",
            "C224T2_Pre", "C224T2_Post",
            "C719T1_Pre", "C719T1_Post",
            "C719T2_Pre", "C719T2_Post",
            "C401T1_Pre", "C401T1_Post",
            "C401T2_Pre", "C401T2_Post",
            "C101_Pre", "C101_Post",
            "C215_Pre", "C215_Post",
            "C319_Pre", "C319_Post",
            "C315_Pre", "C315_Post",
            "C424_Pre", "C424_Post",
            "C415_Pre", "C415_Post",
            "C501_Pre", "C501_Post",
            "C515_Pre", "C515_Post",
            "C615_Pre", "C615_Post",
            "C724_Pre", "C724_Post",
            "C715_Pre", "C715_Post",
            "C815_Pre", "C815_Post",
            "C915_Pre", "C915_Post",
            "C1015_Pre", "C1015_Post"]
            #"C617T6_Pre", "C617T6_Post",
            # "C317T4_Pre", "C317T4_Post",
            # "C317T5_Pre", "C317T5_Post"]
groupList = ["No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "Sox_Pre", "Sox_Post",
             "Sox_Pre", "Sox_Post",
             "Sox_Pre", "Sox_Post",
             "Sox_Pre", "Sox_Post",
             "Sox_Pre", "Sox_Post",
             "Sox_Pre", "Sox_Post",
             "Sox_Pre", "Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post",
             "No Sox_Pre", "No Sox_Post"]
             #"Sox_Pre", "Sox_Post",
            #  "Sox_Pre", "Sox_Post",
            #  "Sox_Pre", "Sox_Post"]

# Sox
# condList = ["C617T6_Pre", "C617T6_Post",
#             "C224T1_Pre", "C224T1_Post",
#             "C224T2_Pre", "C224T2_Post",
#             "C719T1_Pre", "C719T1_Post",
#             "C719T2_Pre", "C719T2_Post",
#             "C401T1_Pre", "C401T1_Post",
#             "C401T2_Pre", "C401T2_Post",
#             #"C617T6_Pre", "C617T6_Post",
#             "C317T4_Pre", "C317T4_Post",
#             "C317T5_Pre", "C317T5_Post"]
# groupList = ["Sox_Pre", "Sox_Post",
#              "Sox_Pre", "Sox_Post",
#              "Sox_Pre", "Sox_Post",
#              "Sox_Pre", "Sox_Post",
#              "Sox_Pre", "Sox_Post",
#              "Sox_Pre", "Sox_Post",
#              "Sox_Pre", "Sox_Post",
#              #"Sox_Pre", "Sox_Post",
#              "Sox_Pre", "Sox_Post",
#              "Sox_Pre", "Sox_Post"]
# groupList = ["Sox_Pre1", "Sox_Post1",
#              "Sox_Pre2", "Sox_Post2",
#              "Sox_Pre3", "Sox_Post3",
#              "Sox_Pre4", "Sox_Post4",
#              "Sox_Pre5", "Sox_Post5",
#              "Sox_Pre6", "Sox_Post6",
#              "Sox_Pre7", "Sox_Post7",
#              #"Sox_Pre8", "Sox_Post8",
#              "Sox_Pre9", "Sox_Post9",
#              "Sox_Pre10", "Sox_Post10"]

# No Sox
# condList = ["C112_Pre", "C112_Post",
#             "C117_Pre", "C117_Post",
#             "C212_Pre", "C212_Post",
#             "C217_Pre", "C217_Post",
#             "C312_Pre", "C312_Post",
#             "C412_Pre", "C412_Post",
#             "C417_Pre", "C417_Post",
#             "C512_Pre", "C512_Post",
#             "C517_Pre", "C517_Post",
#             "C610_Pre", "C610_Post",
#             "C910_Pre", "C910_Post",
#             "C101_Pre", "C101_Post",
#             "C215_Pre", "C215_Post",
#             "C319_Pre", "C319_Post",
#             "C315_Pre", "C315_Post",
#             "C424_Pre", "C424_Post",
#             "C415_Pre", "C415_Post",
#             "C501_Pre", "C501_Post",
#             "C515_Pre", "C515_Post",
#             "C615_Pre", "C615_Post",
#             "C724_Pre", "C724_Post",
#             "C715_Pre", "C715_Post",
#             "C815_Pre", "C815_Post",
#             "C915_Pre", "C915_Post",
#             "C1015_Pre", "C1015_Post"]
# groupList = ["No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post",
#              "No Sox_Pre", "No Sox_Post"]
# groupList = ["No Sox_Pre1", "No Sox_Post1",
#              "No Sox_Pre2", "No Sox_Post2",
#              "No Sox_Pre3", "No Sox_Post3",
#              "No Sox_Pre4", "No Sox_Post4",
#              "No Sox_Pre5", "No Sox_Post5",
#              "No Sox_Pre6", "No Sox_Post6",
#              "No Sox_Pre7", "No Sox_Post7",
#              "No Sox_Pre8", "No Sox_Post8",
#              "No Sox_Pre9", "No Sox_Post9",
#              "No Sox_Pre10", "No Sox_Post10",
#              "No Sox_Pre11", "No Sox_Post11",
#              "No Sox_Pre12", "No Sox_Post12",
#              "No Sox_Pre13", "No Sox_Post13",
#              "No Sox_Pre14", "No Sox_Post14",
#              "No Sox_Pre15", "No Sox_Post15",
#              "No Sox_Pre16", "No Sox_Post16",
#              "No Sox_Pre17", "No Sox_Post17",
#              "No Sox_Pre18", "No Sox_Post18",
#              "No Sox_Pre19", "No Sox_Post19",
#              "No Sox_Pre20", "No Sox_Post20",
#              "No Sox_Pre21", "No Sox_Post21",
#              "No Sox_Pre22", "No Sox_Post22",
#              "No Sox_Pre23", "No Sox_Post23",
#              "No Sox_Pre24", "No Sox_Post24",
#              "No Sox_Pre25", "No Sox_Post25"]
# bigList = ["C112_Pre", "C112_Post", "C112_Others",
#            "C117_Pre", "C117_Post", "C117_Others",
#            "C212_Pre", "C212_Post", "C212_Others",
#            "C217_Pre", "C217_Post", "C217_Others",
#            "C312_Pre", "C312_Post", "C312_Others",
#            "C412_Pre", "C412_Post", "C412_Others",
#            "C417_Pre", "C417_Post", "C417_Others",
#            "C512_Pre", "C512_Post", "C512_Others",
#            "C517_Pre", "C517_Post", "C517_Others",
#            "C610_Pre", "C610_Post", "C610_Others",
#            "C910_Pre", "C910_Post", "C910_Others",
#            "C617T6_Pre", "C617T6_Post", "C617T6_Others",
#            "C224T1_Pre", "C224T1_Post", "C224T1_Others",
#            "C224T2_Pre", "C224T2_Post", "C224T2_Others",
#            "C719T1_Pre", "C719T1_Post", "C719T1_Others",
#            "C719T2_Pre", "C719T2_Post", "C719T2_Others"]

# condList = bigList[6:8]
# groupList = bigList[6:8]
# condList = ["Pre", "Post", "Others"]
# groupList = ["Pre", "Post", "Others"]


# 
# channelViewList, maxChannels = ["0", "1", "ALL"], "1"
channelViewList, maxChannels = ["0"], "0"
# channelViewList, maxChannels = ["0", "1", "ALL"], "1"
# channelViewList, maxChannels = ["0", "2", "ALL"], "2"
# channelViewList, maxChannels = ["ALL"], "2"
# channelViewList, maxChannels = ["0", "2"], "2"
# channelViewList, maxChannels = ["0", "1", "2", "3", "ALL"], "3"

## DO NOT MODIFY BELOW THIS LINE
os.makedirs(saveDir, exist_ok=True)
if runParallel == True:
    import multiprocessing as mp

def sig_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''
    
def plot_decision_boundary(X, y, model, scaler, finalData, PCA_ratio):
        x_min, x_max = X['PCA1'].min() - 1, X['PCA1'].max() + 1
        y_min, y_max = X['PCA2'].min() - 1, X['PCA2'].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, math.ceil((x_max-x_min)/1000)),
                             np.arange(y_min, y_max, math.ceil((y_max-y_min)/1000)))

        mesh_points = scaler.transform(np.c_[xx.ravel(), yy.ravel()])

        Z = model.predict(mesh_points)
        Z = Z.reshape(xx.shape)

        for i in range(len(Z)):
            for j in range(len(Z[i])):
                Z[i][j] = groupList.index(Z[i][j])

        xx = np.array(xx, dtype=float)
        yy = np.array(yy, dtype=float)
        Z = np.array(Z, dtype=float)

        plotTitle = "Decision Boundary"
        if condenseCells == True:
            plotTitle = plotTitle + " condense " + str(condenseNumber)
        if bootstrap == True:
            plotTitle = plotTitle + " bootstrap " + str(n_bootstraps)
        if histRemove == True:
            plotTitle = plotTitle + " size " + str(histRange[0]) + "-" + str(histRange[1])
        fig = px.scatter(finalData, title = plotTitle, x = 'PCA1', y = 'PCA2', color = "Cell", labels = {"PCA1": "PCA1 " + f"({PCA_ratio[0]*100:.2f}%)", "PCA2": "PCA2 " + f"({PCA_ratio[1]*100:.2f}%)"})
        fig = go.Figure(data = go.Contour(x = np.arange(x_min, x_max, math.ceil((x_max-x_min)/1000)), y = np.arange(y_min, y_max, math.ceil((y_max-y_min)/1000)), z=Z, colorscale = px.colors.qualitative.Dark24, showscale = False))
        traceList = finalData['Cell'].unique().tolist()
        for i in range(len(traceList)):
            fig.add_scatter(x = finalData.loc[finalData['Cell'] == traceList[i], 'PCA1'], y = finalData.loc[finalData['Cell'] == traceList[i], 'PCA2'], mode = 'markers', marker = dict(color = px.colors.qualitative.Dark24[i], size = 8), name = traceList[i]) #legendgroup = groupList[i]
        fig.update_layout(width = 1200, height = 800)
        fig.show()
        fig.write_image(saveDir + project_name + "_DB" + saveFormat)

def bootstrapParrallel(arg):
    dataDir, condenseCells, condenseNumber, folDir, condList, groupList, channelViewList, maxChannels = arg
    dataDir = dataDir.sample(frac = 0.8, replace = True, random_state = np.random.randint(0, 1000)).reset_index(drop=True)
    pca_analysis = PCA(n_components=4)
    dataforPCA = dataDir.drop(["Path", "Cell"], axis = 1)

    # z scoring
    scaler = StandardScaler()
    zscoreData = scaler.fit_transform(dataforPCA.iloc[:, 7:].values)
    principalComponents = pca_analysis.fit_transform(zscoreData)
    finalData = pd.DataFrame(data = principalComponents, columns = ['PCA1', 'PCA2', 'PCA3', 'PCA4'])
    finalData["Cell"] = dataDir["Cell"]

    # ctr = principalComponents**2 / np.sum(principalComponents**2, axis = 0)
    dataDir = finalData.copy()

    dataConcat = pd.DataFrame()
    if condenseCells == True:
        cellType = dataDir["Cell"].unique().tolist()
        for n in range(len(cellType)):
            dataType = dataDir.loc[dataDir["Cell"] == cellType[n], ]
            dataType.reset_index(drop = True, inplace = True)
            while len(dataType) > condenseNumber:
                randInd = dataType.sample(n = condenseNumber, random_state = 1, replace = False).index.tolist()
                condense = dataType.iloc[randInd, :-2].mean()
                condense["Path"] = dataType.iloc[randInd[0], -2]
                condense["Cell"] = dataType.iloc[randInd[0], -1]
                dataConcat = pd.concat([dataConcat, condense.to_frame().transpose()], ignore_index = True)
                dataType.drop(randInd, inplace = True)
                dataType.reset_index(drop = True, inplace = True)
    else:
        dataConcat = dataDir.copy()

    finalData = dataConcat.copy()
    # fig = px.scatter(finalData, title = str(condenseNumber), x = 'PCA1', y = 'PCA2', color = "Cell", labels = {"PCA1": "PCA1 ", "PCA2": "PCA2 "})
    # fig.update_traces(dict(marker_line_width = 1))
    # fig.update_layout(width = 1200, height = 800)
    # fig.show()

    ## checking significance
    scaler = StandardScaler()
    # CM confidence
    testData = finalData.sample(frac=1).reset_index(drop=True)
    X = testData[['PCA1', 'PCA2']]
    y = testData['Cell']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn = svm.SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    knn.fit(X_train_scaled, y_train)
    # knn.classes_
    # knn.predict_proba(X_test_scaled)

    y_pred = knn.predict(X_test_scaled)
    return accuracy_score(y_test, y_pred)

def TAS_Analysis(condenseCells, condenseNumber, removeLeft, removeLeftBoundary, histRemove, histRange, bootstrap, n_bootstraps, folDir, condList, groupList, channelViewList, maxChannels, plotFigure = False, reStats = False):
    channelRan = 0
    if reStats == True:
        statsReport = [[None] * 3 for _ in range(len(channelViewList))]
    while channelRan < len(channelViewList):
        if reStats == True:
            statsReport[channelRan][0] = channelViewList[channelRan]
        fileDir = []
        for n in range(len(condList)):
            # fileDir.append("F:/Matt/MIEL/SOX18 Chromatin project/20240531_Chromatin compaction/H3K27ac_488 H3K4me3_647 dapi/Result/" + condList[n] + "/feature_extraction")
            fileDir.append(folDir.replace("\\", "/") + "/" + condList[n] + "/feature_extraction")

        if channelViewList[channelRan] == "ALL":
            groupChannel = True
        else:
            groupChannel = False
        dataDir = pd.DataFrame()
        if groupChannel:
            # multiple conditions combine
            dataDir = pd.DataFrame()
            dataInt = pd.DataFrame()
            for n in range(len(fileDir)):
                files = os.listdir(fileDir[n])
                files.sort()
                row = None
                col = None
                fov = None
                dataTmp = pd.DataFrame()
                dataIntTmp = pd.DataFrame()
                for m in files:
                    r = m.split('_')[0].split('fov')[0].split('col')[0].split('row')[1]
                    c = m.split('_')[0].split('fov')[0].split('col')[1]
                    f = m.split('_')[0].split('fov')[1]
                    if (r == row) and (c == col) and (f in channelViewList):
                        fov = m.split('_')[0].split('fov')[1]
                        data = read_dataframe_from_h5_file(fileDir[n] + "/" + m)
                        # data["Cell"] = groupList[n]
                        dataTmp = pd.concat([dataTmp, data.iloc[:, 7:-1]], axis = 1)
                        dataIntTmp = pd.concat([dataIntTmp, data.iloc[:, 6]], axis = 1)
                        if f == maxChannels:
                            dataDir = pd.concat([dataDir, dataTmp], ignore_index = True)
                            dataTmp = pd.DataFrame()
                            dataInt = pd.concat([dataInt, dataIntTmp], ignore_index = True)
                            dataIntTmp = pd.DataFrame()
                    else:
                        # dataDir = pd.concat([dataDir, dataTmp], ignore_index = True)
                        # dataTmp = pd.DataFrame()
                        row = m.split('_')[0].split('fov')[0].split('col')[0].split('row')[1]
                        col = m.split('_')[0].split('fov')[0].split('col')[1]
                        fov = m.split('_')[0].split('fov')[1]
                        data = read_dataframe_from_h5_file(fileDir[n] + "/" + m)
                        rNumber = re.findall(r'\d+', m)[0]
                        data["Cell"] = groupList[n]
                        dataTmp = data.copy()
                        1
        else:
            # single channel
            for n in range(len(fileDir)):
                files = os.listdir(fileDir[n])
                for m in files:
                    f = m.split('_')[0].split('fov')[1]
                    if f == channelViewList[channelRan]:
                        data = read_dataframe_from_h5_file(fileDir[n] + "/" + m)
                        rNumber = re.findall(r'\d+', m)[0]
                        data["Cell"] = groupList[n]
                        dataDir = pd.concat([dataDir, data], ignore_index = True)
                        1

        ## Intensity saving
        if groupChannel == True:
            # dataDir.loc[:, ["HUV_object_average_intensity", "Cell"]]
            dataIntFinal = pd.concat([dataDir.loc[:, ["Cell", "HUV_object_average_intensity"]], dataInt], axis = 1)
            dataIntFinal = dataIntFinal[(dataDir["MOR_object_pixel_count"] >= histRange[0]) & (dataDir["MOR_object_pixel_count"] <= histRange[1])]
            dataIntFinal.to_excel(saveDir + project_name + "_Intensity" + ".xlsx")
        ## REMOVE THIS FOR DEBUGGING
        dataDirHold = dataDir.loc[:, ["Path", "Cell"]]
        dataDir.drop(["Path", "Cell"], axis = 1, inplace = True)
        dataDir.columns = list(range(np.shape(dataDir)[1]))
        dataDir["Path"] = dataDirHold.loc[:, "Path"]
        dataDir["Cell"] = dataDirHold.loc[:, "Cell"]
        dataDir.rename({5: "MOR_object_pixel_count"}, axis = 1, inplace = True)
        ##

        if plotFigure == True:
            nbins = np.max(dataDir["MOR_object_pixel_count"]) / bin_number_for_plot
            fig = px.histogram(dataDir, x = 'MOR_object_pixel_count', color = "Cell", histnorm = "percent", barmode="overlay", nbins = int(np.round(nbins)), range_x = [0, histPlotRange])
            if histRemove == True:
                fig.add_vline(x = histRange[0], line_width = 1, line_dash = "dash", line_color = "red")
                fig.add_vline(x = histRange[1], line_width = 1, line_dash = "dash", line_color = "red")
            fig.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
            fig.show()
            fig.write_image(saveDir + project_name + "_Size_Condition_C" + str(channelRan) + "_" + saveFormat)

            fig = px.histogram(dataDir, x = 'MOR_object_pixel_count', histnorm = "percent", barmode="overlay", nbins = int(np.round(nbins)), range_x = [0, histPlotRange])
            if histRemove == True:
                fig.add_vline(x = histRange[0], line_width = 1, line_dash = "dash", line_color = "red")
                fig.add_vline(x = histRange[1], line_width = 1, line_dash = "dash", line_color = "red")
            fig.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
            fig.show()
            fig.write_image(saveDir + project_name + "_Size_All_C" + str(channelRan) + "_" + saveFormat)

        if histRemove == True:
            dataDir = dataDir[(dataDir["MOR_object_pixel_count"] >= histRange[0]) & (dataDir["MOR_object_pixel_count"] <= histRange[1])]
            dataDir.reset_index(inplace = True)

            if plotFigure == True:
                nbins = np.max(dataDir["MOR_object_pixel_count"]) / bin_number_for_plot
                fig = px.histogram(dataDir, x = 'MOR_object_pixel_count', color = "Cell", histnorm = "percent", barmode="overlay", nbins = int(np.round(nbins)), range_x = [0, histPlotRange])
                fig.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
                fig.show()
                fig.write_image(saveDir + project_name + "_Size_Condition_Selected_C" + str(channelRan) + "_" + saveFormat)

                fig = px.histogram(dataDir, x = 'MOR_object_pixel_count', histnorm = "percent", barmode="overlay", nbins = int(np.round(nbins)), range_x = [0, histPlotRange])
                fig.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
                fig.show()
                fig.write_image(saveDir + project_name + "_Size_All_Selected_C" + str(channelRan) + "_" + saveFormat)

        ## Bootstrap
        if bootstrap == True:
            dataConcat = pd.DataFrame()
            cellType = dataDir["Cell"].unique().tolist()
            for n in range(len(cellType)):
                dataType = dataDir.loc[dataDir["Cell"] == cellType[n], ]
                dataType.reset_index(drop = True, inplace = True)
                for n in range(n_bootstraps):
                    randInd = dataType.sample(n = int(np.floor(0.5 * len(dataType))), random_state = 1, replace = True).index.tolist()
                    # condense = dataType.iloc[randInd, :-2].mean()
                    # condense["Path"] = dataType.iloc[randInd[0], -2]
                    # condense["Cell"] = dataType.iloc[randInd[0], -1]
                    # dataConcat = pd.concat([dataConcat, condense.to_frame().transpose()], ignore_index = True)
                    dataConcat = pd.concat([dataConcat, dataType.loc[randInd, ]], ignore_index = True)
        else:
            dataConcat = dataDir.copy()
        dataDir = dataConcat.copy()

        ## Real Bootstrap
        if realBS == True:
            mpPool = mp.Pool(mp.cpu_count())
            dataBS = mpPool.map(bootstrapParrallel, ((dataDir, condenseCells, condenseNumber, folDir, condList, groupList, channelViewList, maxChannels) for n in range(1000)))
            mpPool.close()
            dataBSSave = pd.DataFrame(dataBS)
            dataBSSave.to_excel(saveDir + project_name + "_PCA_C" + str(channelRan) + "_Condense_" + str(condenseNumber) + ".xlsx")

        dataConcat = pd.DataFrame()
        if condenseCells == True:
            cellType = dataDir["Cell"].unique().tolist()
            for n in range(len(cellType)):
                dataType = dataDir.loc[dataDir["Cell"] == cellType[n], ]
                dataType.reset_index(drop = True, inplace = True)
                while len(dataType) > condenseNumber:
                    randInd = dataType.sample(n = condenseNumber, random_state = 1, replace = False).index.tolist()
                    condense = dataType.iloc[randInd, :-2].mean()
                    condense["Path"] = dataType.iloc[randInd[0], -2]
                    condense["Cell"] = dataType.iloc[randInd[0], -1]
                    dataConcat = pd.concat([dataConcat, condense.to_frame().transpose()], ignore_index = True)
                    dataType.drop(randInd, inplace = True)
                    dataType.reset_index(drop = True, inplace = True)
        else:
            dataConcat = dataDir.copy()

        finalData = dataConcat.copy()
        #Path PCA
        if condenseCells == True:
            finalData["Cell"] = finalData["Path"]
        else:
            finalData["C"] = finalData["Path"].str.extract(r'/Result/C(\d+)')
            finalData["Path"] = (pd.to_numeric(finalData['Path'].str.extract(r'[Cc][Oo][Ll](\d+)', expand=False), errors='coerce') * 100) + pd.to_numeric(finalData['Path'].str.extract(r'[Rr][Oo][Ww](\d+)', expand=False), errors='coerce')
        dataDir = finalData.copy()
        for cell in np.unique(dataDir.C):
            mask = np.isin(dataDir.C, cell)
            print(np.unique(dataDir[mask]["Path"]))
            if len(dataDir[mask]["Path"])%2 == 0:
                divInt = int(len(dataDir[mask]["Path"]) / 2)
            else:
                divInt = int(np.floor(len(dataDir[mask]["Path"])/2) + 1)
            # dataDir[mask].loc[dataDir[mask]["Path"] >= np.unique(dataDir[mask]["Path"])[divInt], 'Path'] -= np.unique(dataDir[mask]["Path"])[divInt]
            dataDir.loc[mask & (dataDir["Path"] >= np.unique(dataDir[mask]["Path"])[divInt]), 'Path'] -= np.unique(dataDir[mask]["Path"])[divInt]
            print(np.unique(dataDir[mask]["Path"]))
        dataDir = dataDir.drop(["C"], axis = 1)

        # feature map heat map
        # dataforHeatMap = dataDir.iloc[:, 8:]
        # dataforHeatMap = dataforHeatMap.drop(["Path"], axis = 1)
        # dataforHeatMap = dataforHeatMap.groupby(["Cell"]).mean()
        # scaler = StandardScaler()
        # zscoreData = scaler.fit_transform(dataforHeatMap)
        # # dataforHeatMap.drop([6], axis = 1)
        # avgZScoreData = zscoreData.reshape(zscoreData.shape[0], -1, 2).mean(axis=2)
        # fig = px.imshow(avgZScoreData[[2, 3, 0, 1]], color_continuous_scale = px.colors.diverging.RdBu)
        # fig.show()
        # fig.write_image("/home/gic/Downloads/dapi_tert_tas_features.svg")

        # Spearman
        spearData = dataDir.copy()
        spearData = spearData.drop(["Path"], axis = 1)
        spearData = spearData.iloc[:, 7:]
        cond_means = spearData.groupby(["Cell"]).mean().T
        conditions = cond_means.columns.tolist()
        n_conds = len(conditions)
        corr_matrix = pd.DataFrame(np.ones((n_conds, n_conds)),
                                   index=conditions, columns=conditions)
        p_matrix = pd.DataFrame(np.ones((n_conds, n_conds)),
                                index=conditions, columns=conditions)
    
        for i, cond_a in enumerate(conditions):
            for j, cond_b in enumerate(conditions):
                if i == j:
                    continue
                # Extract two vectors (each is a feature list for that condition)
                x = cond_means[cond_a].values
                y = cond_means[cond_b].values
                # Spearman correlation
                rho, p = spearmanr(x, y)
                corr_matrix.loc[cond_a, cond_b] = rho
                p_matrix.loc[cond_a, cond_b] = p
    
        annot = pd.DataFrame(index=conditions, columns=conditions)
        for i in conditions:
            for j in conditions:
                rho = corr_matrix.loc[i, j]
                pval = p_matrix.loc[i, j]
                star = sig_stars(pval)
                annot.loc[i, j] = f"{rho:.2f}{star}"

        lower_triangular_df = corr_matrix.where(np.tril(np.ones(corr_matrix.shape)).astype(bool))
        annot_lf = annot.where(np.tril(np.ones(annot.shape)).astype(bool))
        annot_lf = annot_lf.fillna("")
        fig = px.imshow(lower_triangular_df.round(2), color_continuous_scale = "RdBu_r")
        fig.update_traces(text = annot_lf, texttemplate="%{text}")
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        fig.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
        fig.show(renderer = "browser")
        fig.write_image(saveDir + project_name + "_Spearman_C" + str(channelRan) + saveFormat)

        # Correlation Plot of Spearman
        # dataCorPlot = spearData.groupby(["Cell"]).mean()
        # for g1, g2 in combinations(np.unique(spearData.Cell), 2):
        #     mask = np.isin(dataCorPlot.index, [g1, g2])
        #     corFig = px.scatter(dataCorPlot[mask].T, x = g1, y = g2)
        #     corFig.show(renderer = "browser")

        # LDA
        ldaData = dataDir.copy()
        ldaData = ldaData.drop(["Path"], axis = 1)
        ldaData = ldaData.iloc[:, 7:]
        lda = LinearDiscriminantAnalysis(n_components=1)
        # X_train, X_test, y_train, y_test = train_test_split(ldaData.iloc[:, :-1], ldaData["Cell"], test_size = 0.3)
        # lda.fit(X_train, y_train)
        # confidences = lda.predict_proba(X_test)   # confidence of the split
        # accuracy = lda.score(X_test, y_test)
        # print(confidences)
        # print(accuracy)

        X_lda = lda.fit_transform(ldaData.iloc[:, :-1], ldaData["Cell"])
        ldaResult = pd.DataFrame({"LDA": X_lda.flatten(), "Cell": dataDir["Cell"]})
        ldaResult.groupby(["Cell"]).mean().to_csv(saveDir + project_name + "_LDA_C" + str(channelRan) + ".csv")

        ## PCA
        pca_analysis = PCA(n_components=2)
        # principalComponents = pca_analysis.fit_transform(dataDir.iloc[:, 4:-2].values)
        dataforPCA = dataDir.drop(["Path", "Cell"], axis = 1)

        mask = np.isin(dataDir.Cell, ["Sox_Pre", "Sox_Post"])
        dataforPCA = dataforPCA.iloc[mask,]
        dataDirLab = dataDir.iloc[mask,]
        dataDirLab.reset_index(inplace = True)

        scaler = StandardScaler()
        dataforPCA = dataforPCA.drop(index=np.where(np.isnan(dataforPCA.iloc[:, 8:].values))[0]) #250703: Removing intensity from being used as a factor for PCA
        zscoreData = scaler.fit_transform(dataforPCA.iloc[:, 8:].values)

        principalComponents = pca_analysis.fit_transform(zscoreData)
        finalData = pd.DataFrame(data = principalComponents, columns = ['PCA1', 'PCA2'])
        # finalData["Cell"] = dataDir["Cell"]
        # finalData["Path"] = dataDir["Path"]
        finalData["Cell"] = dataDirLab["Cell"]
        finalData["Path"] = dataDirLab["Path"]

        # ftr_ctr = pca_analysis.components_
        ctr = principalComponents**2 / np.sum(principalComponents**2, axis = 0)
        PCA_ratio = pca_analysis.explained_variance_ratio_
        
        # dataDir = finalData.copy()
        # if removeLeft == True:
        #     dataDir = dataDir.loc[dataDir["PCA1"] >= removeLeftBoundary, ]
        #     dataDir.reset_index(drop = True, inplace = True)
        
        if plotFigure == True:
            plotTitle = "PCA"
            if condenseCells == True:
                plotTitle = plotTitle + " condense " + str(condenseNumber)
            if bootstrap == True:
                plotTitle = plotTitle + " bootstrap " + str(n_bootstraps)
            if histRemove == True:
                plotTitle = plotTitle + " size " + str(histRange[0]) + "-" + str(histRange[1]) # TODO: Colour coding for final PCA
            fig = px.scatter(finalData, title = plotTitle, x = 'PCA1', y = 'PCA2', color = "Cell", labels = {"PCA1": "PCA1 " + f"({PCA_ratio[0]*100:.2f}%)", "PCA2": "PCA2 " + f"({PCA_ratio[1]*100:.2f}%)"}, hover_name = "Path",
                             color_discrete_map = {"Sox_Pre": "#EC3600", "Sox_Post": "#008BA8",
                                                   "No Sox_Pre": "#8F8F96", "No Sox_Post": "#109875"})
            # fig = px.scatter_3d(finalData, title = plotTitle, x = 'PCA1', y = 'PCA2', z = 'PCA3', color = "Cell", labels = {"PCA1": "PCA1 " + f"({PCA_ratio[0]*100:.2f}%)", "PCA2": "PCA2 " + f"({PCA_ratio[1]*100:.2f}%)", "PCA3": "PCA3 " + f"({PCA_ratio[2]*100:.2f}%)"})
            # fig = px.scatter(finalData, title = plotTitle, x = 'PCA1', y = 'PCA2', color = "Cell", labels = {"PCA1": "PCA1 " + f"({PCA_ratio[0]*100:.2f}%)", "PCA2": "PCA2 " + f"({PCA_ratio[1]*100:.2f}%)"}, color_discrete_map={"Ht_7d_SM4": 'rgba(15, 153, 178, 1)', "Htoe_7d_SM4": 'rgba(42, 162, 132, 1)', "Ht_7d_DMSO": 'rgba(151, 151, 156, 1)', "Htoe_7d_DMSO": 'rgba(236, 68, 1, 1)'})
            # fig = px.scatter(finalData, title = plotTitle, x = 'PCA1', y = 'PCA2', color = "Cell", labels = {"PCA1": "PCA1 " + f"({PCA_ratio[0]*100:.2f}%)", "PCA2": "PCA2 " + f"({PCA_ratio[1]*100:.2f}%)"}, color_discrete_map={"LEC SM4": 'rgba(15, 153, 178, 1)', "KLEC SM4": 'rgba(74, 165, 140, 1)', "LEC DMSO": 'rgba(151, 151, 156, 1)', "KLEC DMSO": 'rgba(236, 68, 1, 1)'})
            fig.update_traces(dict(marker_line_width = 1))
            # fig.write_image(fileDir + "/output_pca.svg")
            # fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
            # fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
            # fig.update_layout(width = 1200, height = 800, font = dict(size = 30))
            # fig.show()
            # fig.write_image(saveDir + project_name + "_PCA_C" + str(channelRan) + "_font30" + saveFormat)
            fig.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
            fig.show()
            fig.write_image(saveDir + project_name + "_PCA_C" + str(channelRan) + "_" + saveFormat)
            fig.write_html(saveDir + project_name + "_PCA_C" + str(channelRan) + ".html")

            statData = finalData.groupby(["Cell"]).mean()
            statData["Cell"] = statData.index
            fig2 = px.scatter(statData, x = 'PCA1', y = 'PCA2', color = "Cell", labels = {"PCA1": "PCA1" + f" ({PCA_ratio[0]*100:.2f}%)", 'PCA2': "PCA2" + f" ({PCA_ratio[1]*100:.2f}%)"},
                              color_discrete_map = {"Sox_Pre": "#EC3600", "Sox_Post": "#008BA8",
                                                    "No Sox_Pre": "#8F8F96", "No Sox_Post": "#109875"},
                              symbol_sequence = ["star", "star", "star", "star"])
            fig2.update_traces(marker_size = 20)
            fig2.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
            fig2.write_image(saveDir + project_name + "_PCA_Star_Only_C" + str(channelRan) + "_" + saveFormat)
            fig2.show(renderer = "browser")
            for trace_adding in range(len(list(np.unique(statData["Cell"])))):
                fig.add_trace(fig2.data[trace_adding])
            fig.update_xaxes(range=[-20, 5])
            fig.update_yaxes(range=[-20, 5])
            fig.show(renderer = "browser")
            fig.write_image(saveDir + project_name + "_PCA_Agg_Points_Star_" + "PCA" + "_C" + str(channelRan) + "_2D_PCA_New" + saveFormat)

            fig = px.scatter(finalData, title = plotTitle, x = 'PCA1', y = 'PCA2', symbol = "Cell", color = "Path", labels = {"PCA1": "PCA1 " + f"({PCA_ratio[0]*100:.2f}%)", "PCA2": "PCA2 " + f"({PCA_ratio[1]*100:.2f}%)"}, hover_name = "Path")
            fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside"))
            fig.write_image(saveDir + project_name + "_PCA_shape_C" + str(channelRan) + "_" + saveFormat)
            fig.write_html(saveDir + project_name + "_PCA_shape_C" + str(channelRan) + ".html")
            fig.show(renderer = "browser")
    
            # fig = px.scatter_3d(finalData, title = plotTitle, x = 'PCA1', y = 'PCA2', z = 'PCA3', color = "Cell", labels = {"PCA1": "PCA1 " + f"({PCA_ratio[0]*100:.2f}%)", "PCA2": "PCA2 " + f"({PCA_ratio[1]*100:.2f}%)", "PCA3": "PCA3 " + f"({PCA_ratio[2]*100:.2f}%)"})
            # fig.update_traces(dict(marker_line_width = 1))
            # fig.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
            # fig.show()
            # fig.write_image(saveDir + project_name + "_PCA3D_C" + str(channelRan) + "_" + saveFormat)
            # fig.write_html(saveDir + project_name + "_PCA3D_C" + str(channelRan) + ".html")

        # Outputting Euclidean Distance
        statData = finalData.groupby(["Cell"]).mean()
        statData["Cell"] = statData.index
        dist = squareform(pdist(statData.loc[:, ["PCA1", "PCA2"]], "euclidean"))
        dist_df = pd.DataFrame(dist, columns = statData["Cell"], index = statData["Cell"])
        dist_df.to_csv(saveDir + project_name + "_Euclidean_Distance_PCA_C" + str(channelRan) + ".csv")

        # Euclidean distance heatmap
        plotTitle = "Euclidean Distance"
        dist_df_ll = dist_df.where(np.tril(np.ones(dist_df.shape)).astype(bool))
        dist_df_ll = dist_df_ll.fillna("")
        fig = px.imshow(dist_df_ll.round(2), color_continuous_scale = "RdBu", title = plotTitle)
        fig.update_traces(text = dist_df_ll.round(2), texttemplate="%{text:.2f}")
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        fig.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
        fig.show(renderer = "browser")
        fig.write_image(saveDir + project_name + "_Euclidean_Distance_C" + str(channelRan) + saveFormat)

        ## PCA
        # principalComponents = pca_analysis.fit_transform(dataDir.iloc[:, 4:-2].values)
        dataforPCA = dataDir.drop(["Path", "Cell"], axis = 1)

        # mask = np.isin(dataDir.Cell, ["No Sox_Pre", "No Sox_Post"])
        # dataforPCA = dataforPCA.iloc[mask,]
        # dataDirLab = dataDir.iloc[mask,]
        # dataDirLab.reset_index(inplace = True)

        scaler = StandardScaler()
        dataforPCA = dataforPCA.drop(index=np.where(np.isnan(dataforPCA.iloc[:, 8:].values))[0]) #250703: Removing intensity from being used as a factor for PCA
        zscoreData = scaler.fit_transform(dataforPCA.iloc[:, 8:].values)

        principalComponents = pca_analysis.transform(zscoreData)
        finalData = pd.DataFrame(data = principalComponents, columns = ['PCA1', 'PCA2'])
        finalData["Cell"] = dataDir["Cell"]
        finalData["Path"] = dataDir["Path"]
        # finalData["Cell"] = dataDirLab["Cell"]
        # finalData["Path"] = dataDirLab["Path"]

        # ftr_ctr = pca_analysis.components_
        ctr = principalComponents**2 / np.sum(principalComponents**2, axis = 0)
        PCA_ratio = pca_analysis.explained_variance_ratio_
        
        if plotFigure == True:
            plotTitle = "PCA"
            if condenseCells == True:
                plotTitle = plotTitle + " condense " + str(condenseNumber)
            if bootstrap == True:
                plotTitle = plotTitle + " bootstrap " + str(n_bootstraps)
            if histRemove == True:
                plotTitle = plotTitle + " size " + str(histRange[0]) + "-" + str(histRange[1]) # TODO: Colour coding for final PCA
            fig = px.scatter(finalData, title = plotTitle, x = 'PCA1', y = 'PCA2', color = "Cell", labels = {"PCA1": "PCA1 " + f"({PCA_ratio[0]*100:.2f}%)", "PCA2": "PCA2 " + f"({PCA_ratio[1]*100:.2f}%)"}, hover_name = "Path",
                             color_discrete_map = {"Sox_Pre": "#EC3600", "Sox_Post": "#008BA8",
                                                   "No Sox_Pre": "#8F8F96", "No Sox_Post": "#109875"})
            # fig = px.scatter_3d(finalData, title = plotTitle, x = 'PCA1', y = 'PCA2', z = 'PCA3', color = "Cell", labels = {"PCA1": "PCA1 " + f"({PCA_ratio[0]*100:.2f}%)", "PCA2": "PCA2 " + f"({PCA_ratio[1]*100:.2f}%)", "PCA3": "PCA3 " + f"({PCA_ratio[2]*100:.2f}%)"})
            # fig = px.scatter(finalData, title = plotTitle, x = 'PCA1', y = 'PCA2', color = "Cell", labels = {"PCA1": "PCA1 " + f"({PCA_ratio[0]*100:.2f}%)", "PCA2": "PCA2 " + f"({PCA_ratio[1]*100:.2f}%)"}, color_discrete_map={"Ht_7d_SM4": 'rgba(15, 153, 178, 1)', "Htoe_7d_SM4": 'rgba(42, 162, 132, 1)', "Ht_7d_DMSO": 'rgba(151, 151, 156, 1)', "Htoe_7d_DMSO": 'rgba(236, 68, 1, 1)'})
            # fig = px.scatter(finalData, title = plotTitle, x = 'PCA1', y = 'PCA2', color = "Cell", labels = {"PCA1": "PCA1 " + f"({PCA_ratio[0]*100:.2f}%)", "PCA2": "PCA2 " + f"({PCA_ratio[1]*100:.2f}%)"}, color_discrete_map={"LEC SM4": 'rgba(15, 153, 178, 1)', "KLEC SM4": 'rgba(74, 165, 140, 1)', "LEC DMSO": 'rgba(151, 151, 156, 1)', "KLEC DMSO": 'rgba(236, 68, 1, 1)'})
            fig.update_traces(dict(marker_line_width = 1))
            # fig.write_image(fileDir + "/output_pca.svg")
            # fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
            # fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
            # fig.update_layout(width = 1200, height = 800, font = dict(size = 30))
            # fig.show()
            # fig.write_image(saveDir + project_name + "_PCA_C" + str(channelRan) + "_font30" + saveFormat)
            fig.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
            fig.show(renderer = "browser")
            fig.write_image(saveDir + project_name + "_PCA_C" + str(channelRan) + "_" + saveFormat)
            fig.write_html(saveDir + project_name + "_PCA_C" + str(channelRan) + ".html")
            
            statData = finalData.groupby(["Cell"]).mean()
            statData["Cell"] = statData.index
            fig2 = px.scatter(statData, x = 'PCA1', y = 'PCA2', color = "Cell", labels = {"PCA1": "PCA1" + f" ({PCA_ratio[0]*100:.2f}%)", 'PCA2': "PCA2" + f" ({PCA_ratio[1]*100:.2f}%)"},
                              color_discrete_map = {"Sox_Pre": "#EC3600", "Sox_Post": "#008BA8",
                                                    "No Sox_Pre": "#8F8F96", "No Sox_Post": "#109875"},
                              symbol_sequence = ["star", "star"])
            fig2.update_traces(marker_size = 20)
            fig2.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
            fig2.write_image(saveDir + project_name + "_PCA_Star_Only_C" + str(channelRan) + "_" + saveFormat)
            fig2.show(renderer = "browser")
            for trace_adding in range(len(list(np.unique(statData["Cell"])))):
                fig.add_trace(fig2.data[trace_adding])
            fig.update_xaxes(range=[-20, 5])
            fig.update_yaxes(range=[-20, 5])
            fig.show(renderer = "browser")
            fig.write_image(saveDir + project_name + "_PCA_Agg_Points_NoSox_Star_" + "PCA" + "_C" + str(channelRan) + "_2D_PCA_New" + saveFormat)

            fig = px.scatter(finalData, title = plotTitle, x = 'PCA1', y = 'PCA2', symbol = "Cell", color = "Path", labels = {"PCA1": "PCA1 " + f"({PCA_ratio[0]*100:.2f}%)", "PCA2": "PCA2 " + f"({PCA_ratio[1]*100:.2f}%)"}, hover_name = "Path")
            fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside"))
            fig.write_image(saveDir + project_name + "_PCA_shape_C" + str(channelRan) + "_" + saveFormat)
            fig.write_html(saveDir + project_name + "_PCA_shape_C" + str(channelRan) + ".html")
            fig.show(renderer = "browser")
            1

        # Outputting Euclidean Distance
        statData = finalData.groupby(["Cell"]).mean()
        statData["Cell"] = statData.index
        dist = squareform(pdist(statData.loc[:, ["PCA1", "PCA2"]], "euclidean"))
        dist_df = pd.DataFrame(dist, columns = statData["Cell"], index = statData["Cell"])
        dist_df.to_csv(saveDir + project_name + "_Euclidean_Distance_PCA_C" + str(channelRan) + ".csv")

        # Euclidean distance heatmap
        plotTitle = "Euclidean Distance"
        dist_df_ll = dist_df.where(np.tril(np.ones(dist_df.shape)).astype(bool))
        dist_df_ll = dist_df_ll.fillna("")
        fig = px.imshow(dist_df_ll.round(2), color_continuous_scale = "RdBu", title = plotTitle)
        fig.update_traces(text = dist_df_ll.round(2), texttemplate="%{text:.2f}")
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        fig.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
        fig.show(renderer = "browser")
        fig.write_image(saveDir + project_name + "_Euclidean_Distance_NoSox_C" + str(channelRan) + saveFormat)

        1
        channelRan = channelRan + 1

        # ## TSNE
        # tsne = TSNE(n_components=2, random_state=0)
        # finalData = tsne.fit_transform(dataConcat.iloc[:, 4:-2].values)

        # fig = px.scatter(
        #     finalData, x=0, y=1,
        #     color=dataConcat.Cell, labels={'color': 'species'}
        # )
        # # fig.write_image(fileDir + "/output_tsne.svg")
        # # fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        # # fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
        # fig.show()
        # 1

        # ## UMAP
        # umap_2d = UMAP(n_components=2, init='random', random_state=0)
        # proj_2d = umap_2d.fit_transform(dataConcat.iloc[:, 4:-2].values)
        # fig = px.scatter(
        #     finalData, x=0, y=1,
        #     color=dataConcat.Cell, labels={'color': 'species'}
        # )
        # # fig.write_image(fileDir + "/output_umap.svg")
        # fig.show()
        # 1

        # Overlay segmentation on raw iamges
        # from matplotlib import pyplot as plt
        # import plotly.express as px
        # plotFolDir = '/'.join(folDir.replace("\\", "/").split("/")[0:-1]) + '/Result/' + condList[n] + '/segmentation'
        # files = os.listdir(plotFolDir)
        # for i in files:
        #     data = read_segmentation_data(plotFolDir + "/" + i)
        #     test = data[1]['points']
        #     # plt.imshow(test, interpolation='nearest')
        #     # plt.show()
        #     test = data[0]['1']['HUV']
        #     fig = px.imshow(test)
        #     fig.show()
        #     # get coordinates from the [1]['coord'][0][0] and [1]['coord'][0][1] and plot them to see how they look
        #     pass
        # plt.imshow(test[0], interpolation='nearest')
        # plt.show()
        # import plotly.express as px
        # fig = px.imshow(test[0])
        # fig.show()

        ## checking significance
        scaler = StandardScaler()
        cm_score = np.zeros(n_test)
        clf_score = np.zeros(n_test)
        for n in range(n_test):
            # CM confidence
            testData = finalData.sample(frac=1).reset_index(drop=True)
            X = testData[['PCA1', 'PCA2']]
            y = testData['Cell']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            knn = svm.SVC()
            # knn = svm.SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
            knn.fit(X_train_scaled, y_train)
            # knn.classes_
            # knn.predict_proba(X_test_scaled)

            y_pred = knn.predict(X_test_scaled)
            cm_score[n] = accuracy_score(y_test, y_pred)
            clf = QuadraticDiscriminantAnalysis()
            clf.fit(X, y)
            clf_score[n] = clf.score(X, y)
        
        finalData = finalData.sample(frac=1, random_state = 42).reset_index(drop=True)

        X = finalData[['PCA1', 'PCA2']]
        y = finalData['Cell']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # knn = KNeighborsClassifier(n_neighbors=5)
        # knn = LogisticRegression()
        # knn = LinearRegression()
        knn = svm.SVC()
        knn.fit(X_train_scaled, y_train)

        y_pred = knn.predict(X_test_scaled)

        # accuracy = accuracy_score(y_test, y_pred)
        # precision = precision_score(y_test, y_pred)
        # recall = recall_score(y_test, y_pred)
        # f1 = f1_score(y_test, y_pred)
        if plotFigure == True:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
        # print(f'Accuracy: {accuracy:.2f}')
        # print(f'Precision: {precision:.2f}')
        # print(f'Recall: {recall:.2f}')
        # print(f'F1-score: {f1:.2f}')

        classLabels = y_test.unique().tolist()
        classLabels.sort()
        if manualOrder == True:
            classLabels = manualOrderList
        cm = confusion_matrix(y_test, y_pred, labels = classLabels)
        if plotFigure == True:
            plotTitle = "Confusion Matrix"
            if condenseCells == True:
                plotTitle = plotTitle + " condense " + str(condenseNumber)
            if bootstrap == True:
                plotTitle = plotTitle + " bootstrap " + str(n_bootstraps)
            if histRemove == True:
                plotTitle = plotTitle + " size " + str(histRange[0]) + "-" + str(histRange[1])
            plotTitle = plotTitle + " average confidence " + str(np.mean(cm_score))
            fig = px.imshow(cm, text_auto = True, title = plotTitle, color_continuous_scale = px.colors.sequential.Blues)
            fig.update_layout(xaxis_title = "Predicted", yaxis_title = "Actual",
                              xaxis = dict(tickmode = 'array', tickvals = list(range(0, len(classLabels))), ticktext = classLabels),
                              yaxis = dict(tickmode = 'array', tickvals = list(range(0, len(classLabels))), ticktext = classLabels))
            fig.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
            fig.show()
            fig.write_image(saveDir + project_name + "_CM" + saveFormat)

            # Plot decision boundary
            plot_decision_boundary(X, y, knn, scaler, finalData, PCA_ratio)
        clf = QuadraticDiscriminantAnalysis()
        clf.fit(X, y)
        y_pred_clf = clf.predict(X)
        clf_df = pd.DataFrame({"real": y, "predict": y_pred_clf})
        clf_data = np.zeros((len(classLabels), len(classLabels)))
        for n in range(len(classLabels)): # row (real)
            for m in range(len(classLabels)): # col (predicted)
                clf_data[n, m] = np.round(len(clf_df.loc[(clf_df["predict"] == classLabels[m]) & (clf_df["real"] == classLabels[n]),]) / len(clf_df.loc[clf_df["real"] == classLabels[n]]) * 100, decimals = 1)
        if plotFigure == True:
            plotTitle = "Quadratic Discriminant Analysis"
            if condenseCells == True:
                plotTitle = plotTitle + " condense " + str(condenseNumber)
            if bootstrap == True:
                plotTitle = plotTitle + " bootstrap " + str(n_bootstraps)
            if histRemove == True:
                plotTitle = plotTitle + " size " + str(histRange[0]) + "-" + str(histRange[1])
            plotTitle = plotTitle + " average confidence " + str(np.mean(clf_score))
            fig = px.imshow(clf_data, text_auto = True, title = plotTitle, color_continuous_scale = px.colors.sequential.Blues)
            fig.update_layout(xaxis_title = "Predicted", yaxis_title = "Actual",
                              xaxis = dict(tickmode = 'array', tickvals = list(range(0, len(classLabels))), ticktext = classLabels),
                              yaxis = dict(tickmode = 'array', tickvals = list(range(0, len(classLabels))), ticktext = classLabels))
            fig.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
            fig.show()
            fig.write_image(saveDir + project_name + "_QDA" + saveFormat)
        1

        ## Distance matrix
        distData = finalData.sort_values(by=["Cell"])
        distance_matrix = distance.squareform(distance.pdist(distData.loc[:, ["PCA1", "PCA2"]].to_numpy().astype(float)))
        if plotFigure == True:
            plotTitle = "Distance Matrix"
            if condenseCells == True:
                plotTitle = plotTitle + " condense " + str(condenseNumber)
            if bootstrap == True:
                plotTitle = plotTitle + " bootstrap " + str(n_bootstraps)
            if histRemove == True:
                plotTitle = plotTitle + " size " + str(histRange[0]) + "-" + str(histRange[1])
            fig = px.imshow(distance_matrix, title = plotTitle, color_continuous_scale = px.colors.diverging.RdBu)
            yy = distData.Cell.unique()
            labels = [[q+' '+str(yy[0])+' '+q+' '+str(yy[1])] for q in distData.Cell.unique()]
            # fig.update_xaxes(tickvals = np.cumsum([sum(distData.Cell == condition) for condition in distData.Cell.unique()]) - 1, ticktext = labels, showgrid = True, ticks = "outside", tickson = "boundaries", ticklen = 20)
            # fig.update_yaxes(tickvals = np.cumsum([sum(distData.Cell == condition) for condition in distData.Cell.unique()]) - 1, ticktext = labels, showgrid = True, ticks = "outside", tickson = "boundaries", ticklen = 20)

            tickLoc = np.cumsum([sum(distData.Cell == condition) for condition in distData.Cell.unique()]) - 1
            tickLocF = np.zeros(len(tickLoc) * 2)
            tickLabels = [None] * len(tickLoc) * 2
            tickLocF[0] = tickLoc[0] / 2
            tickLocF[1] = tickLoc[0]
            tickLabels[0] = labels[0]
            tickLabels[1] = ""
            for n in range(3, len(tickLocF), 2):
                ind = int((n - 1) / 2)
                tickLocF[n - 1] =  np.floor(np.floor(tickLoc[ind] - tickLoc[ind - 1]) / 2 + tickLoc[ind - 1])
                tickLocF[n] = tickLoc[ind]
                tickLabels[n - 1] = labels[ind]
                tickLabels[n] = ""
            fig.update_xaxes(tickvals = tickLocF, ticktext = tickLabels, showgrid = True, ticks = "outside", tickson = "boundaries", ticklen = 20)
            fig.update_yaxes(tickvals = tickLocF, ticktext = tickLabels, showgrid = True, ticks = "outside", tickson = "boundaries", ticklen = 20)
            fig.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
            fig.show()
            fig.write_image(saveDir + project_name + "_DM" + saveFormat)
            nnList = [None] * len(tickLoc)
            matOI = distance_matrix[0:tickLoc[0], 0:tickLoc[0]]
            matOI = np.where(matOI != 0, matOI, np.nan)
            nnList[0] = np.nanmin(matOI, axis = 1)
            for n in range(len(tickLoc) - 1):
                matOI = distance_matrix[tickLoc[n]:tickLoc[n + 1], tickLoc[n]:tickLoc[n + 1]]
                matOI = np.where(matOI != 0, matOI, np.nan)
                nnList[n + 1] = np.nanmin(matOI, axis = 1)
            nnMatrix = pd.DataFrame(nnList).T
            nnMatrix.columns = yy
            nnMatrix.to_excel(saveDir + project_name + "_DM.xlsx", index = False)

            ## Summary of distance matrix
            distPD = pd.DataFrame(distance_matrix, index = distData.Cell, columns = distData.Cell)
            distPD = distPD.stack().groupby(level = [0, 1]).mean().unstack()
            plotTitle = "Distance Matrix Condensed"
            if condenseCells == True:
                plotTitle = plotTitle + " condense " + str(condenseNumber)
            if bootstrap == True:
                plotTitle = plotTitle + " bootstrap " + str(n_bootstraps)
            if histRemove == True:
                plotTitle = plotTitle + " size " + str(histRange[0]) + "-" + str(histRange[1])
            fig = px.imshow(np.log(distPD), title = plotTitle, color_continuous_scale = px.colors.diverging.RdBu)
            fig.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
            fig.show()
            fig.write_image(saveDir + project_name + "_DMC" + saveFormat)
        
        if reStats == True:
            statsReport[channelRan - 1][1] = cm_score
            statsReport[channelRan - 1][2] = clf_score
    if reStats == True:
        return statsReport

def condenseOptimisation(condenseCellsTry, condenseCells, removeLeft, removeLeftBoundary, histRemove, histRange, bootstrap, n_bootstraps, folDir, condList, groupList, channelViewList, maxChannels, plotStats):
    statsDF = pd.DataFrame(columns = channelViewList)
    statsDF.insert(0, "condenseNumber", condenseCellsTry)
    # if mp.cpu_count() > 60:
    #     mpPool = mp.Pool(60)
    # else:
    #     mpPool = mp.Pool(mp.cpu_count())
    # data = mpPool.map(TAS_Analysis, ((True, condenseTry, removeLeft, removeLeftBoundary, histRemove, histRange, bootstrap, n_bootstraps, folDir, condList, groupList, channelViewList, maxChannels, False, True) for condenseTry in condenseCellsTry))
    # mpPool.close()
    for condenseTry in condenseCellsTry:
        statsReport = TAS_Analysis(True, condenseTry, removeLeft, removeLeftBoundary, histRemove, histRange, bootstrap, n_bootstraps, folDir, condList, groupList, channelViewList, maxChannels, plotFigure = False, reStats = True)
        for n in range(len(channelViewList)):
            for m in range(n_test):
                statsDF.loc[statsDF["condenseNumber"] == condenseTry, channelViewList[n] + "_CM-" + str(m)] = statsReport[n][1][m]
            statsDF.loc[statsDF["condenseNumber"] == condenseTry, channelViewList[n]] = np.mean(statsReport[n][1])
            # data variation
            statsDF.loc[statsDF["condenseNumber"] == condenseTry, channelViewList[n] + "_CM_min"] = np.mean(statsReport[n][1]) - np.min(statsReport[n][1])
            statsDF.loc[statsDF["condenseNumber"] == condenseTry, channelViewList[n] + "_CM_max"] = np.max(statsReport[n][1]) - np.mean(statsReport[n][1])
            # std
            statsDF.loc[statsDF["condenseNumber"] == condenseTry, channelViewList[n] + "_CM_min"] = np.std(statsReport[n][1])
            statsDF.loc[statsDF["condenseNumber"] == condenseTry, channelViewList[n] + "_CM_max"] = np.std(statsReport[n][1])
        for n in range(len(channelViewList)):
            for m in range(n_test):
                statsDF.loc[statsDF["condenseNumber"] == condenseTry, channelViewList[n] + "_CLF-" + str(m)] = statsReport[n][2][m]
            statsDF.loc[statsDF["condenseNumber"] == condenseTry, channelViewList[n]] = np.mean(statsReport[n][2])
            # data variation
            # statsDF.loc[statsDF["condenseNumber"] == condenseTry, channelViewList[n] + "_CLF_min"] = np.mean(statsReport[n][2]) - np.min(statsReport[n][2])
            # statsDF.loc[statsDF["condenseNumber"] == condenseTry, channelViewList[n] + "_CLF_max"] = np.max(statsReport[n][2]) - np.mean(statsReport[n][2])
            # STD
            statsDF.loc[statsDF["condenseNumber"] == condenseTry, channelViewList[n] + "_CLF_min"] = np.std(statsReport[n][2])
            statsDF.loc[statsDF["condenseNumber"] == condenseTry, channelViewList[n] + "_CLF_max"] = np.std(statsReport[n][2])
        # statsDF.loc[statsDF["condenseNumber"] == condenseTry, channelViewList] = np.reshape(statsReport, (len(channelViewList), 2)).T[1,:]
    meltID = ["condenseNumber"]
    meltID.extend([a + "_CM_min" for a in channelViewList])
    meltID.extend([a + "_CM_max" for a in channelViewList])
    minList = [a + "_CM_min" for a in channelViewList]
    maxList = [a + "_CM_max" for a in channelViewList]
    plotDF = statsDF.astype(np.float64).melt(id_vars = meltID, value_vars = channelViewList)
    plotDF["minus"] = np.max(plotDF.loc[:, minList], axis = 1)
    plotDF["max"] = np.max(plotDF.loc[:, maxList], axis = 1)
    plotDF["minConfident"] = plotDF["value"] - plotDF["minus"]
    statsQC = statsDF.copy()
    statsQC.loc[:, channelViewList] = np.reshape(plotDF["minConfident"], (len(channelViewList), len(condenseCellsTry))).T
    qcCheck = statsQC.loc[:, channelViewList].astype(np.float64).ge(0.95, axis = 1).all(axis = 1)
    if plotStats == True:
        # meltID = ["condenseNumber"]
        # meltID.extend([a + "_CM_min" for a in channelViewList])
        # meltID.extend([a + "_CM_max" for a in channelViewList])
        # minList = [a + "_CM_min" for a in channelViewList]
        # maxList = [a + "_CM_max" for a in channelViewList]
        # plotDF = statsDF.astype(np.float64).melt(id_vars = meltID, value_vars = channelViewList)
        # plotDF["minus"] = np.max(plotDF.loc[:, minList], axis = 1)
        # plotDF["max"] = np.max(plotDF.loc[:, maxList], axis = 1)
        # fig = px.line(statsDF.astype(np.float64).melt(id_vars = ["condenseNumber"], value_vars = channelViewList), x = "condenseNumber", y = "value", color = "variable")
        # data variation
        # fig = px.line(plotDF, title = "CM Confidence", x = "condenseNumber", y = "value", error_y = "minus", error_y_minus = "max", color = "variable")
        # std
        plotTitle = "CM Confidence"
        if sum(qcCheck) == 0:
            pass
        else:
            plotTitle = plotTitle + "condense " + str(condenseCellsTry[qcCheck.idxmax()])
        fig = px.line(plotDF, title = plotTitle, x = "condenseNumber", y = "value", error_y = "minus", color = "variable")
        fig.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
        fig.show()
        fig.write_image(saveDir + project_name + "_CM_Confidence" + saveFormat)

        meltID = ["condenseNumber"]
        meltID.extend([a + "_CLF_min" for a in channelViewList])
        meltID.extend([a + "_CLF_max" for a in channelViewList])
        minList = [a + "_CLF_min" for a in channelViewList]
        maxList = [a + "_CLF_max" for a in channelViewList]
        plotDF = statsDF.astype(np.float64).melt(id_vars = meltID, value_vars = channelViewList)
        plotDF["minus"] = np.max(plotDF.loc[:, minList], axis = 1)
        plotDF["max"] = np.max(plotDF.loc[:, maxList], axis = 1)
        # fig = px.line(statsDF.astype(np.float64).melt(id_vars = ["condenseNumber"], value_vars = channelViewList), x = "condenseNumber", y = "value", color = "variable")
        # data variation
        # fig = px.line(plotDF, title = "CLF Confidence", x = "condenseNumber", y = "value", error_y = "minus", error_y_minus = "max", color = "variable")
        # std
        fig = px.line(plotDF, title = "CLF Confidence", x = "condenseNumber", y = "value", error_y = "minus", color = "variable")
        fig.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
        fig.show()
        fig.write_image(saveDir + project_name + "_CLF_Confidence" + saveFormat)
    statsCheck = statsDF.loc[:, channelViewList].astype(np.float64).ge(0.95, axis = 1).all(axis = 1)
    if sum(statsCheck) == 0:
        print("No condense convergence to statisfactory quality")
        return None, statsDF
    else:
        condenseCellsFinal = condenseCellsTry[statsCheck.idxmax()]
        return condenseCellsFinal, statsDF
    
def bootstrapOptimisation(bootstrapsTry, optimiseCondense, condenseCellsTry, condenseCells, condenseNumber, removeLeft, removeLeftBoundary, histRemove, histRange, bootstrap, n_bootstraps, folDir, condList, groupList, channelViewList, maxChannels):
    statsDF = pd.DataFrame(columns = channelViewList)
    statsDF.insert(0, "bootstrapNumber", bootstrapsTry)
    for bootstrapTry in bootstrapsTry:
        if optimiseCondense == True:
            condenseCells = True
            condenseCellsFinal, condenseStats = condenseOptimisation(condenseCellsTry, condenseCells, removeLeft, removeLeftBoundary, histRemove, histRange, True, bootstrapTry, folDir, condList, groupList, channelViewList, maxChannels, plotStats = True)
            statsDF.loc[statsDF["bootstrapNumber"] == bootstrapTry, channelViewList] = condenseStats.loc[condenseStats.loc[:, channelViewList].astype(np.float64).sum(axis = 1).idxmax(), channelViewList].tolist()
        else:
            statsReport = TAS_Analysis(condenseCells, condenseNumber, removeLeft, removeLeftBoundary, histRemove, histRange, True, bootstrapTry, folDir, condList, groupList, channelViewList, maxChannels, plotFigure = False, reStats = True)
            statsDF.loc[statsDF["bootstrapNumber"] == bootstrapTry, channelViewList] = np.reshape(statsReport, (3,2)).T[1,:]
    fig = px.line(statsDF.astype(np.float64).melt(id_vars = ["bootstrapNumber"], value_vars = channelViewList), x = "bootstrapNumber", y = "value", color = "variable")
    fig.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
    fig.show()
    fig.write_image(saveDir + project_name + "_Bootstrap_Confidence" + saveFormat)
    statsCheck = statsDF.loc[:, channelViewList].astype(np.float64).ge(0.95, axis = 1).all(axis = 1)
    if sum(statsCheck) == 0:
        print("No bootstrap convergence to statisfactory quality")
        return None, statsDF
    else:
        bootstrapFinal = bootstrapsTry[statsCheck.idxmax()]
        return bootstrapFinal, statsDF
        
def optimiseCondenseRun(optimiseCondense, condenseCellsTry, condenseCells, condenseNumber, removeLeft, removeLeftBoundary, histRemove, histRange, bootstrap, n_bootstraps, folDir, condList, groupList, channelViewList, maxChannels):
    if optimiseCondense == True:
        condenseCells = True
        condenseCellsFinal, _ = condenseOptimisation(condenseCellsTry, condenseCells, removeLeft, removeLeftBoundary, histRemove, histRange, bootstrap, n_bootstraps, folDir, condList, groupList, channelViewList, maxChannels, plotStats = True)
        if condenseCellsFinal != None:
            TAS_Analysis(True, condenseCellsFinal, removeLeft, removeLeftBoundary, histRemove, histRange, bootstrap, n_bootstraps, folDir, condList, groupList, channelViewList, maxChannels, plotFigure = True, reStats = False)
            print("--- Recommending %s cells to condense ---" % (condenseCellsFinal))
    else:
        TAS_Analysis(condenseCells, condenseNumber, removeLeft, removeLeftBoundary, histRemove, histRange, bootstrap, n_bootstraps, folDir, condList, groupList, channelViewList, maxChannels, plotFigure = True, reStats = False) # TODO: plot figure

if __name__ == "__main__":
    # Comment from here
    start_time = time.time()
    if optimiseBootstrap == True:
        bootstrap = True
        bootstrapFinal, statsDF = bootstrapOptimisation(bootstrapsTry, optimiseCondense, condenseCellsTry, condenseCells, condenseNumber, removeLeft, removeLeftBoundary, histRemove, histRange, bootstrap, n_bootstraps, folDir, condList, groupList, channelViewList, maxChannels)
        if bootstrapFinal != None:
            print("--- Recommending %s bootstraps to run ---" % (bootstrapFinal))
            optimiseCondenseRun(optimiseCondense, condenseCellsTry, condenseCells, condenseNumber, removeLeft, removeLeftBoundary, histRemove, histRange, bootstrap, bootstrapFinal, folDir, condList, groupList, channelViewList, maxChannels)
        else:
            print("No bootstrap convergence to statisfactory quality")
    else:
        bootstrap = False
        optimiseCondenseRun(optimiseCondense, condenseCellsTry, condenseCells, condenseNumber, removeLeft, removeLeftBoundary, histRemove, histRange, bootstrap, n_bootstraps, folDir, condList, groupList, channelViewList, maxChannels)
    print("--- %s seconds ---" % (time.time() - start_time))
    # to here for plotting

    ## Plotting confidence from real BS
    # exlfiles = [file for file in os.listdir(saveDir + project_name) if file.endswith(".xlsx")]
    exlfiles = [file for file in os.listdir(saveDir) if file.endswith(".xlsx")]
    condenseCellsTry.sort()
    colNames = ["Condense Number"]
    colNames.extend([x for xs in [[a, a + "_min", a + "_max"] for a in channelViewList] for x in xs])
    plotData = pd.DataFrame(columns = colNames)
    plotData["Condense Number"] = condenseCellsTry
    for file in exlfiles:
        filename = file.split("_")
        if ((filename[0] == project_name) and (filename[1] == "PCA") and (filename[3] == "Condense")):
        # if ((filename[0] + "_" + filename[1] == project_name) and (filename[2] == "PCA") and (filename[4] == "Condense")):
        #     filename = [filename[0] + "_" + filename[1], filename[2], filename[3], filename[4], filename[5]]
        # if ((filename[1] == "PCA") and (filename[3] == "Condense")):
            # data = pd.read_excel(saveDir + project_name + "/" + file)
            data = pd.read_excel(saveDir + "/" + file)
            plotData.loc[plotData.loc[:, "Condense Number"] == int(filename[4][:-5]), channelViewList[int(filename[2][1:])]] = np.mean(data.iloc[:, -1])
            # SEM
            # plotData.loc[plotData.loc[:, "Condense Number"] == int(filename[4][:-5]), channelViewList[int(filename[2][1:])] + "_min"] = data.iloc[:, -1].sem()
            # 95 percentile
            plotData.loc[plotData.loc[:, "Condense Number"] == int(filename[4][:-5]), channelViewList[int(filename[2][1:])] + "_min"] = np.percentile(data.iloc[:, -1], 2.5)
            plotData.loc[plotData.loc[:, "Condense Number"] == int(filename[4][:-5]), channelViewList[int(filename[2][1:])] + "_max"] = np.percentile(data.iloc[:, -1], 97.5)
            # 95 CI
            # plotData.loc[plotData.loc[:, "Condense Number"] == int(filename[4][:-5]), channelViewList[int(filename[2][1:])] + "_min"] = np.mean(data.iloc[:, -1]) - 2 * np.std(data.iloc[:, -1])
            # plotData.loc[plotData.loc[:, "Condense Number"] == int(filename[4][:-5]), channelViewList[int(filename[2][1:])] + "_max"] = np.mean(data.iloc[:, -1]) + 2 * np.std(data.iloc[:, -1])
    plotDF = []
    for _, row in plotData.iterrows():
        condense_number = row["Condense Number"]

        for var in channelViewList:
            plotDF.append({
                "Condense Number": condense_number,
                "variable": var,
                "value": row[var],
                # SEM
                # "min": row[f"{var}_min"],
                # CI
                "min": row[var] - row[f"{var}_min"],
                "max": row[f"{var}_max"] - row[var]
            })
    plotDF = pd.DataFrame(plotDF)
    fig = px.line(plotDF, title = "Bootstrap Confidence", x = "Condense Number", y = "value", error_y_minus = "min", error_y = "max", color = "variable", range_x = [0, 150])
    fig.update_layout(yaxis_title = "Accuracy")
    # fig = px.line(plotDF, title = "Bootstrap Confidence", x = "Condense Number", y = "value", error_y = "min", color = "variable")
    fig.update_layout(width = 1200, height = 800, font = dict(size = fontSize))
    fig.add_vline(x = 50, line_width = 1, line_dash = "dash", line_color = "red")
    fig.show()
    fig.write_image(saveDir + project_name + "_BS_Confidence" + saveFormat)
    1