import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#Plot the comparison between 2 plots with the name within
#plotting the training and validation curves
def plot2lineGeneral(x1_value,y1_value, leg1, x2_value, y2_value, leg2,
                 xlab,ylab, title, savename):
    """
    The Generic function for plotting 2 lines and comparing these 2
    :param x1_value: x1
    :param y1_value: y1
    :param leg1: legend of 1
    :param x2_value: x2
    :param y2_value: y2
    :param leg2: legend of 2
    :param xlab: label of x axis
    :param ylab: label of y axis
    :param title: title of the graph
    :param savename: saving name of the graph
    :return:
    """
    f=plt.figure()
    plt.title(title)
    plt.plot(x1_value,y1_value,label = leg1)
    plt.plot(x2_value,y2_value,label = leg2)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    f.savefig(savename)

def plotonerun(directory):
    """
    Plot the loss and accuracy of one run for training and testing
    :param directory:
    :return:
    """
    data = []
    for i in range(4):
        data.append(pd.read_csv(os.path.join(directory,i,'.txt'),header=None).values)
    commonX = range(len(data[0]))
    plot2lineGeneral(commonX,data[0],'training',commonX,data[2],'testing',
                     'epoch','loss','loss vs epoch','loss graph for data ' + directory)
    plot2lineGeneral(commonX, data[1], 'training', commonX, data[3], 'testing',
                     'epoch', 'acc', 'acc vs epoch', 'acc graph for data ' + directory)


def plotNcompare(directory1, directory2, plot_index, leg1pref, leg2pref, result_dir = 'results'):
    """
    Plot and compare the 2 lines by specifying the result directory and plot index value
    :param directory1: The directory that contains the 1st line data points
    :param directory2: The directory that contains the 2nd line data points
    :param plot_index: Check the storing for clarification. Usually 0:train_loss 1:train_acc 2:val_loss 3:val_acc
    :param result_dir: The diectory that stores the results dir
    :param leg1pref: prefix of legend 1
    :param leg2pref: prefix of legend 2
    :return:
    """
    data1name = os.path.join(result_dir, directory1, plot_index, '.txt')
    data2name = os.path.join(result_dir, directory2, plot_index, '.txt')
    data1 = pd.read_csv(data1name, header = None)
    data2 = pd.read_csv(data2name, header=None)
    value_name_list = ['train_loss','train_acc','val_loss','val_acc']
    plot2lineGeneral(range(len(data1)), data1.values, leg1pref,
                     range(len(data2)), data2.values, leg2pref,
                     "epoch",value_name_list[plot_index],"comparison of "+ value_name_list[plot_index],
                     "comparison of "+ value_name_list[plot_index] + "between" + directory1 + directory2)
class HMpoint(object):
    """
    This is a HeatMap point class where each object is a point in the heat map
    properties:
    1. BV_loss: best_validation_loss of this run
    2. feature_1: feature_1 value
    3. feature_2: feature_2 value, none is there is no feature 2
    """
    def __init__(self, bv_loss, f1, f2 = None, f1_name = 'f1', f2_name = 'f2'):
        self.bv_loss = bv_loss
        self.feature_1 = f1
        self.feature_2 = f2
        self.f1_name = f1_name
        self.f2_name = f2_name
        #print(type(f1))
    def to_dict(self):
        return {
            self.f1_name: self.feature_1,
            self.f2_name: self.feature_2,
            self.bv_loss: self.bv_loss
        }


def HeatMapBVL(plot_x_name, plot_y_name, title, save_name='HeatMap.png', HeatMap_dir='HeatMap',
               feature_1_name=None, feature_2_name=None,
               heat_value_name='best_validation_loss'):
    """
    Plotting a HeatMap of the Best Validation Loss for a batch of hyperswiping thing
    First, copy those models to a folder called "HeatMap"
    Algorithm: Loop through the directory using os.look and find the parameters.txt files that stores the
    :param HeatMap_dir: The directory where the checkpoint folders containing the parameters.txt files are located
    :param feature_1_name: The name of the first feature that you would like to plot on the feature map
    :param feature_2_name: If you only want to draw the heatmap using 1 single dimension, just leave it as None
    """
    one_dimension_flag = False  # indication flag of whether it is a 1d or 2d plot to plot
    # Check the data integrity
    if (feature_1_name == None):
        print("Please specify the feature that you want to plot the heatmap");
        return
    if (feature_2_name == None):
        one_dimension_flag = True
        print("You are plotting feature map with only one feature, plotting loss curve instead")

    # Get all the parameters.txt running related data and make HMpoint objects
    HMpoint_list = []
    df_list = []  # make a list of data frame for further use
    for subdir, dirs, files in os.walk(HeatMap_dir):
        for file_name in files:
            if (file_name == 'parameters.txt'):
                file_path = os.path.join(subdir, file_name)  # Get the file relative path from
                df = pd.read_csv(file_path, index_col=0)
                # df = df.reset_index()                           #reset the index to get ride of
                print(df.T)
                if (one_dimension_flag):
                    # print(df[[heat_value_name, feature_1_name]])
                    # print(df[heat_value_name][0])
                    # print(df[heat_value_name].iloc[0])
                    df_list.append(df[[heat_value_name, feature_1_name]])
                    HMpoint_list.append(HMpoint(float(df[heat_value_name][0]), eval(str(df[feature_1_name][0])),
                                                f1_name=feature_1_name))
                else:
                    df_list.append(df[[heat_value_name, feature_1_name, feature_2_name]])
                    HMpoint_list.append(HMpoint(float(df[heat_value_name][0]), eval(str(df[feature_1_name][0])),
                                                eval(str(df[feature_2_name][0])), feature_1_name, feature_2_name))

    # Concatenate all the dfs into a single aggregate one for 2 dimensional usee
    df_aggregate = pd.concat(df_list, ignore_index=True, sort=False)
    # print(df_aggregate[heat_value_name])
    # print(type(df_aggregate[heat_value_name]))
    df_aggregate.astype({heat_value_name: 'float'})
    # print(type(df_aggregate[heat_value_name]))
    # df_aggregate = df_aggregate.reset_index()
    print("before transformation:", df_aggregate)
    [h, w] = df_aggregate.shape
    for i in range(h):
        for j in range(w):
            # print(i,j, df_aggregate.iloc[i,j])
            if (isinstance(df_aggregate.iloc[i, j], str)):
                ij_tuple = eval(df_aggregate.iloc[i, j])
                df_aggregate.iloc[i, j] = len(ij_tuple)

    print("after transoformation:", df_aggregate)

    # Change the feature if it is a tuple, change to length of it
    for cnt, point in enumerate(HMpoint_list):
        print("For point {} , it has {} loss, {} for feature 1 and {} for feature 2".format(cnt,
                                                                                            point.bv_loss,
                                                                                            point.feature_1,
                                                                                            point.feature_2))
        assert (isinstance(point.bv_loss, float))  # make sure this is a floating number
        if (isinstance(point.feature_1, tuple)):
            point.feature_1 = len(point.feature_1)
        if (isinstance(point.feature_2, tuple)):
            point.feature_2 = len(point.feature_2)

    f = plt.figure()
    # After we get the full list of HMpoint object, we can start drawing
    if (feature_2_name == None):
        print("plotting 1 dimension HeatMap (which is actually a line)")
        HMpoint_list_sorted = sorted(HMpoint_list, key=lambda x: x.feature_1)
        # Get the 2 lists of plot
        bv_loss_list = []
        feature_1_list = []
        for point in HMpoint_list_sorted:
            bv_loss_list.append(point.bv_loss)
            feature_1_list.append(point.feature_1)
        print("bv_loss_list:", bv_loss_list)
        print("feature_1_list:", feature_1_list)
        # start plotting
        plt.plot(feature_1_list, bv_loss_list, 'o-')
    else:  # Or this is a 2 dimension HeatMap
        print("plotting 2 dimension HeatMap")
        # point_df = pd.DataFrame.from_records([point.to_dict() for point in HMpoint_list])
        df_aggregate = df_aggregate.reset_index()
        df_aggregate.sort_values(feature_1_name, axis=0, inplace=True)
        df_aggregate.sort_values(feature_2_name, axis=0, inplace=True)
        print(df_aggregate)
        point_df_pivot = df_aggregate.reset_index().pivot(feature_1_name, feature_2_name, heat_value_name)
        sns.heatmap(point_df_pivot, vmin=1.24e-3, cmap="YlGnBu")
    plt.xlabel(plot_x_name)
    plt.ylabel(plot_y_name)
    plt.title(title)
    plt.savefig(save_name)

