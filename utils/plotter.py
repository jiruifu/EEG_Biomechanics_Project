import os
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd


def log_iter_result(data, exp_name="exp1", mode:str="LC", smoothWindow: int = 10):
    """
    Plot and save the training result and smooth the curve if necessary
    :param data: the training result save in a dict
    :param path: path to save the figure
    :param exp_name: name of the experiment
    :param smoothWindow: length of window to smooth the result
    :param smoothOrder: order of polynomial to smooth the result
    :return: None
    """

    def smooth(data, box_pts):
        box = np.ones(box_pts) / box_pts
        data = np.convolve(data, box, mode='same')
        return data

    def save_data_to_csv(data: dict = None, exp_name: str = "exp", path: str = "result"):
        """
        Used to save a single dictionary as csv file.
        """
        csvName = exp_name+"_result.csv"
        fname = os.path.join(path, csvName)
        df_save = pd.DataFrame(data)
        df_save.to_csv(fname, index=False)
        print("\rThe training data is saved to a CSV file!")

    def save_plot_to_pdf(path, plot:list=[]):
        ensure_directory_exists(path)
        for _, fig in enumerate(plot):
            figName = fig[0]
            figPlot = fig[1]
            fName = os.path.join(path, figName)
            line_fig = figPlot.get_figure()
            line_fig.savefig(fName)
            print("\rThe figure:{}, is saved".format(figName))

    def smooth_curve(y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def ensure_directory_exists(directory_path):
        """
        Check if a directory exists. If not, create it.
        Args:
        directory_path (str): The path of the directory to check/create.
        """
        if not os.path.exists(directory_path):
            try:
                os.makedirs(directory_path)
                print(f"Directory created successfully: {directory_path}")
            except OSError as e:
                print(f"Error creating directory {directory_path}: {e}")
        else:
            print(f"Directory already exists: {directory_path}")

    def plot_learning_result(data: dict = None, exp_name: str = "exp1", smooth: bool=True, smoothWin: int=2):
        plot = []
        plotName = []
        if isinstance(data, dict):
            num = 0
            # Iterate over the
            for i, result in data.items():
                if len(result) == 0:
                    pass
                else:
                    num += 1
                    figName = exp_name + "_plot_" + i + ".pdf"
                    resultArray = np.array(result)
                    Steps = np.arange(1, len(result) + 1)
                    # Smooth the result using savgol filter
                    if smooth:
                        smoothedResult = smooth_curve(resultArray, smoothWin)
                    else:
                        smoothedResult = resultArray
                    d = {'Epochs': Steps, i: smoothedResult}
                    pdnum = pd.DataFrame(d)
                    title_list = list(pdnum)
                    plt.figure(num)
                    sns.set_theme(style="darkgrid")
                    fig = sns.lineplot(x=title_list[0], y=title_list[1], data=pdnum)
                    plot.append(fig)
                    plotName.append(figName)
            outputFig = zip(plotName, plot)

        else:
            raise NotImplementedError("To plot the result, the result must be a dictionary")

        return outputFig

    #Check if the directory to log result is existed
    currentDir = currentDir = os.path.dirname(os.path.abspath(__file__))
    root_plot = os.path.join(currentDir, "figure")
    ensure_directory_exists(root_plot)
    root_csv = os.path.join(currentDir, "result")
    ensure_directory_exists(root_csv)
    save_data_to_csv(data=data, exp_name=exp_name, path=root_csv)
    plot = plot_learning_result(data=data, exp_name=exp_name, smoothWin=smoothWindow)
    save_plot_to_pdf(path=root_plot, plot=plot)
    print(f"The data logging is done, the plots are saved to: {root_plot}, the results are saved to: {root_csv}")


