import numpy as np
def resampleData(data, windowLen, stepLen, type:str="DEPTH"):
    """
    Resample the dataset for 3D CNN training.
    :param data: an array of data to resample
    :param windowLen: width of new window
    :param stepLen: distance between consecutive windows
    :param type: type of resample, either only resample the depth or the whole dataset.
    :return:
    """
    def resampleWindow(data, windowLen, stepLen):
        featureWindow = []
        for start in range(0, len(data) - windowLen + 1, stepLen):
            # get the current window
            window = data[start:start + windowLen]
            rms = np.sqrt(np.mean(np.square(window)))
            featureWindow.append(rms)
        return featureWindow
    def resampleRaw(data, windowLen, stepLen):
        pass
    if type.upper() == "DEPTH":
        resampledData = resampleWindow(data, windowLen, stepLen)

    else:
        assert "Other resample functions are still under construction."
    return resampledData