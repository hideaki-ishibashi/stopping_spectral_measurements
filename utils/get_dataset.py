import numpy as np
import random
import pandas as pd


def get_dataset(data_name):
    if data_name == "Mn2+":
        dataset = np.loadtxt("./dataset/simulation/Mn2+.xy")
        input = dataset[:, 0][:, None]
        output = dataset[:, 1]
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies], output[indecies]]

    elif data_name == "Co2+":
        dataset = np.loadtxt("./dataset/simulation/Co2+.xy")
        input = dataset[:, 0][:, None]
        output = dataset[:, 1]
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies], output[indecies]]

    elif data_name == "Ni2+":
        dataset = np.loadtxt("./dataset/simulation/Ni2+.xy")
        input = dataset[:, 0][:, None]
        output = dataset[:, 1]
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies], output[indecies]]

    elif data_name == "MnO2_1_exp":
        data_dir = "dataset/experiment/"
        data = pd.read_csv(data_dir + "MnO2_1", header=7,
                           names=["Target", "Energy", "Time", "Ch0", "Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7",
                                  "Ch4_corr"])
        input = np.array(data.Energy)[:, None]
        output = np.array(data.Ch3 / data.Ch2)
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies], output[indecies]]

    elif data_name == "MnO2_4_exp":
        data_dir = "dataset/experiment/"
        data = pd.read_csv(data_dir + "MnO2_4", header=7,
                           names=["Target", "Energy", "Time", "Ch0", "Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7",
                                  "Ch4_corr"])
        input = np.array(data.Energy)[:, None]
        output = np.array(data.Ch3 / data.Ch2)
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies], output[indecies]]

    elif data_name == "MnO2_7_exp":
        data_dir = "dataset/experiment/"
        data = pd.read_csv(data_dir + "MnO2_7", header=7,
                           names=["Target", "Energy", "Time", "Ch0", "Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7",
                                  "Ch4_corr"])
        input = np.array(data.Energy)[:, None]
        output = np.array(data.Ch3 / data.Ch2)
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies], output[indecies]]

    elif data_name == "Co_2_exp":
        data_dir = "dataset/experiment/"
        data = pd.read_csv(data_dir + "Co_2", header=7,
                           names=["Target", "Energy", "Time", "Ch0", "Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7",
                                  "Ch4_corr"])
        input = np.array(data.Energy)[:, None]
        output = np.array(data.Ch3 / data.Ch2)
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies], output[indecies]]

    elif data_name == "Co_3_exp":
        data_dir = "dataset/experiment/"
        data = pd.read_csv(data_dir + "Co_3", header=7,
                           names=["Target", "Energy", "Time", "Ch0", "Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7",
                                  "Ch4_corr"])
        input = np.array(data.Energy)[:, None]
        output = np.array(data.Ch3 / data.Ch2)
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies], output[indecies]]

    elif data_name == "Co_5_exp":
        data_dir = "dataset/experiment/"
        data = pd.read_csv(data_dir + "Co_5", header=7,
                           names=["Target", "Energy", "Time", "Ch0", "Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7",
                                  "Ch4_corr"])
        input = np.array(data.Energy)[:, None]
        output = np.array(data.Ch3 / data.Ch2)
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies], output[indecies]]

    elif data_name == "Ni_3_exp":
        data_dir = "dataset/experiment/"
        data = pd.read_csv(data_dir + "Ni_3", header=7,
                           names=["Target", "Energy", "Time", "Ch0", "Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7",
                                  "Ch4_corr"])
        input = np.array(data.Energy)[:, None]
        output = np.array(data.Ch3 / data.Ch2)
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies], output[indecies]]

    elif data_name == "Ni_4_exp":
        data_dir = "dataset/experiment/"
        data = pd.read_csv(data_dir + "Ni_4", header=7,
                           names=["Target", "Energy", "Time", "Ch0", "Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7",
                                  "Ch4_corr"])
        input = np.array(data.Energy)[:, None]
        output = np.array(data.Ch3 / data.Ch2)
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies], output[indecies]]

    elif data_name == "Ni_6_exp":
        data_dir = "dataset/experiment/"
        data = pd.read_csv(data_dir + "Ni_6", header=7,
                           names=["Target", "Energy", "Time", "Ch0", "Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7",
                                  "Ch4_corr"])
        input = np.array(data.Energy)[:, None]
        output = np.array(data.Ch3 / data.Ch2)
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies], output[indecies]]
