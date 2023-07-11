import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import CSS4_COLORS as colors
import matplotlib.colors as mcolors
import random
import os

colorOptions = []
badColors = []

by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                    name)
                for name, color in colors.items())
names = [name for hsv, name in by_hsv]

for option in names:
    if not any(bad in option for bad in badColors):
        colorOptions.append(option)

def get_random_color_set(num):
    i = random.randint(0, len(colorOptions) - (num + 1))
    colorSet = []
    for j in range(num):
        colorSet.append(colorOptions[i+j])
    return colorSet

def load_data(fileName):
    df = pd.read_csv(fileName)
    return df

def plot_success_bars_traj():
    path = '/home/liam/success_rate_data/'
    baseFile = path + '2022-10-19 11:20:10.670330_packed_force_aware_transfer_base.csv'
    base_df = load_data(baseFile)
    forceFile = path + '2022-10-19 08:37:09.815544_packed_force_aware_transfer_rne.csv'
    force_df = load_data(forceFile)

    base_counts = []
    force_counts = []
    force_counts.append

    mass = force_df["MassPerObject"][0]
    force_counts.append(len((force_df[(force_df['Solved']==True) & (force_df['TorquesExceded']==True)]["Solved"])))
    force_counts[0] += len((force_df[(force_df['Solved']==False)]["Solved"]))
    force_counts.append(len((force_df[(force_df['Solved']==True) & (force_df['TorquesExceded']==False)]["Solved"])))

    base_counts.append(len((base_df[(base_df['Solved']==True) & (base_df['TorquesExceded']==True)]["Solved"])))
    base_counts[0] += len((base_df[(base_df['Solved']==False)]["Solved"]))
    base_counts.append(len((base_df[(base_df['Solved']==True) & (base_df['TorquesExceded']==False)]["Solved"])))


    plotdata = pd.DataFrame({
    "RNE":force_counts,
    "BaseLine":base_counts})
    data_set_size = sum(force_counts)
    plotdata.plot(kind="bar")
    plt.title(f"Success Per Run Distribution {mass}kg Over {data_set_size} Runs")
    plt.xticks([0,1], ["Failure", "Success"])
    plt.show()




def plot_success_bars_traj_multi():
    path = '/home/liam/success_rate_data/'
    novFiles = {3: path + '2022-10-23 10:41:34.265279_packed_force_aware_transfer_nov.csv',
                4: path + '2022-10-23 08:29:23.699993_packed_force_aware_transfer_nov.csv',
                5 : path + '2022-10-22 09:02:55.112166_packed_force_aware_transfer_nov.csv',
                6 : path + '2022-10-25 18:34:47.296309_packed_force_aware_transfer_nov.csv'}
    baseFiles = {3 : path + '2022-10-19 18:13:24.638753_packed_force_aware_transfer_base.csv',
                 4 : path + '2022-10-20 15:36:09.716424_packed_force_aware_transfer_base.csv',
                 5 : path + '2022-10-25 07:25:35.534641_packed_force_aware_transfer_base.csv',
                 6 : path + '2022-10-25 10:56:26.690561_packed_force_aware_transfer_base.csv'}
    forceFiles = {3 : path + '2022-10-19 20:00:56.614030_packed_force_aware_transfer_rne.csv',
                  4 : path + '2022-10-21 10:53:05.525938_packed_force_aware_transfer_rne.csv',
                  5 : path + '2022-10-24 13:41:01.311165_packed_force_aware_transfer_rne.csv',
                  6 : path + '2022-10-24 19:28:18.378828_packed_force_aware_transfer_rne.csv'}
    n = 3
    if len(forceFiles) != len(baseFiles):
        print(f'File length mismatch: {len(forceFiles)} != {len(baseFiles)}')
        return
    data = []
    x_ticks = []
    allColors = []
    for m in baseFiles:
        colorSet = get_random_color_set(n)
        allColors += colorSet
        base_df = load_data(baseFiles[m])
        force_df = load_data(forceFiles[m])
        nov_df = load_data(novFiles[m])

        base_counts = 0
        force_counts = 0
        nov_counts = 0

        mass = force_df["MassPerObject"][0]

        force_counts = len((force_df[(force_df['Solved']==True) & (force_df['TorquesExceded']==False)]["Solved"]))
        base_counts = len((base_df[(base_df['Solved']==True) & (base_df['TorquesExceded']==False)]["Solved"]))
        nov_counts = len((nov_df[(nov_df['Solved']==True) & (nov_df['TorquesExceded']==False)]["Solved"]))
        x_ticks.append(f'RNE {m}kg')
        x_ticks.append(f'Baseline {m}kg')
        x_ticks.append(f'Static RNE {m}kg')
        data.append(force_counts)
        data.append(base_counts)
        data.append(nov_counts)

    data_set_size = len(force_df["Solved"])
    plt.bar(x_ticks, data, color=allColors)
    plt.title(f"Success Per Run Distribution Over {data_set_size} Runs")
    plt.xticks(range(len(data)), x_ticks, rotation='vertical')
    plt.show()

def plot_success_bars_dist_multi():
    path = '/home/liam/success_rate_dist_data_random/'
    novFiles = {3: path + ' 2022-11-01 11:42:52.595715_packed_force_aware_transfer_nov_0.3/success_data.csv',
                4: path + ' 2022-11-01 12:45:27.687496_packed_force_aware_transfer_nov_0.4/success_data.csv',}
                # 5 : path + ' 2022-10-31 11:34:58.882061_packed_force_aware_transfer_nov_0.5/success_data.csv',
                # 6 : path + ' 2022-10-31 14:57:12.675759_packed_force_aware_transfer_nov_0.6/success_data.csv'}
    baseFiles = {3 : path + ' 2022-11-01 08:49:54.223851_packed_force_aware_transfer_base_0.3/success_data.csv',
                 4 : path + ' 2022-11-01 13:30:13.659539_packed_force_aware_transfer_base_0.4/success_data.csv',}
                #  5 : path + ' 2022-10-31 11:06:22.773531_packed_force_aware_transfer_base_0.5/success_data.csv',
                #  6 : path + ' 2022-10-31 14:22:51.299952_packed_force_aware_transfer_base_0.6/success_data.csv'}
    forceFiles = {3 : path + ' 2022-11-01 09:32:45.710267_packed_force_aware_transfer_rne_0.3/success_data.csv',
                 4 : path + ' 2022-11-01 14:01:07.278073_packed_force_aware_transfer_rne_0.4/success_data.csv',}
                #  5 : path + ' 2022-10-31 12:13:07.049462_packed_force_aware_transfer_rne_0.5/success_data.csv',
                #  6 : path + ' 2022-10-31 13:02:32.596480_packed_force_aware_transfer_rne_0.6/success_data.csv'}
    n = 3

    if len(forceFiles) != len(baseFiles):
        print(f'File length mismatch: {len(forceFiles)} != {len(baseFiles)}')
        return
    data = []
    x_ticks = []
    allColors = []
    for m in baseFiles:
        colorSet = get_random_color_set(n)
        allColors += colorSet
        base_df = load_data(baseFiles[m])
        force_df = load_data(forceFiles[m])
        nov_df = load_data(novFiles[m])
        data_set_size = len(force_df["Solved"])
        base_counts = 0
        force_counts = 0
        nov_counts = 0

        mass = force_df["MassPerObject"][0]

        force_counts = len((force_df[(force_df['Solved']==True) & (force_df['TorquesExceded']==False)]["Solved"]))/data_set_size
        base_counts = len((base_df[(base_df['Solved']==True) & (base_df['TorquesExceded']==False)]["Solved"]))/data_set_size
        nov_counts = len((nov_df[(nov_df['Solved']==True) & (nov_df['TorquesExceded']==False)]["Solved"]))/data_set_size
        x_ticks.append(f'RNE 0.{m}m')
        x_ticks.append(f'Baseline 0.{m}m')
        x_ticks.append(f'Static RNE 0.{m}m')
        data.append(force_counts)
        data.append(base_counts)
        data.append(nov_counts)

    data_set_size = len(force_df["Solved"])
    plt.bar(x_ticks, data, color=allColors)
    plt.title(f"Success Per Run Distribution Over {data_set_size} Runs")
    plt.xticks(range(len(data)), x_ticks, rotation='vertical')
    plt.show()

def plot_success_bars_ik():
    baseFile = '/home/liam/exp_data/2022-09-10_11:36:43.301316_packed.csv'
    base_df = load_data(baseFile)
    forceFile = '/home/liam/exp_data/2022-09-10_12:17:21.972933_packed_force_aware.csv'
    force_df = load_data(forceFile)
    base_counts = [0]*2
    force_counts = [0]*2
    ip_counts = []
    rc_counts = []
    # for i in range(2):
    #     numDelivered = bool(i)
    #     force_counts.append(len(force_df[(force_df['Solved'].astype('bool')==True) & (force_df['SuccessfulDeliveries'].astype('bool')==numDelivered) & (force_df['TrayOnTable'].astype('bool')==True) &  (force_df['TorquesExceded'].astype('bool')==False)]["SuccessfulDeliveries"]))
    #     base_counts.append(len(base_df[ (base_df['Solved']==True) & (base_df['SuccessfulDeliveries']==numDelivered) & (base_df['TrayOnTable']==True) &  (base_df['TorquesExceded']==False)]["SuccessfulDeliveries"]))
    force_counts[1] += len(force_df[(force_df['Solved'].astype('bool')==True) &  (force_df['TorquesExceded']=='False')]["SuccessfulDeliveries"])
    force_counts[0] += len((force_df[ (force_df['Solved']==True) & (force_df['TorquesExceded']=='True')]))
    base_counts[1] += len(base_df[ (base_df['Solved']==True) & (base_df['SuccessfulDeliveries'].astype('bool')==True) & (base_df['TrayOnTable']==True) &  (base_df['TorquesExceded']=='False')]["SuccessfulDeliveries"])
    base_counts[0] += len((base_df[(base_df['Solved']==True) & (base_df['TorquesExceded']=='True')]))
    # force_counts[0] += len((force_df[ (force_df['Solved']==False)]))
    # base_counts[0] += len((base_df[(base_df['Solved']==False)]))
    print('te')
    print(type(force_df['TorquesExceded'][0]))
    print(force_df.head())
    plotdata = pd.DataFrame({
    "Birrt with Collisions + Force Limits":force_counts,
    "Birrt with Collisions":base_counts})

    plotdata.plot(kind="bar")
    plt.title("Trajectory Planning without Exceding Torque Limits 2kg")
    plt.xlabel("Success")
    plt.ylabel("Number of Trials")
    plt.show()

def npz_to_torque_df(file, labels, groups):
    data = np.load(file)
    dfs = []
    for group in groups:
        torque_data = data[group]
        df = pd.DataFrame(torque_data, columns=labels)
        dfs.append(df)
    return dfs

def npz_to_torque_array(file, labels, groups):
    data = np.load(file)
    dfs = []
    for group in groups:
        torque_data = data[group]
        dfs.append(torque_data)
    return dfs


torque_axes = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,1]]

def plot_torque_data_diff_traj():
    max_limits = [87,87,87,87,12,12,12]
    base_path = '/home/liam/success_rate_mass_data_random/'
    rne_dir = base_path+ ' 2022-11-11 09:38:36.213329_packed_force_aware_transfer_arne_11kg/2022-11-11 09:38:36.213329_trajectory_data_0.npz'
    base_dir = base_path + ' 2022-11-11 09:50:19.361690_packed_force_aware_transfer_base_11kg/2022-11-11 09:50:19.361690_trajectory_data_0.npz'
    rbt_dir = base_path + ' 2022-11-11 09:49:55.178752_packed_force_aware_transfer_dyn_11kg/2022-11-11 09:49:55.178752_trajectory_data_0.npz'
    nov_dir = base_path + ' 2022-11-11 09:50:56.884137_packed_force_aware_transfer_nov_11kg/2022-11-11 09:50:56.884137_trajectory_data_0.npz'
    NUM_COLS = 7
    MASS = 11
    Distance = 0.5
    colLabels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
    force_df = npz_to_torque_df(rne_dir, colLabels, ["zz"])[0]
    dyn_df = npz_to_torque_df(rbt_dir, colLabels, ["zz"])[0]
    base_df = npz_to_torque_df(base_dir, colLabels, ["zz"])[0]
    nov_df = npz_to_torque_df(nov_dir, colLabels, ["zz"])[0]

    force_label = labels_from_keys['arne']
    base_label = labels_from_keys['base']
    nov_label = labels_from_keys['nov']
    dyn_label = labels_from_keys['dyn']

    X = range(max(len(force_df), len(base_df), len(nov_df), len(dyn_df)))
    # Initialise the subplot function using number of rows and columns
    force_df = extend_df(len(X), force_df, NUM_COLS, colLabels)
    base_df = extend_df(len(X), base_df, NUM_COLS, colLabels)
    nov_df = extend_df(len(X), nov_df, NUM_COLS, colLabels)
    dyn_df = extend_df(len(X), dyn_df, NUM_COLS, colLabels)
    figure, axis = plt.subplots(3, 3)
    print(force_df['J1'])
    for i in range(len(colLabels)):
        a = axis[torque_axes[i][0], torque_axes[i][1]]
        plot_one_torque(a, X, force_df, colLabels[i], force_label)
        plot_one_torque(a, X, base_df, colLabels[i], base_label)
        plot_one_torque(a, X, dyn_df, colLabels[i], dyn_label)
        plot_one_torque(a, X, nov_df, colLabels[i], nov_label)
        a.set_title(colLabels[i])
        a.legend()
        a.plot(X, [max_limits[i]]*len(X))
        a.plot(X, [-max_limits[i]]*len(X))
        a.set_xlabel('Index of Robot State In Trajectory')
        a.set_ylabel('Torque Output at Robot State')
    figure.suptitle(f'Joint Torques Over One Run {MASS}kg at distance {Distance}m')
    plt.legend()
    
    plt.show()

def plot_one_torque(axis, x, data, key, label):
    axis.plot(x, data[key].to_list(), label = label)

def plot_torque_data_same_traj():
    max_limits = [87,87,87,87,12,12,12]
    forceFile = '/home/liam/success_rate_dist_data_random/ 2022-11-11 13:41:36.361311_packed_force_aware_transfer_base_0.5/2022-11-11 13:41:36.361311_trajectory_data_0.npz'
    NUM_COLS = 7
    MASS = 8
    Distance = 0.5
    colLabels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
    force_df, base_df = npz_to_torque_df(forceFile, colLabels, ["zz", "yy"])
    force_label = 'RNE'
    base_label = 'Robot Equation'
    X = range(max(len(force_df), len(base_df)))
    # Initialise the subplot function using number of rows and columns
    force_df = extend_df(len(X), force_df, NUM_COLS, colLabels)
    base_df = extend_df(len(X), base_df, NUM_COLS, colLabels)
    figure, axis = plt.subplots(3, 3)

    axis[0, 0].plot(X, force_df['J1'], label = force_label)
    axis[0, 0].plot(X, base_df['J1'], label = base_label)
    # axis[0, 0].plot(X, [max_limits[0]]*len(X))
    # axis[0, 0].plot(X, [-max_limits[0]]*len(X))
    axis[0, 0].set_title("Joint1")
    axis[0, 0].legend()

    axis[0, 1].plot(X, force_df['J2'], label = force_label)
    axis[0, 1].plot(X, base_df['J2'], label = base_label)
    axis[0, 1].plot(X, [max_limits[1]]*len(X))
    axis[0, 1].plot(X, [-max_limits[1]]*len(X))
    axis[0, 1].set_title("Joint2")
    axis[0, 1].legend()

    axis[0, 2].plot(X, force_df['J3'], label = force_label)
    axis[0, 2].plot(X, base_df['J3'], label = base_label)
    axis[0, 2].plot(X, [max_limits[2]]*len(X))
    axis[0, 2].plot(X, [-max_limits[2]]*len(X))
    axis[0, 2].set_title("Joint3")
    axis[0, 2].legend()

    axis[1, 0].plot(X, force_df['J4'], label = force_label)
    axis[1, 0].plot(X, base_df['J4'], label = base_label)
    axis[1, 0].plot(X, [max_limits[3]]*len(X))
    axis[1, 0].plot(X, [-max_limits[3]]*len(X))
    axis[1, 0].set_title("Joint4")
    axis[1, 0].legend()

    axis[1, 1].plot(X, force_df['J5'], label = force_label)
    axis[1, 1].plot(X, base_df['J5'], label = base_label)
    axis[1, 1].plot(X, [max_limits[4]]*len(X))
    axis[1, 1].plot(X, [-max_limits[4]]*len(X))
    axis[1, 1].set_title("Joint5")
    axis[1, 1].legend()

    axis[1, 2].plot(X, force_df['J6'], label = force_label)
    axis[1, 2].plot(X, base_df['J6'], label = base_label)
    axis[1, 2].plot(X, [max_limits[5]]*len(X))
    axis[1, 2].plot(X, [-max_limits[5]]*len(X))
    axis[1, 2].set_title("Joint6")

    axis[2, 1].plot(X, force_df['J7'], label = force_label)
    axis[2, 1].plot(X, base_df['J7'], label = base_label)
    # axis[2, 1].plot(X, [max_limits[6]]*len(X))
    # axis[2, 1].plot(X, [-max_limits[6]]*len(X))
    axis[2, 1].set_title("Joint7")
    figure.suptitle(f'Joint Torques Over One Run {MASS}kg at distance {Distance}m')
    plt.legend()
    plt.show()

def extend_df(size, df, cols, colLabels = None):
    hold = {}
    if colLabels is None:
        colLabels =  ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
    for label in colLabels:
        hold[label] = [0.0]
    for _ in range(size - len(df)):
        df2 = pd.DataFrame(hold)
        df = pd.concat([df, df2], ignore_index = True, axis = 0)
    return df

def calc_error_stats(directory, keys, labels):
    rne_df = None
    dyn_df = None
    for file in os.listdir(directory):
        if file.endswith('.npz'):
            path = directory + file
            rne_i, dyn_i = npz_to_torque_array(path, None, keys)
            if rne_df is None:
                rne_df = rne_i
                dyn_df = dyn_i
            else:
                rne_df = np.concatenate((rne_df, rne_i))
                dyn_df = np.concatenate((dyn_df, dyn_i))
    diff = np.subtract(rne_df, dyn_df)
    diff = np.linalg.norm(diff, axis=1)
    print("average error: ", np.average(diff))
    print("standard deviation: ", diff.std())

labels_from_keys = {
    'dyn': 'Robot Equation',
    'arne': 'RNE',
    'base': 'No Torque Checking',
    'nov': 'Static RNE'
}

def plot_dist_success_dist_dir():
    dir_name = '/home/liam/success_rate_dist_data_hiro_sim_2/'
    dist_file_name = 'success_data.csv'
    mass = 8
    data = {}
    for d in os.listdir(dir_name):
        bits = d.split('_')
        dist = float(bits[-1])
        method = bits[-2]
        if method not in data:
            data[method] = {}
        df = load_data(dir_name + d + "/" + dist_file_name)
        data[method][dist] = len(df[(df['Solved']==True) & df["TorquesExceded"]==False]) / len(df['Solved'])

    x_s = sorted([key for key in data['arne']])
    ys = []
    data2 = {}
    for key in data:
        data2[key] = []
        for x in x_s:
            data2[key].append(data[key][x] * 100)

    for key in data2:
        plt.plot(x_s, data2[key], label=labels_from_keys[key])
    plt.legend()
    plt.title(f'Success Rate vs Distance of Object Randomized Start Position from the Base of the Robot {mass}kg')
    plt.xlabel('Distance From Robot Base (m)')
    plt.ylabel('Percent of Successful Runs (%)')
    plt.show()

def plot_dist_success_mass_dir():
    dir_name = '/home/liam/success_rate_mass_data_hiro_sim_2/'
    mass_file_name = 'success_data.csv'
    DIST = 0.5
    data = {}
    for d in os.listdir(dir_name):
        bits = d.split('_')
        mass = bits[-1]
        if int(bits[-1][:-2]) < 10:
            mass = '0' + mass
        method = bits[-2]
        if method not in data:
            data[method] = {}
        df = load_data(dir_name + d + "/" + mass_file_name)
        data[method][mass] = len(df[(df['Solved']==True) & df["TorquesExceded"]==False]) / len(df['Solved'])

    x_s = sorted([key for key in data['arne']])
    ys = []
    data2 = {}
    for key in data:
        data2[key] = []
        for x in x_s:
            data2[key].append(data[key][x] * 100)

    for key in data2:
        plt.plot(x_s, data2[key], label=labels_from_keys[key])
    plt.legend()
    plt.title(f'Success Rate vs Object Mass with Randomized Start Position at Distance {DIST}m from the Base of the Robot')
    plt.xlabel('Mass of Object (kg)')
    plt.ylabel('Percent of Successful Runs (%)')
    plt.show()

def plot_solution_time_dist():
    dir_name = '/home/liam/success_rate_mass_data_hiro_sim/'
    mass_file_name = 'success_data.csv'
    DIST = 0.5
    data = {}
    for d in os.listdir(dir_name):
        bits = d.split('_')
        mass = bits[-1]
        if int(bits[-1][:-2]) < 10:
            mass = '0' + mass
        method = bits[-2]
        if method not in data:
            data[method] = {}
         
        df = load_data(dir_name + d + "/" + mass_file_name)
        data[method][10] = len(df[(df[''])])

def plot_torque():
    MASS = '6'
    NUM_COLS = 7
    max_limits = [87,87,87,87,12,12,12]
    path = '/home/liam/exp_data/'
    forceFile = path + '2022-10-16_09:09:25.306405_torque_data_rne_6.0kg.csv'
    baseFile = path + '2022-10-16_09:18:09.628155_torque_data_base_6.0kg.csv'
    dynFile = path + '2022-10-16_09:20:36.627254_torque_data_dyn_6.0kg.csv'
    force_df = load_data(forceFile)
    base_df = load_data(baseFile)
    dyn_df = load_data(dynFile)

    X = range(max(len(force_df), len(base_df), len(dyn_df)))
    force_df.columns = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
    base_df.columns = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
    dyn_df.columns = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
    colLabels =  ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
    force_df = extend_df(len(X), force_df, NUM_COLS, colLabels)
    base_df = extend_df(len(X), base_df, NUM_COLS, colLabels)
    dyn_df = extend_df(len(X), dyn_df, NUM_COLS, colLabels)

    figure, axis = plt.subplots(2, 3)

    axis[0, 0].plot(X, force_df['J1'], label = 'RNE')
    axis[0, 0].plot(X, base_df['J1'], label = 'Baseline')
    axis[0, 0].plot(X, dyn_df['J1'], label = 'Dynamics Eq')
    axis[0, 0].plot(X, [max_limits[0]]*len(X))
    axis[0, 0].plot(X, [-max_limits[0]]*len(X))
    axis[0, 0].set_title("Joint1")
    axis[0, 0].legend()

    axis[0, 1].plot(X, force_df['J2'], label = 'RNE')
    axis[0, 1].plot(X, base_df['J2'], label = 'Baseline')
    axis[0, 1].plot(X, dyn_df['J2'], label = 'Dynamics Eq')
    axis[0, 1].plot(X, [max_limits[1]]*len(X))
    axis[0, 1].plot(X, [-max_limits[1]]*len(X))
    axis[0, 1].set_title("Joint2")
    axis[0, 1].legend()

    axis[0, 2].plot(X, force_df['J3'], label = 'RNE')
    axis[0, 2].plot(X, base_df['J3'], label = 'Baseline')
    axis[0, 2].plot(X, dyn_df['J3'], label = 'Dynamics Eq')
    axis[0, 2].plot(X, [max_limits[2]]*len(X))
    axis[0, 2].plot(X, [-max_limits[2]]*len(X))
    axis[0, 2].set_title("Joint3")
    axis[0, 2].legend()

    axis[1, 0].plot(X, force_df['J4'], label = 'RNE')
    axis[1, 0].plot(X, base_df['J4'], label = 'Baseline')
    axis[1, 0].plot(X, dyn_df['J4'], label = 'Dynamics Eq')
    axis[1, 0].plot(X, [max_limits[3]]*len(X))
    axis[1, 0].plot(X, [-max_limits[3]]*len(X))
    axis[1, 0].set_title("Joint4")
    axis[1, 0].legend()

    axis[1, 1].plot(X, force_df['J5'], label = 'RNE')
    axis[1, 1].plot(X, base_df['J5'], label = 'Baseline')
    axis[1, 1].plot(X, dyn_df['J5'], label = 'Dynamics Eq')
    axis[1, 1].plot(X, [max_limits[4]]*len(X))
    axis[1, 1].plot(X, [-max_limits[4]]*len(X))
    axis[1, 1].set_title("Joint5")
    axis[1, 1].legend()

    axis[1, 2].plot(X, force_df['J6'], label = 'RNE')
    axis[1, 2].plot(X, base_df['J6'], label = 'Baseline')
    axis[1, 2].plot(X, dyn_df['J6'], label = 'Dynamics Eq')
    axis[1, 2].plot(X, [max_limits[5]]*len(X))
    axis[1, 2].plot(X, [-max_limits[5]]*len(X))
    axis[1, 2].set_title("Joint6")

    # axis[2, 1].plot(X, force_df['J7'], label = 'RNE')
    # axis[2, 1].plot(X, base_df['J7'], label = 'Baseline')
    # axis[2, 1].plot(X, [max_limits[5]]*len(X))
    # axis[2, 1].plot(X, [-max_limits[5]]*len(X))
    # axis[2, 1].set_title("Joint7")

    figure.suptitle(f'Joint Torques Over One Run {MASS}kg')
    plt.legend()
    plt.show()

def npz_to_torque_array(file, labels, groups):
    data = np.load(file)
    dfs = []
    for group in groups:
        torque_data = data[group]
        dfs.append(torque_data)
    return dfs

def plot_one_arr(axis, x, data, key, label):
    axis.plot(x, data[key], label = label)

def plot_velocity():
    MASS = '8.5'
    NUM_COLS = 7
    max_limits = [87,87,87,87,12,12,12]
    forceFile = f'/home/liam/success_rate_dist_data_hiro_sim/ 2022-11-15 17:44:03.073950_packed_force_aware_transfer_HIRO_arne_0.4/2022-11-15 17:44:03.073950_trajectory_data_0.npz'
    columns = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
    vel_df = npz_to_torque_array(forceFile, columns , ['qd'])[0]
    dist_df = npz_to_torque_array(forceFile, columns, ['q'])[0]
    dist_diff_arr = np.array([[0.0]*len(columns)] * vel_df.shape[0])
    print(dist_diff_arr.shape)
    print(dist_df.shape)
    for i in range(1, dist_diff_arr.shape[0]):
        dist_diff_arr[i] = np.subtract(dist_df[i], dist_df[i-1])
    X = range(vel_df.shape[0])
    figure, axis = plt.subplots(3, 3)
    for i in range(len(columns)):
        a = axis[torque_axes[i][0], torque_axes[i][1]]
        plot_one_arr(a, X, dist_diff_arr.T, i, "Delta q")
        plot_one_arr(a, X, vel_df.T, i, 'velocity')
        a.set_title(columns[i])
        a.legend()
    figure.suptitle('Joint Velocity and Difference between qi+1 - qi')
    plt.show()

def plot_single_torque():
    MASS = '8.5'
    METHOD = 'base'
    NUM_COLS = 7
    max_limits = [87,87,87,87,12,12,12]
    data_path = "/home/liam/exp_data/"
    # forceFile = data_path + f'torque_data_{METHOD}_{MASS}kg.csv'
    forceFile = data_path + "2022-10-13_16:19:07.568984_torque_data_base_8.5kg.csv"
    force_df = load_data(forceFile)

    X = range(len(force_df))
    force_df.columns = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']

    figure, axis = plt.subplots(2, 3)

    axis[0, 0].plot(X, force_df['J1'], label = METHOD)
    axis[0, 0].plot(X, [max_limits[0]]*len(X))
    axis[0, 0].plot(X, [-max_limits[0]]*len(X))
    axis[0, 0].set_title("Joint1")
    axis[0, 0].legend()

    axis[0, 1].plot(X, force_df['J2'], label = METHOD)
    axis[0, 1].plot(X, [max_limits[1]]*len(X))
    axis[0, 1].plot(X, [-max_limits[1]]*len(X))
    axis[0, 1].set_title("Joint2")
    axis[0, 1].legend()

    axis[0, 2].plot(X, force_df['J3'], label = METHOD)
    axis[0, 2].plot(X, [max_limits[2]]*len(X))
    axis[0, 2].plot(X, [-max_limits[2]]*len(X))
    axis[0, 2].set_title("Joint3")
    axis[0, 2].legend()

    axis[1, 0].plot(X, force_df['J4'], label = METHOD)
    axis[1, 0].plot(X, [max_limits[3]]*len(X))
    axis[1, 0].plot(X, [-max_limits[3]]*len(X))
    axis[1, 0].set_title("Joint4")
    axis[1, 0].legend()

    axis[1, 1].plot(X, force_df['J5'], label = METHOD)
    axis[1, 1].plot(X, [max_limits[4]]*len(X))
    axis[1, 1].plot(X, [-max_limits[4]]*len(X))
    axis[1, 1].set_title("Joint5")
    axis[1, 1].legend()

    axis[1, 2].plot(X, force_df['J6'], label = METHOD)
    axis[1, 2].plot(X, [max_limits[5]]*len(X))
    axis[1, 2].plot(X, [-max_limits[5]]*len(X))
    axis[1, 2].set_title("Joint6")

    # axis[2, 1].plot(X, force_df['J7'], label = METHOD)
    # axis[2, 1].plot(X, base_df['J7'], label = 'Baseline')
    # axis[2, 1].plot(X, [max_limits[5]]*len(X))
    # axis[2, 1].plot(X, [-max_limits[5]]*len(X))
    # axis[2, 1].set_title("Joint7")

    figure.suptitle(f'Joint Torques Over One Run {MASS}kg')
    plt.legend()
    plt.show()

def plot_single_torque_and_velocity():
    MASS = '6'
    METHOD = 'base'
    NUM_COLS = 7
    max_limits = [87,87,87,87,12,12,12]
    data_path = "/home/liam/exp_data/"
    # forceFile = data_path + f'torque_data_{METHOD}_{MASS}kg.csv'
    forceFile = data_path + "2022-10-16_09:18:09.628155_torque_data_base_6.0kg.csv"
    velFile = data_path + "2022-10-16_09:18:09.628155_velocity_data_base_6.0kg.csv"
    force_df = load_data(forceFile)
    vel_df = load_data(velFile)
    X = range(len(force_df))
    velMult = 1
    force_df.columns = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']

    vel_df.columns = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', "J8", "J9"]

    figure, axis = plt.subplots(2, 3)

    axis[0, 0].plot(X, force_df['J1'], label = METHOD)
    axis[0, 0].plot(X, vel_df['J1']*velMult, label = "Velocity")
    axis[0, 0].plot(X, [max_limits[0]]*len(X))
    axis[0, 0].plot(X, [-max_limits[0]]*len(X))
    axis[0, 0].set_title("Joint1")
    axis[0, 0].legend()

    axis[0, 1].plot(X, force_df['J2'], label = METHOD)
    axis[0, 1].plot(X, vel_df['J2']*velMult, label = "velocity")
    axis[0, 1].plot(X, [max_limits[1]]*len(X))
    axis[0, 1].plot(X, [-max_limits[1]]*len(X))
    axis[0, 1].set_title("Joint2")
    axis[0, 1].legend()

    axis[0, 2].plot(X, force_df['J3'], label = METHOD)
    axis[0, 2].plot(X, vel_df['J3']*velMult, label = "velocity")
    axis[0, 2].plot(X, [max_limits[2]]*len(X))
    axis[0, 2].plot(X, [-max_limits[2]]*len(X))
    axis[0, 2].set_title("Joint3")
    axis[0, 2].legend()

    axis[1, 0].plot(X, force_df['J4'], label = METHOD)
    axis[1, 0].plot(X, vel_df['J4']*velMult, label = "velocity")
    axis[1, 0].plot(X, [max_limits[3]]*len(X))
    axis[1, 0].plot(X, [-max_limits[3]]*len(X))
    axis[1, 0].set_title("Joint4")
    axis[1, 0].legend()

    axis[1, 1].plot(X, force_df['J5'], label = METHOD)
    axis[1, 1].plot(X, vel_df['J5']*velMult, label = "velocity")
    axis[1, 1].plot(X, [max_limits[4]]*len(X))
    axis[1, 1].plot(X, [-max_limits[4]]*len(X))
    axis[1, 1].set_title("Joint5")
    axis[1, 1].legend()

    axis[1, 2].plot(X, force_df['J6'], label = METHOD)
    axis[1, 2].plot(X, vel_df['J6']*velMult, label = "velocity")
    axis[1, 2].plot(X, [max_limits[5]]*len(X))
    axis[1, 2].plot(X, [-max_limits[5]]*len(X))
    axis[1, 2].set_title("Joint6")

    # axis[2, 1].plot(X, force_df['J7'], label = METHOD)
    # axis[2, 1].plot(X, base_df['J7'], label = 'Baseline')
    # axis[2, 1].plot(X, [max_limits[5]]*len(X))
    # axis[2, 1].plot(X, [-max_limits[5]]*len(X))
    # axis[2, 1].set_title("Joint7")

    figure.suptitle(f'Joint Torques Over One Run {MASS}kg')
    plt.legend()
    plt.show()

def plot_eoat_velocity():
    MASS = '8.5'
    METHOD = 'base'
    eoatVelocityBase = f'/home/liam/exp_data/eoat_velocity_data_base_{MASS}kg.csv'
    eoatVelocityRne = f'/home/liam/exp_data/eoat_velocity_data_rne_{MASS}kg.csv'
    eoatVelocityDyn = f'/home/liam/exp_data/eoat_velocity_data_dyn_{MASS}kg.csv'
    evb_df = load_data(eoatVelocityBase)
    evr_df = load_data(eoatVelocityRne)
    evd_df = load_data(eoatVelocityDyn)
    X = range(max(len(evb_df), len(evd_df), len(evr_df)))
    colLabels = ['x', 'y', 'z', 'w', 'p', 'r']
    evb_df.columns = ['x', 'y', 'z', 'w', 'p', 'r']
    evr_df.columns = ['x', 'y', 'z', 'w', 'p', 'r']
    evd_df.columns = ['x', 'y', 'z', 'w', 'p', 'r']
    evb_df = extend_df(len(X), evb_df, 6, colLabels)
    evr_df = extend_df(len(X), evr_df, 6, colLabels)
    evd_df = extend_df(len(X), evd_df, 6, colLabels)
    print(colLabels)
    linvelsB = []
    for index, row in evb_df.iterrows():
        velocity = math.sqrt( row["x"]**2 +row["y"]**2 + row["z"]**2)
        linvelsB.append(velocity)
    linvelsD = []
    for index, row in evd_df.iterrows():
        velocity = math.sqrt( row["x"]**2 +row["y"]**2 + row["z"]**2)
        linvelsD.append(velocity)
    linvelsR = []
    for index, row in evr_df.iterrows():
        velocity = math.sqrt( row["x"]**2 +row["y"]**2 + row["z"]**2)
        linvelsR.append(velocity)
    plt.plot(X, linvelsB, label="Base")
    plt.plot(X, linvelsR, label="Rne")
    plt.plot(X, linvelsD, label="Dynamics Eq")

    plt.title(f"Eoat Velocity over 1 run with {MASS}kg")
    plt.xlabel("Robot Configuration during task")
    plt.ylabel("Eoat Velocity (m/s)")
    plt.legend()
    plt.show()

def plot_eoat_velocity3d():
    MASS = '8.5'
    METHOD = 'base'
    eoatVelocityBase = f'/home/liam/exp_data/eoat_velocity_data_base_{MASS}kg.csv'
    eoatVelocityRne = f'/home/liam/exp_data/eoat_velocity_data_rne_{MASS}kg.csv'
    eoatVelocityDyn = f'/home/liam/exp_data/eoat_velocity_data_dyn_{MASS}kg.csv'
    evb_df = load_data(eoatVelocityBase)
    evr_df = load_data(eoatVelocityRne)
    evd_df = load_data(eoatVelocityDyn)
    X = range(max(len(evb_df), len(evd_df), len(evr_df)))
    colLabels = ['x', 'y', 'z', 'w', 'p', 'r']
    evb_df.columns = ['x', 'y', 'z', 'w', 'p', 'r']
    evr_df.columns = ['x', 'y', 'z', 'w', 'p', 'r']
    evd_df.columns = ['x', 'y', 'z', 'w', 'p', 'r']
    evb_df = extend_df(len(X), evb_df, 6, colLabels)
    evr_df = extend_df(len(X), evr_df, 6, colLabels)
    evd_df = extend_df(len(X), evd_df, 6, colLabels)

    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    ax.plot3D(evb_df["x"], evb_df["y"], evb_df["z"], label="Base")
    ax.plot3D(evr_df["x"], evr_df["y"], evr_df["z"], label="RNE")
    ax.plot3D(evd_df["x"], evd_df["y"], evd_df["z"], label="Dynamics Eq")
    plt.title(f"Eoat Velocity over 1 run with {MASS}kg")
    ax.set_xlabel('X velocity (m/s)')
    ax.set_ylabel('Y velocity (m/s)')
    ax.set_zlabel('Z velocity (m/s)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_dist_success_mass_dir()
    # plot_success_bars_ik()
    # plot_velocity()
    # directory = '/home/liam/success_rate_dist_data_random/ 2022-11-08 16:19:18.561285_packed_force_aware_transfer_arne_0.3/'
    # keys = ['zz', 'yy']
    # labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
    # calc_error_stats(directory, keys, labels)
