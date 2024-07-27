"""Importing relevant modules"""

from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import tables_io
from collections import OrderedDict
import qp

def algorithm_flavor(flavor):
    """
    Extract the algorithm name from the flavor string.
    """
    prefix = flavor.split('_')[0]
    return 'fzboost' if prefix == 'fzb' else prefix

def convert_df(test_file):
    """Read test file and convert to pandas DataFrame"""

    def convert_ordereddict_to_dataframe(odict):
        """Convert nested OrderedDict to pandas DataFrame"""
        flat_dict = {}
        for key, value in odict.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        for subsubkey, subsubvalue in subvalue.items():
                            flat_dict[f"{key}_{subsubkey}"] = subsubvalue
                    else:
                        flat_dict[f"{key}_{subkey}"] = subvalue
            else:
                flat_dict[key] = value
        return pd.DataFrame(flat_dict)

    data = tables_io.read(test_file)

    if isinstance(data, OrderedDict):
        if "photometry" in data:
            df = convert_ordereddict_to_dataframe(data["photometry"])
        else:
            df = convert_ordereddict_to_dataframe(data)
    else:
        df = tables_io.convert(data, tables_io.types.PD_DATAFRAME)

    pd.set_option("display.max_columns", None)
    return df

def zrelation(testFile, flavor, selection):
    """
    2-D Histogram: Estimated Redshift vs True Redshift
    """
    reference = convert_df(testFile)

    algorithm = algorithm_flavor(flavor)

    outputFile_path = f'/sdf/data/rubin/shared/pz/projects/roman_rubin_2023/data/{selection}_{flavor}/output_estimate_{algorithm}.hdf5'

    estimate = qp.read(outputFile_path)

    plt.scatter(reference["redshift"], np.squeeze(estimate.ancil["zmode"]), s=1)
    plt.xlabel("True Redshift")
    plt.ylabel("Estimated Redshift")
    plt.title(f"{algorithm}")

    plt.hist2d(reference["redshift"], np.squeeze(estimate.ancil["zmode"]), bins=100, norm="log")
    plt.xlabel("True Redshift")
    plt.ylabel("Estimated Redshift")
    #plt.gca().set_aspect('1.0')
    plt.title(f"{algorithm}")
    
    components = flavor.split('_')
    algorithm_name = components[0]

    if len(components) == 3:
        subdir_name = components[1]
        flavor_name = components[2]
    else:
        raise ValueError("The flavor string does not match the expected format.")

    base_dir = 'resultsShare'
    dir_path = os.path.join(base_dir, algorithm_name, flavor_name, subdir_name)
    os.makedirs(dir_path, exist_ok=True)

    save_path = os.path.join(dir_path, "zrelation.png")
    plt.savefig(save_path)
    plt.show()

def accuracyMag(testFile, flavors, selection, band_name='LSST_obs_g', threshold=0.05):
    """
    Redshift Estimator Accuracy vs Mag for multiple flavors/algorithms 
    """
    import numpy as np
    import pandas as pd
    import qp
    from matplotlib import pyplot as plt
    from rail.utils import catalog_utils
    from rail.core.stage import RailStage
    import tables_io
    import os

    algorithms = [algorithm_flavor(flavor) for flavor in flavors]
    
    if isinstance(flavors, str):
        flavors = [flavors]
    if isinstance(algorithms, str):
        algorithms = [algorithms]

    testFile = convert_df(testFile)

    bins = (17, 20, 21, 22, 23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27, 30)
    testFile["band_bin"] = pd.cut(testFile[band_name], bins)
    
    plt.figure(figsize=(10, 6))

    for flavor, algorithm in zip(flavors, algorithms):
        file_path = f'/sdf/data/rubin/shared/pz/projects/roman_rubin_2023/data/{selection}_{flavor}/output_estimate_{algorithm}.hdf5'

        outputFile = qp.read(file_path)

        testFile['estimated_redshift'] = np.squeeze(outputFile.ancil['zmode'])
        accuracy = testFile.groupby('band_bin').apply(
            lambda x: np.mean(np.abs(x['redshift'] - x['estimated_redshift']) <= threshold)
        )
        error = np.sqrt(accuracy * (1 - accuracy) / testFile['band_bin'].value_counts())
        plt.errorbar(bins[:-1], accuracy, yerr=error, capsize=5, label=flavor)

        components = flavor.split('_')
        algorithm_name = components[0]

        if len(components) == 3:
            subdir_name = components[1]
            flavor_name = components[2]
        else:
            raise ValueError("The flavor string does not match the expected format.")

        base_dir = 'resultsShare'
        dir_path = os.path.join(base_dir, algorithm_name, flavor_name, subdir_name)
        os.makedirs(dir_path, exist_ok=True)

        #save_path = os.path.join(dir_path, "accuracy.png")
        plt.xlabel(f"Magnitude: {band_name}")
        plt.ylabel("Redshift Estimator Accuracy")
        plt.title(f"Comparing {algorithm} Flavors")
        plt.grid()
        plt.legend()
        plt.ylim(0,1)
        #plt.savefig(save_path)
    
    plt.xlabel(f"Magnitude: {band_name}")
    plt.ylabel("Redshift Estimator Accuracy")
    plt.title(f"Comparing Different Flavors and Algorithms")
    plt.grid()
    plt.legend()
    plt.ylim(0,1)
    plt.show()
    
def accuracyColor(reference, flavors, selection, threshold=0.05, color='g-i'):
    """
    Redshift Estimator Accuracy vs g-i color for multiple flavors/algorithms 
    **Allow for different color bands soon**
    """
    import numpy as np
    import pandas as pd
    import qp
    from matplotlib import pyplot as plt
    import os

    algorithms = [algorithm_flavor(flavor) for flavor in flavors]
    if isinstance(flavors, str):
        flavors = [flavors]
    if isinstance(algorithms, str):
        algorithms = [algorithms]

    reference = convert_df(reference)

    band1, band2 = color.split('-')
    reference['color'] = reference[f'LSST_obs_{band1}'] - reference[f'LSST_obs_{band2}']
    reference = reference.dropna(subset=['color', 'redshift'])

    bins = np.linspace(reference['color'].min(), reference['color'].max(), 14)
    reference['color_bin'] = pd.cut(reference['color'], bins)

    plt.figure(figsize=(10, 6))

    mean_accuracies = []

    for flavor, algorithm in zip(flavors, algorithms):
        file_path = f'/sdf/data/rubin/shared/pz/projects/roman_rubin_2023/data/{selection}_{flavor}/output_estimate_{algorithm}.hdf5'
        outputFile = qp.read(file_path)
        
        reference['estimated_redshift'] = np.squeeze(outputFile.ancil['zmode'])

        accuracy = reference.groupby('color_bin').apply(
            lambda x: np.mean(np.abs(x['redshift'] - x['estimated_redshift']) <= threshold)
        )

        error = np.sqrt(accuracy * (1 - accuracy) / reference['color_bin'].value_counts())
        bin_centers = bins[:-1] + np.diff(bins) / 2

        plt.errorbar(bin_centers, accuracy, yerr=error, capsize=5, label=flavor)

        mean_accuracy = accuracy.mean()
        mean_accuracies.append(mean_accuracy)

        components = flavor.split('_')
        algorithm_name = components[0]
        if len(components) == 3:
            subdir_name = components[1]
            flavor_name = components[2]
        else:
            raise ValueError("The flavor string does not match the expected format.")

        base_dir = 'resultsShare'
        dir_path = os.path.join(base_dir, algorithm_name, flavor_name, subdir_name)
        os.makedirs(dir_path, exist_ok=True)

    plt.xlabel(f"{color}")
    plt.ylabel("Redshift Estimator Accuracy")
    plt.title("Redshift Estimator Accuracy Comparison")
    plt.legend()
    plt.grid()
    plt.show()

    return mean_accuracies

def colorspaceBias(testFile, flavors, selection, band_name='LSST_obs_g', threshold=0.05, frac=0.1):
    """
    r-i vs g-r, Filter dataset to points with bias greater than threshold. Downsample remaining dataset by frac. 
    """
    import numpy as np
    import pandas as pd
    import qp
    from matplotlib import pyplot as plt
    from rail.utils import catalog_utils
    from rail.core.stage import RailStage
    import tables_io
    from matplotlib.colors import Normalize, LinearSegmentedColormap

    algorithm = algorithm_flavor(flavors)
    if isinstance(flavors, str):
        flavors = [flavors]

    testFile = convert_df(testFile)
    
    base_path = '/sdf/data/rubin/shared/pz/projects/roman_rubin_2023/data/{selection}_{flavor}/output_estimate_{algorithm}.hdf5'
    file_paths = [base_path.format(selection=selection, flavor=flavor, algorithm=algorithm) for flavor in flavors]    

    outputFiles = [qp.read(file_path) for file_path in file_paths]
    
    algo_names = flavors

    fig, ax = plt.subplots()

    cdict = {
        'red':   ((0.0, 1.0, 1.0),   # At 0, red is 1.0 (full intensity)
                  (0.5, 0.0, 0.0),   # At 0.5, red is 0.0
                  (1.0, 1.0, 1.0)),  # At 1.0, red is 1.0

        'green': ((0.0, 0.0, 0.0),   # At 0, green is 0.0
                  (0.5, 1.0, 1.0),   # At 0.5, green is 1.0 (full intensity)
                  (1.0, 0.0, 0.0)),  # At 1.0, green is 0.0

        'blue':  ((0.0, 0.0, 0.0),   # At 0, blue is 0.0
                  (0.5, 0.0, 0.0),   # At 0.5, blue is 0.0
                  (1.0, 0.0, 0.0))   # At 1.0, blue is 0.0
    }
    cmap = LinearSegmentedColormap('RedGreenRed', cdict)

    for outputFile, flavor in zip(outputFiles, flavors):
        testFile['estimated_redshift'] = np.squeeze(outputFile.ancil['zmode'])
        testFile['g-r'] = testFile['LSST_obs_g'] - testFile['LSST_obs_r']
        testFile['r-i'] = testFile['LSST_obs_r'] - testFile['LSST_obs_i']

        testFile['bias'] = testFile['redshift'] - testFile['estimated_redshift']

        norm = Normalize(vmin=-3, vmax=3)

        filtered_testFile = testFile[(testFile['bias'] < -0.2) | (testFile['bias'] > 0.2)]
        downsampled_testFile = testFile[(testFile['bias'] >= -0.2) & (testFile['bias'] <= 0.2)].sample(frac=frac)
        combined_testFile = pd.concat([filtered_testFile, downsampled_testFile])

        components = flavor.split('_')
        algorithm_name = components[0]
        scatter = ax.scatter(combined_testFile['g-r'], combined_testFile['r-i'], c=combined_testFile['bias'], cmap=cmap, norm=norm, s=1, alpha=0.8)

    cbar = fig.colorbar(scatter, ax=ax, label='Bias')

    ax.set_xlabel("g-r")
    ax.set_ylabel("r-i")
    ax.set_title(f"Color Space Bias: {algorithm}")
    ax.grid()
    plt.show()

def colorspaceCompare(testFile, flavors, selection, band_name='LSST_obs_g', threshold=0.05, comparison_type='compare'):
    """
    Plot r-i vs g-r for the points based on the specified comparison_type:
    - 'compare': algorithm 1 has a bias below threshold and algorithm 2 has bias above threshold
    - 'good': both algorithms have a bias below threshold
    - 'poor': both algorithms have a bias above threshold
    
    Also plots the contour of the unfiltered data
    """
    import numpy as np
    import pandas as pd
    import qp
    from matplotlib import pyplot as plt
    from rail.utils import catalog_utils
    from rail.core.stage import RailStage
    import tables_io
    from scipy.stats import gaussian_kde

    algorithms = [algorithm_flavor(flavor) for flavor in flavors]
    
    if isinstance(flavors, str):
        flavors = [flavors]
    if isinstance(algorithms, str):
        algorithms = [algorithms]

    testFile = convert_df(testFile)
    
    base_path = '/sdf/data/rubin/shared/pz/projects/roman_rubin_2023/data/{selection}_{flavor}/output_estimate_{algorithm}.hdf5'
    file_paths = [base_path.format(selection=selection, flavor=flavor, algorithm=algorithm) 
                  for flavor, algorithm in zip(flavors, algorithms)]

    outputFiles = [qp.read(file_path) for file_path in file_paths]
    
    for outputFile, flavor, algorithm in zip(outputFiles, flavors, algorithms):
        testFile[f'{algorithm}_estimated_redshift'] = np.squeeze(outputFile.ancil['zmode'])
        testFile[f'{algorithm}_bias'] = testFile['redshift'] - testFile[f'{algorithm}_estimated_redshift']

    if comparison_type == 'compare':
        filtered_testFile = testFile[(np.abs(testFile[f'{algorithms[0]}_bias']) < threshold) & (np.abs(testFile[f'{algorithms[1]}_bias']) > threshold)]
        title = f"Estimations in which {algorithms[0]} performs well and {algorithms[1]} performs poorly."
    elif comparison_type == 'good':
        filtered_testFile = testFile[(np.abs(testFile[f'{algorithms[0]}_bias']) < threshold) & (np.abs(testFile[f'{algorithms[1]}_bias']) < threshold)]
        title = f"Estimations in which {algorithms[0]}, {algorithms[1]} both perform well."
    elif comparison_type == 'poor':
        filtered_testFile = testFile[(np.abs(testFile[f'{algorithms[0]}_bias']) > threshold) & (np.abs(testFile[f'{algorithms[1]}_bias']) > threshold)]
        title = f"Estimations in which {algorithms[0]}, {algorithms[1]} both perform poorly."
    else:
        raise ValueError("Invalid comparison type. Choose from 'compare', 'good', or 'poor'.")

    filtered_testFile['g-r'] = filtered_testFile['LSST_obs_g'] - filtered_testFile['LSST_obs_r']
    filtered_testFile['r-i'] = filtered_testFile['LSST_obs_r'] - filtered_testFile['LSST_obs_i']

    fig, ax = plt.subplots()
    ax.scatter(filtered_testFile['g-r'], filtered_testFile['r-i'], s=1, alpha=0.5, label='Filtered Points')

    testFile['g-r'] = testFile['LSST_obs_g'] - testFile['LSST_obs_r']
    testFile['r-i'] = testFile['LSST_obs_r'] - testFile['LSST_obs_i']

    x = testFile['g-r']
    y = testFile['r-i']
    
    k = gaussian_kde([x, y], bw_method=0.3)  # Adjust bandwidth here
    xi, yi = np.mgrid[x.min():x.max():200j, y.min():y.max():200j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    
    ax.contour(xi, yi, zi.reshape(xi.shape), levels=7, colors='k', alpha=0.5, label='Unfiltered Data')

    ax.set_xlabel("g-r")
    ax.set_ylabel("r-i")
    ax.set_title(title)
    ax.grid()
    ax.legend()
    plt.show()

# Functions above have been tested, below need work














def estimator_acc(reference, estimate, band_name, threshold, algo_name):
    """Plot the accuracy of the redshift estimator as a function of chosen magnitude band"""
    reference["estimated_redshift"] = np.squeeze(estimate.ancil["zmode"])

    bins = (17, 20, 21, 22, 23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27, 30)
    reference["band_bin"] = pd.cut(reference[band_name], bins)

    accuracy = reference.groupby("band_bin").apply(
        lambda x: np.mean(np.abs(x["redshift"] - x["estimated_redshift"]) <= threshold)
    )

    error = np.sqrt(accuracy * (1 - accuracy) / reference["band_bin"].value_counts())

    plt.errorbar(bins[:-1], accuracy, yerr=error, capsize=5)
    plt.xlabel(f"Magnitude: {band_name}")
    plt.ylabel("Redshift Estimator Accuracy")
    plt.title(f"{algo_name}")
    plt.grid()
    plt.savefig(f"{algo_name} Estimator Accuracy")
    plt.show()
    return accuracy.mean()


def redshiftstd(reference, estimate, band_name, algo_name):
    """Plot the standard deviation of redshift estimation as function of chosen mag band"""
    reference["estimated_redshift"] = np.squeeze(estimate.ancil["zmode"])

    bins = (17, 20, 21, 22, 23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27, 30)
    reference["band_bin"] = pd.cut(reference[band_name], bins)

    std = reference.groupby("band_bin").apply(
        lambda x: np.std(x["redshift"] - x["estimated_redshift"])
    )
    count = reference.groupby("band_bin").size()

    std_error = std / np.sqrt(count)

    plt.errorbar(bins[:-1], std, yerr=std_error, capsize=5)
    plt.xlabel(f"Magnitude: {band_name}")
    plt.ylabel("Standard Deviation")
    plt.title(f"{algo_name}")
    plt.grid
    plt.savefig(f"{algo_name} STD")
    plt.show()


def redshiftbias(pdtable, pz, band_name, binrange):
    """Plot redshift bias as a function of chose mag band"""
    pdtable["mode"] = np.squeeze(pz.ancil["mode"])

    bins = binrange
    pdtable["band_bin"] = pd.cut(pdtable[band_name], bins)

    bias = pdtable.groupby("band_bin").apply(
        lambda x: np.mean(x["redshift"] - x["mode"])
    )

    plt.plot(bins[:-1], bias)
    plt.xlabel(f"{band_name} Magnitude")
    plt.ylabel("Bias")
    plt.title(f"Bias vs. {band_name} Magnitude")
    plt.grid()


def inform_estimate_z(
    model_name, trainFile, testFile, training_data, test_data, model_dict={}
):
    """Inform model and estimate redshift."""
    import qp
    import time
    from rail.stages import import_and_attach_all
    from rail.estimation.algos.train_z import TrainZInformer, TrainZEstimator
    from rail.estimation.algos.sklearn_neurnet import (
        SklNeurNetInformer,
        SklNeurNetEstimator,
    )
    from rail.estimation.algos.k_nearneigh import (
        KNearNeighInformer,
        KNearNeighEstimator,
    )
    from rail.estimation.algos.flexzboost import FlexZBoostInformer, FlexZBoostEstimator
    from rail.estimation.algos.bpz_lite import BPZliteInformer, BPZliteEstimator
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator
    from rail.estimation.algos.tpz_lite import TPZliteInformer, TPZliteEstimator
    from rail.estimation.algos.lephare import LephareInformer, LephareEstimator

    ALL_ALGORITHMS = {
        "train_z": {"Inform": TrainZInformer, "Estimate": TrainZEstimator},
        "simplenn": {"Inform": SklNeurNetInformer, "Estimate": SklNeurNetEstimator},
        "knn": {"Inform": KNearNeighInformer, "Estimate": KNearNeighEstimator},
        "fzboost": {"Inform": FlexZBoostInformer, "Estimate": FlexZBoostEstimator},
        "bpz": {"Inform": BPZliteInformer, "Estimate": BPZliteEstimator},
        "gpz": {"Inform": GPzInformer, "Estimate": GPzEstimator},
        "tpz": {"Inform": TPZliteInformer, "Estimate": TPZliteEstimator},
        "lephare": {"Inform": LephareInformer, "Estimate": LephareEstimator},
    }

    InformerClass = ALL_ALGORITHMS[f"{model_name}"]["Inform"]
    EstimatorClass = ALL_ALGORITHMS[f"{model_name}"]["Estimate"]

    inform_dict = model_dict.get('inform', {})
    estimate_dict = model_dict.get('estimate', {})

    start_time = time.time()

    pz_train = InformerClass.make_stage(
        name=f"inform_{model_name}", model=f"demo_{model_name}.pkl", hdf5_groupname = '', **inform_dict
    )
    pz_train.config

    pz_train.inform(training_data)

    modelhandle = pz_train.get_handle("model")  # This is the data that it trained
    modelhandle.path

    pz = EstimatorClass.make_stage(
        name=f"{model_name}",
        hdf5_groupname='',
        model=pz_train.get_handle("model"), **estimate_dict,
    )
    results = pz.estimate(test_data)
    results = pz.get_handle("output")
    results.data
    results.data.npdf
    results.path
    output = qp.read(results.path)
    output

    end_time = time.time()
    total_time = end_time - start_time
    return output, total_time

def analyze_output(testFile, flavors, selection, algorithm, band_name='LSST_obs_g', threshold=0.05):
    """
    Analyze Parameter Outputs based on different flavors and a specified algorithm
    """
    import numpy as np
    import pandas as pd
    import qp
    from matplotlib import pyplot as plt
    from rail.utils import catalog_utils
    from rail.core.stage import RailStage
    import tables_io
    
    if isinstance(flavors, str):
        flavors = [flavors]

    # Convert the test file data frame if necessary
    testFile = convert_df(testFile)
    
    # Generate file paths
    base_path = '/sdf/data/rubin/shared/pz/projects/roman_rubin_2023/data/{selection}_{flavor}/output_estimate_{algorithm}.hdf5'
    file_paths = [base_path.format(selection=selection, flavor=flavor, algorithm=algorithm) for flavor in flavors]    

    # Load output files
    outputFiles = [qp.read(file_path) for file_path in file_paths]
    
    algo_names = flavors

    # Define bins
    bins = (17, 20, 21, 22, 23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27, 30)
    testFile["band_bin"] = pd.cut(testFile[band_name], bins)

    # Calculate accuracy for each flavor and plot results
    for outputFile, flavor in zip(outputFiles, flavors):
        testFile['estimated_redshift'] = np.squeeze(outputFile.ancil['zmode'])
        accuracy = testFile.groupby('band_bin').apply(
            lambda x: np.mean(np.abs(x['redshift'] - x['estimated_redshift']) <= threshold)
        )
        error = np.sqrt(accuracy * (1 - accuracy) / testFile['band_bin'].value_counts())
        plt.errorbar(bins[:-1], accuracy, yerr=error, capsize=5, label=flavor)
        components = flavor.split('_')
        algorithm_name = components[0]

        # Ensure components have the expected length
        if len(components) == 3:
            subdir_name = components[1]
            flavor_name = components[2]
        else:
            raise ValueError("The flavor string does not match the expected format.")

        base_dir = 'resultsShare'
        # Create the directory if it doesn't exist, including subdir
        dir_path = os.path.join(base_dir, algorithm_name, flavor_name, subdir_name)
        os.makedirs(dir_path, exist_ok=True)

        # Save the figure
        save_path = os.path.join(dir_path, "accuracy.png")
        plt.xlabel(f"Magnitude: {band_name}")
        plt.ylabel("Redshift Estimator Accuracy")
        plt.title(f"Comparing {algorithm} Flavors")
        plt.grid()
        plt.legend()
        plt.ylim(0,1)
        plt.savefig(save_path)

    plt.xlabel(f"Magnitude: {band_name}")
    plt.ylabel("Redshift Estimator Accuracy")
    plt.title(f"Comparing {algorithm} Flavors")
    plt.grid()
    plt.legend()
    plt.ylim(0,1)
    plt.show()


truth_template = "/sdf/data/rubin/shared/pz/projects/roman_rubin_2023/data/{selection}_{flavor}/true_NZ_true_nz_{classifier}_bin{ibin}.hdf5"
est_template = "/sdf/data/rubin/shared/pz/projects/roman_rubin_2023/data/{selection}_{flavor}/single_NZ_summarize_{classifier}_bin{ibin}_{summarizer}.hdf5"

colors = {
    0: 'blue',
    1: 'green',
    2: 'red',
    3: 'cyan',
    4: 'black',
}


def compare_bins_true(selection, flavor, classifier='uniform_binning', summarizer='naive_stack'):
    fig = plt.figure()
    grid = np.linspace(0, 3., 301)
    for ibin in range(5):
        truth_file = truth_template.format(selection=selection, flavor=flavor, classifier=classifier, ibin=ibin)
        est_file = est_template.format(selection=selection, flavor=flavor, classifier=classifier, summarizer=summarizer, ibin=ibin)
        truth_ens = qp.read(truth_file)
        est_ens = qp.read(est_file)
        
        plt.plot(grid, np.squeeze(truth_ens.pdf(grid)), color=colors[ibin], linestyle='dashed')
        #plt.plot(grid, np.squeeze(est_ens.pdf(grid)), color=colors[ibin])
        
    plt.grid()
    return fig

def compare_bins_est(selection, flavor, classifier='uniform_binning', summarizer='naive_stack'):
    fig = plt.figure()
    grid = np.linspace(0, 3., 301)
    for ibin in range(5):
        truth_file = truth_template.format(selection=selection, flavor=flavor, classifier=classifier, ibin=ibin)
        est_file = est_template.format(selection=selection, flavor=flavor, classifier=classifier, summarizer=summarizer, ibin=ibin)
        truth_ens = qp.read(truth_file)
        est_ens = qp.read(est_file)
        
        #plt.plot(grid, np.squeeze(truth_ens.pdf(grid)), color=colors[ibin], linestyle='dashed')
        plt.plot(grid, np.squeeze(est_ens.pdf(grid)), color=colors[ibin])
        
    plt.grid()
    return fig

def compare_bins_all(selection, flavor, classifier, summarizer):
    fig = plt.figure()
    grid = np.linspace(0, 3., 301)
    for ibin in range(5):
        truth_file = truth_template.format(selection=selection, flavor=flavor, classifier=classifier, ibin=ibin)
        est_file = est_template.format(selection=selection, flavor=flavor, classifier=classifier, summarizer=summarizer, ibin=ibin)
        truth_ens = qp.read(truth_file)
        est_ens = qp.read(est_file)
        
        plt.plot(grid, np.squeeze(truth_ens.pdf(grid)), color=colors[ibin], linestyle='dashed')
        plt.plot(grid, np.squeeze(est_ens.pdf(grid)), color=colors[ibin])
        
    plt.grid()
    components = flavor.split('_')
    algorithm_name = components[0]
    flavor_name = f"{components[2]}"
    subdir_name = f"{components[1]}"

    base_dir = 'resultsShare'
    # Create the directory if it doesn't exist
    dir_path = os.path.join(base_dir, algorithm_name, flavor_name, subdir_name)
    os.makedirs(dir_path, exist_ok=True)

    # Save the figure
    save_path = os.path.join(dir_path, "tomography.png")
    plt.savefig(save_path)
    return fig

def tomographic_bias(selection, flavor, classifier='uniform_binning', summarizer='naive_stack'):
    biases = []
    bin_numbers = range(5)
    for ibin in bin_numbers:
        truth_file = truth_template.format(selection=selection, flavor=flavor, classifier=classifier, ibin=ibin)
        est_file = est_template.format(selection=selection, flavor=flavor, classifier=classifier, summarizer=summarizer, ibin=ibin)
        truth_ens = qp.read(truth_file)
        est_ens = qp.read(est_file)
        
        truth_mean = np.mean(truth_ens.mean(), axis=0)
        est_mean = np.mean(est_ens.mean(), axis=0)
        
        bias = truth_mean - est_mean
        biases.append(bias)
    
    biases = np.array(biases).reshape(-1)
    
    fig, ax = plt.subplots()
    ax.plot(bin_numbers, biases, marker='o')
    ax.set_xlabel('Bin Number')
    ax.set_ylabel('Bias (Delta n(z))')
    ax.set_title(f'{flavor}: Bias within Bins')
    ax.set_xticks(np.arange(1, 4 + 1, 1))
    plt.grid()
    components = flavor.split('_')
    algorithm_name = components[0]

    # Ensure components have the expected length
    if len(components) == 3:
        subdir_name = components[1]
        flavor_name = components[2]
    else:
        raise ValueError("The flavor string does not match the expected format.")

    base_dir = 'resultsShare'
    # Create the directory if it doesn't exist, including subdir
    dir_path = os.path.join(base_dir, algorithm_name, flavor_name, subdir_name)
    os.makedirs(dir_path, exist_ok=True)

    # Save the figure
    save_path = os.path.join(dir_path, "tomographicBias.png")
    plt.savefig(save_path)
    return fig, biases

def compare_bins_separated(selection, flavor, classifier='uniform_binning', summarizer='naive_stack'):
    fig, axs = plt.subplots(5, 1, figsize=(8, 20), sharex=True)
    grid = np.linspace(0, 3., 301)
    
    for ibin, ax in enumerate(axs):
        truth_file = truth_template.format(selection=selection, flavor=flavor, classifier=classifier, ibin=ibin)
        est_file = est_template.format(selection=selection, flavor=flavor, classifier=classifier, summarizer=summarizer, ibin=ibin)
        truth_ens = qp.read(truth_file)
        est_ens = qp.read(est_file)
        
        ax.plot(grid, np.squeeze(truth_ens.pdf(grid)), color=colors[ibin], linestyle='dashed', label=f'Truth Bin {ibin}')
        ax.plot(grid, np.squeeze(est_ens.pdf(grid)), color=colors[ibin], label=f'Est Bin {ibin}')
        ax.grid()
        ax.legend()
        
    #axs[-1].set_xlabel('Redshift')
    #fig.supylabel('Density')
    plt.tight_layout()
    return fig

def output_metric(testFile, flavor, selection, algorithm, band_name='LSST_obs_g'):
    """
    From output evaluate file, create plot of mean of values as a function of magnitude.
    """
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    import qp

    if isinstance(testFile, str):
        testFile = convert_df(testFile)

    outputFile_path = f'/sdf/data/rubin/shared/pz/projects/roman_rubin_2023/data/{selection}_{flavor}/output_evaluate_{algorithm}.hdf5'

    #outputFile = qp.read(outputFile_path)
    outputFile = convert_df(outputFile_path)

    testFile['output_point_stats_ez_zmode_redshift'] = np.squeeze(outputFile['point_stats_ez_zmode_redshift'])

    bins = (17, 20, 21, 22, 23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27, 30)
    testFile["band_bin"] = pd.cut(testFile[band_name], bins)

    mean_values = testFile.groupby('band_bin')['output_point_stats_ez_zmode_redshift'].mean()

    plt.plot(bins[:-1], mean_values, marker='o', label=f'{algorithm} - {flavor}')
    plt.xlabel(f"{band_name}")
    plt.ylabel("output_point_stats_ez_zmode_redshift")
    components = flavor.split('_')
    algorithm_name = components[0]

    # Ensure components have the expected length
    if len(components) == 3:
        subdir_name = components[1]
        flavor_name = components[2]
    else:
        raise ValueError("The flavor string does not match the expected format.")

    base_dir = 'resultsShare'
    # Create the directory if it doesn't exist, including subdir
    dir_path = os.path.join(base_dir, algorithm_name, flavor_name, subdir_name)
    os.makedirs(dir_path, exist_ok=True)

    # Save the figure
    save_path = os.path.join(dir_path, "outputMetric.png")
    plt.title(f"{algorithm_name}: {flavor} Output Metric")
    plt.grid()
    y_min = min(mean_values.min(), -mean_values.max())
    y_max = max(mean_values.max(), -mean_values.min())
    plt.ylim(y_min, y_max)

    plt.axhline(0, color='black', linewidth=0.8)
    
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def output_metric2(testFile, flavors, selection, algorithms, band_name='LSST_obs_g'):
    """
    From output evaluate file, create plot of mean of values as a function of magnitude.
    """
    if isinstance(testFile, str):
        testFile = convert_df(testFile)
    
    # Define bins
    bins = (17, 20, 21, 22, 23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27, 30)
    testFile["band_bin"] = pd.cut(testFile[band_name], bins)
    
    plt.figure(figsize=(10, 6))
    
    for flavor, algorithm in zip(flavors, algorithms):
        outputFile_path = f'/sdf/data/rubin/shared/pz/projects/roman_rubin_2023/data/{selection}_{flavor}/output_evaluate_{algorithm}.hdf5'
        print(f"Checking file: {outputFile_path}")

        if not os.path.exists(outputFile_path):
            print(f"File not found: {outputFile_path}")
            continue
        
        outputFile = convert_df(outputFile_path)
        print(f"Columns in output file: {outputFile.columns}")

        if 'point_stats_ez_zmode_redshift' not in outputFile.columns:
            print(f"Column 'point_stats_ez_zmode_redshift' not found in file: {outputFile_path}")
            continue

        testFile['output_point_stats_ez_zmode_redshift'] = np.squeeze(outputFile['point_stats_ez_zmode_redshift'])
        mean_values = testFile.groupby('band_bin')['output_point_stats_ez_zmode_redshift'].mean()
        print(f"Mean values for {algorithm} - {flavor}: {mean_values}")

        plt.plot(bins[:-1], mean_values, marker='o', label=f'{algorithm} - {flavor}')

    plt.xlabel(f"{band_name}")
    plt.ylabel("output_point_stats_ez_zmode_redshift")

    plt.title(f"Output Metric for Multiple Algorithms and Flavors")
    plt.grid()
    plt.axhline(0, color='black', linewidth=0.8)
    plt.legend()

    try:
        y_min = min(testFile.groupby('band_bin')['output_point_stats_ez_zmode_redshift'].mean().min(), 
                    -testFile.groupby('band_bin')['output_point_stats_ez_zmode_redshift'].mean().max())
        y_max = max(testFile.groupby('band_bin')['output_point_stats_ez_zmode_redshift'].mean().max(), 
                    -testFile.groupby('band_bin')['output_point_stats_ez_zmode_redshift'].mean().min())
        plt.ylim(y_min, y_max)
    except KeyError as e:
        print(f"Error in calculating y-axis limits: {e}")

    base_dir = 'resultsShare'
    algorithm_name = '_'.join(algorithms)
    flavor_name = '_'.join(flavors)
    subdir_name = selection

    # Create the directory if it doesn't exist, including subdir
    dir_path = os.path.join(base_dir, algorithm_name, flavor_name, subdir_name)
    os.makedirs(dir_path, exist_ok=True)

    # Save the figure
    save_path = os.path.join(dir_path, "outputMetric.png")
    plt.savefig(save_path)
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def all(testFile, flavor, selection, algorithm, summarizer='point_est_hist', band_name='LSST_obs_g', threshold=0.05, tom_option='all'):
    """
    Run the pipeline with the specified options.
    
    Arguments:
    testFile -- Path to the test file
    flavor -- The flavor to use
    selection -- The selection to use
    algorithm -- The algorithm to use
    band_name -- The band name to use
    threshold -- The threshold to use for analyze_output (default 0.05)
    tom_option -- The option to choose for compare_bins ('all', 'est', 'true', 'separated')
    
    Returns:
    A list of figures
    """
    classifier = f'{algorithm}_uniform_binning'
    figures = []

    fig5 = output_metric(testFile, flavor, selection, algorithm)
    figures.append(fig5)
    
    fig1 = zrelation(testFile, flavor, selection, algorithm)
    figures.append(fig1)
    
    fig2 = analyze_output(testFile, [flavor], selection, algorithm, band_name, threshold)
    figures.append(fig2)
    
    if tom_option == 'all':
        fig3 = compare_bins_all(selection, flavor, classifier, summarizer)
    elif tom_option == 'est':
        fig3 = compare_bins_est(selection, flavor, classifier, summarizer)
    elif tom_option == 'true':
        fig3 = compare_bins_true(selection, flavor, classifier, summarizer)
    elif tom_option == 'separated':
        fig3 = compare_bins_separated(selection, flavor, classifier, summarizer)
    figures.append(fig3)

    fig4 = tomographic_bias(selection, flavor, classifier, summarizer)

    figures.append(fig4)
    
    return figures

