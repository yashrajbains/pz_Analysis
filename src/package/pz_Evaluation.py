"""Importing modules"""

from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import tables_io
from collections import OrderedDict
import qp
import h5py

def algorithm_flavor(flavor):
    """
    Extract the algorithm name from the flavor string. 
    Necessary due to the flavor naming conventions I used: {algorithm}_{flavorConfigs}_{specSelection}
    """
    prefix = flavor.split('_')[0]
    return 'fzboost' if prefix == 'fzb' else prefix

def convert_df(testFile):
    """Convert string testFile path to pandas DataFrame"""

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

    data = tables_io.read(testFile)

    if isinstance(data, OrderedDict):
        if "photometry" in data:
            df = convert_ordereddict_to_dataframe(data["photometry"])
        else:
            df = convert_ordereddict_to_dataframe(data)
    else:
        df = tables_io.convert(data, tables_io.types.PD_DATAFRAME)

    pd.set_option("display.max_columns", None)
    return df

def createEnsemble(testFile, flavor, selection, zmin=0, zmax=6, nzbins=601):
    """
    Create p(z) ensemble from sompz output files. Save to path for compatibility with functions. 
    - testFile - string. Path to testing file used.
    - flavor - string. Flavor name
    - selection - string. Selection used (i.e. 'maglim_25.5')
    """
    widepath = f'/sdf/data/rubin/shared/pz/roman_rubin_2023/data/{selection}_{flavor}/wide_data_assignment_estimate_sompz.hdf5'
    pz_chat_path = f'/sdf/data/rubin/shared/pz/roman_rubin_2023/data/{selection}_{flavor}/pz_chat_estimate_sompz.hdf5'
    
    wide = convert_df(widepath)
    testing = convert_df(testFile)

    testing['cells'] = wide['cells']

    grid = np.linspace(zmin, zmax, nzbins)

    # Initialize list to store estimated redshifts
    with h5py.File(pz_chat_path, 'r') as f:
        pz_chat_data = f['pz_chat']
        estimated_redshifts = []
        for cell in testing['cells']:
            cell_data = pz_chat_data[int(cell)][:]
            estimated_redshift = grid[np.argmax(cell_data)]
            estimated_redshifts.append(estimated_redshift)
        testing['estimated_redshift'] = estimated_redshifts
        estimated_redshifts = np.array(estimated_redshifts)

    # Reshape array to 2D 
    if estimated_redshifts.ndim == 1:
        estimated_redshifts = estimated_redshifts[:, np.newaxis]

    # Redshift grid for histograms
    z_grid = np.linspace(zmin, zmax, nzbins)

    # Create histograms for each estimated redshift pdf
    histograms = []
    for redshift in estimated_redshifts:
        hist, _ = np.histogram(redshift, bins=z_grid, density=True)
        histograms.append(hist)

    histograms = np.array(histograms)

    # Create ensemble
    ensemble = qp.Ensemble(gen_func=qp.hist, data=dict(bins=z_grid, pdfs=histograms))

    # Assign zmode values for each object 
    zmode_values = estimated_redshifts.max(axis=1)
    ensemble.set_ancil(dict(zmode=zmode_values))

    # Saving the ensemble to the path expected by other functions
    ensemble.write_to(f'/sdf/data/rubin/shared/pz/roman_rubin_2023/data/{selection}_{flavor}/output_estimate_som.hdf5')
    
    return ensemble

def zrelation(testFile, flavor, selection):
    """
    2-D Histogram: Estimated Redshift vs True Redshift
    - testFile - string. Path to testing file used
    - flavor - string. Flavor name
    - selection - string. Selection used (i.e. 'maglim_25.5')
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
    - testFile - string. Path to testing file used
    - flavors - list. Flavor names
    - selection - string. Selection used (i.e. 'maglim_25.5')
    - band_name - string, optional. Column name for color filter, default is g band. 
    - threshold - float, optional. Threshold for accuracy calculation. Default is 0.05
    """
    if isinstance(flavors, str):
        flavors = [flavors]

    algorithms = [algorithm_flavor(flavor) for flavor in flavors]

    testFile = convert_df(testFile)

    # Creating magnitude bins. I used variable bin sizes due to the magnitude distribution in the testing file 
    bins = (17, 20, 21, 22, 23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27, 30)
    testFile["band_bin"] = pd.cut(testFile[band_name], bins)
    
    plt.figure(figsize=(10, 6))

    # If the algorithm is a SOM, we first need to extract p(z) estimates. This may be unnecessary due to the createEnsemble function. Will update.
    for flavor, algorithm in zip(flavors, algorithms):
        if algorithm == 'som':
            wide_path = f'/sdf/data/rubin/shared/pz/roman_rubin_2023/data/{selection}_{flavor}/wide_data_assignment_estimate_sompz.hdf5'
            wide = convert_df(wide_path)
            testFile['cells'] = wide['cells']

            pz_chat_path = f'/sdf/data/rubin/shared/pz/roman_rubin_2023/data/{selection}_{flavor}/pz_chat_estimate_sompz.hdf5'
            grid = np.linspace(0, 6, 601)
            estimated_redshifts = []

            with h5py.File(pz_chat_path, 'r') as f:
                for cell in testFile['cells']:
                    cell_data = f['pz_chat'][int(cell)][:]
                    estimated_redshift = grid[np.argmax(cell_data)]
                    estimated_redshifts.append(estimated_redshift)

            estimated_redshifts = np.array(estimated_redshifts)

            if estimated_redshifts.ndim == 1:
                estimated_redshifts = estimated_redshifts[:, np.newaxis]

            z_grid = np.linspace(0, 6, 601)
            histograms = []

            for redshift in estimated_redshifts:
                hist, _ = np.histogram(redshift, bins=z_grid, density=True)
                histograms.append(hist)

            histograms = np.array(histograms)

            ensemble = qp.Ensemble(gen_func=qp.hist, data=dict(bins=z_grid, pdfs=histograms))
            zmode_values = estimated_redshifts.max(axis=1)
            ensemble.set_ancil(dict(zmode=zmode_values))

            testFile['estimated_redshift'] = np.squeeze(ensemble.ancil['zmode'])
        else:
            file_path = f'/sdf/data/rubin/shared/pz/projects/roman_rubin_2023/data/{selection}_{flavor}/output_estimate_{algorithm}.hdf5'
            outputFile = qp.read(file_path)
            testFile['estimated_redshift'] = np.squeeze(outputFile.ancil['zmode']) # Assign estimated redshift to testFile

        # Calculate accuracy. Accuracy is the fraction of estimates within the magnitude bin with a bias within the threshold 
        accuracy = testFile.groupby('band_bin').apply(
            lambda x: np.mean(np.abs(x['redshift'] - x['estimated_redshift']) <= threshold)
        )
        error = np.sqrt(accuracy * (1 - accuracy) / testFile['band_bin'].value_counts()) # 
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

        plt.xlabel(f"Magnitude: {band_name}")
        plt.ylabel("Redshift Estimator Accuracy")
        #plt.title(f"Comparing {algorithm} Flavors")
        plt.grid()
        plt.legend()
        plt.ylim(0,1)

    plt.xlabel(f"Magnitude: {band_name}")
    plt.ylabel("Redshift Estimator Accuracy")
    #plt.title(f"Comparing Different Flavors and Algorithms")
    plt.grid()
    plt.legend()
    plt.ylim(0,1)
    plt.show()


def accuracyColor(reference, flavors, selection, threshold=0.05, color='g-i'):
    """
    Redshift Estimator Accuracy vs g-i color for multiple flavors/algorithms
    - testFile - string. Path to testing file used
    - flavors - list. Flavor names
    - selection - string. Selection used (i.e. 'maglim_25.5')
    - band_name - string, optional. Column name for color filter, default is g band. 
    - threshold - float, optional. Threshold for accuracy calculation. Default is 0.05
    - color - string, optional. Color is calculated from string and used for analysis. Default is g-i 
    """

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
    Plot r-i vs g-r, Filter dataset to points with bias greater than threshold. Downsample remaining dataset by frac. 
    - testFile - string. Path to testing file used
    - flavors - list. Flavor names
    - selection - string. Selection used (i.e. 'maglim_25.5')
    - band_name - string, optional. Column name for color filter, default is g band. 
    - threshold - float, optional. Threshold for accuracy calculation. Default is 0.05
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
        'red':   ((0.0, 1.0, 1.0),
                  (0.5, 0.0, 0.0),
                  (1.0, 1.0, 1.0)),

        'green': ((0.0, 0.0, 0.0),
                  (0.5, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),

        'blue':  ((0.0, 0.0, 0.0),
                  (0.5, 0.0, 0.0),
                  (1.0, 0.0, 0.0))
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
    
    Contour of unfiltered data is plotted for comparison 
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
    
    k = gaussian_kde([x, y], bw_method=0.3)
    xi, yi = np.mgrid[x.min():x.max():200j, y.min():y.max():200j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    
    ax.contour(xi, yi, zi.reshape(xi.shape), levels=7, colors='k', alpha=0.5, label='Unfiltered Data')

    ax.set_xlabel("g-r")
    ax.set_ylabel("r-i")
    ax.set_title(title)
    ax.grid()
    ax.legend()
    plt.show()