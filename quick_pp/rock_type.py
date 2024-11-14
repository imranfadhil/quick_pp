import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (classification_report, auc, ConfusionMatrixDisplay, confusion_matrix,
                             r2_score, mean_absolute_error)

from quick_pp.utils import min_max_line
from quick_pp.lithology import shale_volume_steiber

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update(
    {
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'legend.fontsize': 'small'
    }
)


def calc_rqi(pore, perm):
    """Calculate RQI (Rock Quality Index) from Kozeny-Carman equation, based on Amaefule et al. (1993)

    Args:
        perm (float): Permeability in mD
        pore (float): Total porosity in fraction

    Returns:
        float: RQI
    """
    return 0.0314 * (perm / pore)**0.5


def calc_fzi(pore, perm):
    """Calculate FZI (Flow Zone Indicator) from Kozeny-Carman equation, based on Amaefule et al. (1993)

    Args:
        perm (float): Permeability in mD
        pore (float): Total porosity in fraction

    Returns:
        float: FZI
    """
    return calc_rqi(pore, perm) / (pore / (1 - pore))


def calc_fzi_perm(fzi, pore):
    """Calculate permeability from FZI and porosity, based on Amaefule et al. (1993)

    Args:
        fzi (float): Flow Zone Indicator
        pore (float): Total porosity in fraction

    Returns:
        float: Permeability in mD
    """
    return pore * ((pore * fzi) / (.0314 * (1 - pore)))**2


def calc_r35(pore, perm):
    """Calculate Winland R35 from Kozeny-Carman equation, based on Winland (1979)

    Args:
        perm (float): Permeability in mD
        pore (float): Total porosity in fraction

    Returns:
        float: R35
    """
    return 10**(.731 + .538 * np.log10(perm) - .864 * np.log10(pore * 100))


def calc_r35_perm(r35, pore):
    """Calculate permeability from Winland R35 and porosity, based on Winland (1979)

    Args:
        r35 (float): Winland R35
        pore (float): Total porosity in fraction

    Returns:
        float: Permeability in mD
    """
    return 10**((np.log10(r35) - 0.732 + 0.864 * np.log10(pore * 100)) / 0.588)


def plot_fzi(cpore, cperm, cut_offs=None, rock_type=None, title='Flow Zone Indicator (FZI)'):
    """Plot the FZI cross plot.

    Args:
        perm (float): Permeability in mD
        pore (float): Total porosity in fraction
    """
    # Plot the FZI cross plot
    _, ax = plt.subplots(figsize=(10, 8))
    plt.title(title)
    plt.scatter(cpore, cperm, marker='.', c=rock_type, cmap='viridis')
    cut_offs = cut_offs if cut_offs is not None else np.arange(0.5, 5)
    phit_points = np.geomspace(0.001, 1, 20)
    for fzi in cut_offs:
        perm_points = phit_points * ((phit_points * fzi) / (.0314 * (1 - phit_points)))**2
        plt.plot(phit_points, perm_points, linestyle='dashed', label=f'FZI={round(fzi, 3)}')

    plt.xlabel('Porosity (frac)')
    plt.xlim(-.05, .5)
    plt.ylabel('Permeability (mD)')
    plt.ylim(1e-3, 1e4)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x), 0)))).format(x)))

    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
    plt.grid(True, which='minor', linestyle=':', linewidth='0.3', color='gray')


def plot_rfn(cpore, cperm, rock_type=None, title='Lucia RFN'):
    """Plot the Rock Fabric Number (RFN) lines on porosity and permeability cross plot. The permeability (mD) is
    calculated based on Lucia-Jenkins, 2003.
    ```
    perm = 10**(9.7892 - 12.0838 * log(RFN) + (8.6711 - 8.2965 * log(RFN)) * log(phi))
    ```

    Args:
        cpore (float): Critical porosity in v/v
        cperm (float): Critical permeability in mD
    """
    # Plot the RFN cross plot
    _, ax = plt.subplots(figsize=(10, 8))
    plt.title(title)
    plt.scatter(cpore, cperm, marker='.', c=rock_type, cmap='viridis')
    pore_points = np.linspace(0, .6, 20)
    for rfn in np.arange(.5, 4.5, .5):
        perm_points = 10**(9.7892 - 12.0838 * np.log10(rfn) + (8.6711 - 8.2965 * np.log10(rfn)) * np.log10(pore_points))
        plt.plot(pore_points, perm_points, linestyle='dashed', label=f'RFN={rfn}')

    plt.xlabel('Porosity (frac)')
    plt.xlim(-.05, .5)
    plt.ylabel('Permeability (mD)')
    plt.ylim(1e-3, 1e4)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x), 0)))).format(x)))

    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
    plt.grid(True, which='minor', linestyle=':', linewidth='0.3', color='gray')


def plot_winland(cpore, cperm, cut_offs=None, rock_type=None, title='Winland R35'):
    """Plot the Winland R35 lines on porosity and permeability cross plot. The permeability (mD) is calculated based on
    Winland, 1979.
    ```
    perm = 10**((log(r35) - 0.731 + 0.864 * log(phi)) / 0.538)
    ```

    Args:
        cpore (float): Critical porosity in v/v
        cperm (float): Critical permeability in mD
    """
    # Plot the Winland R35 cross plot
    _, ax = plt.subplots(figsize=(10, 8))
    plt.title(title)
    plt.scatter(cpore, cperm, marker='.', c=rock_type, cmap='viridis')
    cut_offs = cut_offs if cut_offs is not None else [.05, .1, .5, 2, 10, 100]
    pore_points = np.linspace(0.001, .6, 20)
    for r35 in cut_offs:
        perm_points = calc_r35_perm(r35, pore_points)
        plt.plot(pore_points, perm_points, linestyle='dashed', label=f'R35={r35}')

    plt.xlabel('Porosity (frac)')
    plt.xlim(-.05, .5)
    plt.ylabel('Permeability (mD)')
    plt.ylim(1e-3, 1e4)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x), 0)))).format(x)))

    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
    plt.grid(True, which='minor', linestyle=':', linewidth='0.3', color='gray')


def plot_ward(cpore, cperm, title="Ward's Plot"):
    """Plot Ward's plot to identify possible flow units.

    Args:
        cpore (float): Core porosity in fraction.
        cperm (float): Core permeability in mD.
        title (str, optional): Title of the plot. Defaults to "Ward's Plot".
    """
    fzi = calc_fzi(cpore, cperm)
    fzi = fzi[~np.isnan(fzi)]
    sorted_fzi = sorted(fzi)
    log_fzi = np.log10(sorted_fzi)
    p_log_fzi = (np.arange(1, len(log_fzi) + 1) - .5) / len(log_fzi)
    t = (-2 * np.log(p_log_fzi))**0.5
    zi = abs(t - ((2.30753 + .27061 * t) / (1 + 0.99229 * t + 0.04481 * t**2)))

    # Generate Ward's plot
    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.scatter(log_fzi, zi, marker='.')
    plt.xlabel('log(fzi)')
    plt.ylabel('zi')
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
    plt.grid(True, which='minor', linestyle=':', linewidth='0.3', color='gray')
    plt.xticks(np.arange(min(log_fzi[log_fzi != -np.inf]), max(log_fzi[log_fzi != np.inf]), .2))


def plot_lorenz_heterogeneity(cpore, cperm, title="Lorenz's Plot"):
    """Plot Lorenz's plot to estimate heteroginity.

    Args:
        cpore (float): Core porosity in fraction.
        cperm (float): Core permeability in mD.
        title (str, optional): Title of the plot. Defaults to "Lorenz's Plot".
    """
    sorted_perm, sorted_phit = zip(*sorted(zip(cperm, cpore), reverse=True))
    perm_cdf = np.cumsum(sorted_perm) / np.sum(sorted_perm)
    phit_cdf = np.cumsum(sorted_phit) / np.sum(sorted_phit)
    lorenz_coeff = (auc(phit_cdf, perm_cdf) - 0.5) / 0.5
    # Generate Lorenz's plot
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.text(.4, .1, f'Lorenz Coefficient: {lorenz_coeff:.2f}', fontsize=10, transform=plt.gca().transAxes)
    plt.scatter(phit_cdf, perm_cdf, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='dashed', color='gray')
    plt.xlabel('CDF of Porosity')
    plt.ylabel('CDF of Permeability')
    plt.xlim(0, 1)
    plt.ylim(0, 1)


def plot_modified_lorenz(cpore, cperm, title="Modified Lorenz's Plot"):
    """Plot Lorenz's plot to identify possible flow units.

    Args:
        cpore (float): Core porosity in fraction.
        cperm (float): Core permeability in mD.
        title (str, optional): Title of the plot. Defaults to "Lorenz's Plot".
    """
    fzi = calc_fzi(cpore, cperm)
    sorted_fzi, sorted_perm, sorted_pore = zip(*sorted(zip(fzi, cperm, cpore), reverse=False))
    perm_cdf = np.cumsum(sorted_perm) / np.sum(sorted_perm)
    pore_cdf = np.cumsum(sorted_pore) / np.sum(sorted_pore)
    # Generate Lorenz's plot
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.scatter(pore_cdf, perm_cdf, marker='.')
    plt.xlabel('CDF of Porosity')
    plt.ylabel('CDF of Permeability')
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
    plt.grid(True, which='minor', linestyle=':', linewidth='0.3', color='gray')


def estimate_vsh_gr(gr, min_gr=None, max_gr=None, alpha=0.1):
    """Estimate volume of shale from gamma ray. If min_gr and max_gr are not provided,
    it will be automatically estimated.

    Args:
        gr (float): Gamma ray from well log.
        min_gr (float, optional): Minimum gamma ray value. Defaults to None.
        max_gr (float, optional): Maximum gamma ray value. Defaults to None.
        alpha (float, optional): Alpha value for min-max normalization. Defaults to 0.1.

    Returns:
        float: VSH_GR.
    """
    # Normalize gamma ray
    if not max_gr or (not min_gr and min_gr != 0):
        # Remove high outliers and forward fill missing values
        gr = np.where(gr <= np.nanmean(gr) + 1.5 * np.nanstd(gr), gr, np.nan)
        mask = np.isnan(gr)
        idx = np.where(~mask, np.arange(len(mask)), 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        gr = gr[idx]

        _, max_gr = min_max_line(gr, alpha)
        gri = np.where(gr < max_gr, gr / max_gr, 1)
    else:
        gri = (gr - min_gr) / (max_gr - min_gr)
        gri = np.where(gri < 1, gri, 1)
    return shale_volume_steiber(gri).flatten()


def estimate_vsh_dn(phin, phid, phin_sh=0.35, phid_sh=0.05):
    """Estimate volume of shale from neutron porosity and density porosity.

    Args:
        phin (float): Neutron porosity in fraction.
        phid (float): Density porosity in fraction.
        phin_sh (float, optional): Neutron porosity for shale. Defaults to 0.35.
        phid_sh (float, optional): Density porosity for shale. Defaults to 0.05.

    Returns:
        float: Volume of shale.
    """
    return (phin - phid) / (phin_sh - phid_sh)


def rock_typing(curve, cut_offs=[.1, .2, .3, .4], higher_is_better=True):
    """Rock typing based on cutoffs.

    Args:
        curve (float): Curve to be used for rock typing.
        cut_offs (list, optional): 3 cutoffs to group the curve into 4 rock types. Defaults to [.1, .2, .3, .4].
        higher_is_better (bool, optional): Whether higher value of curve is better quality or not. Defaults to True.

    Returns:
        float: Rock type.
    """
    # Set number of rock types
    rock_type = np.arange(1, len(cut_offs) + 2)
    rock_type = rock_type[::-1] if higher_is_better else rock_type
    # Rock typing based on cutoffs
    conditions = [curve < cut_off for cut_off in cut_offs] + [curve >= cut_offs[-1]]
    choices = rock_type[-len(conditions):]
    return np.where(np.isnan(curve), np.nan, np.select(conditions, choices))


def train_classification_model(data, input_features: list, target_feature: str):
    """Train a classification Random Forest model to predict a binary feature.

    Args:
        data (DataFrame): Dataframe containing input and target features.
        input_features (list): List of input features.
        target_feature (str): The target feature.

    Returns:
        RandomForestClassifier: Trained model.
    """
    random_seed = 123
    X = data[input_features]
    y = data[target_feature]
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

    # Hyperparameter tuning
    param_dist = {
        'n_estimators': [10, 100, 200],
        'max_depth': [5, 30, None],
        'min_samples_split': [2, 5, 10],
    }
    model = RandomizedSearchCV(RandomForestClassifier(), param_dist, cv=5, random_state=random_seed)
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    print(f'Score for {target_feature} model\n')
    print('Best parameters found:\n', model.best_params_)
    print('Classification Report:\n', classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot()

    return model


def train_regression_model(data, input_features: list, target_feature: list, stratifier=None):
    """Train a regression Random Forest model to predict a continuous feature.

    Args:
        data (DataFrame): Dataframe containing input, target and stratifier features.
        input_features (list): List of input features.
        target_feature (str): The target feature.
        stratifier (array, optional): Stratifier for train-test split. Defaults to None.

    Returns:
        RandomForestRegressor: Trained model.
    """
    random_seed = 123
    X = data[input_features]
    y = data[target_feature]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=stratifier)

    # Hyperparameter tuning
    param_dist = {
        'n_estimators': [10, 50, 100],
        'max_depth': [5, 30, None],
        'min_samples_split': [2, 5, 10],
    }
    model = RandomizedSearchCV(RandomForestRegressor(), param_dist, cv=3, random_state=random_seed)
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    print(f'Score for {target_feature} model\n')
    print('Best parameters found:\n', model.best_params_)
    print('R2 Score:', r2_score(y_test, y_pred))
    print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

    # Plot the true vs predicted values
    plt.figure(figsize=(10, 8))
    plt.plot(y_test, y_pred, '.')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title(f'{target_feature} Prediction')
    plt.show()

    return model
