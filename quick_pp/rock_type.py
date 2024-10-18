import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, r2_score, mean_absolute_error

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


def calc_rqi(k, phit):
    """Calculate RQI (Rock Quality Index) from Kozeny-Carman equation, based on Amaefule et al. (1993)

    Args:
        k (float): Permeability in mD
        phit (float): Total porosity in fraction

    Returns:
        float: RQI
    """
    return 0.0314 * (k / phit)**0.5


def calc_fzi(k, phit):
    """Calculate FZI (Flow Zone Indicator) from Kozeny-Carman equation, based on Amaefule et al. (1993)

    Args:
        k (float): Permeability in mD
        phit (float): Total porosity in fraction

    Returns:
        float: FZI
    """
    return calc_rqi(k, phit) / (phit / (1 - phit))


def calc_fzi_perm(fzi, phit):
    """Calculate permeability from FZI and porosity, based on Amaefule et al. (1993)

    Args:
        fzi (float): Flow Zone Indicator
        phit (float): Total porosity in fraction

    Returns:
        float: Permeability in mD
    """
    return phit * ((phit * fzi) / (.0314 * (1 - phit)))**2


def plot_fzi(cpore, cperm, fzi=None, rock_type=None, title='Flow Zone Indicator (FZI)'):
    """Plot the FZI cross plot.

    Args:
        k (float): Permeability in mD
        phit (float): Total porosity in fraction
    """
    # Plot the FZI cross plot
    plt.figure(figsize=(5, 4))
    plt.title(title)
    plt.scatter(cpore, cperm, marker='s', c=rock_type, cmap='viridis')
    fzi_list = fzi if fzi is not None else np.arange(0.5, 5)
    phit_points = np.geomspace(0.001, 1, 20)
    for fzi in fzi_list:
        perm_points = phit_points * ((phit_points * fzi) / (.0314 * (1 - phit_points)))**2
        plt.plot(phit_points, perm_points, linestyle='dashed', label=f'FZI={round(fzi, 3)}')

    plt.xlabel('Porosity (frac)')
    plt.xlim(-.05, .5)
    plt.ylabel('Permeability (mD)')
    plt.ylim(0.001, 10000)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.yscale('log')


def plot_ward(cpore, cperm, title="Ward's Plot"):
    """Plot Ward's plot to identify possible flow units.

    Args:
        cpore (float): Core porosity in fraction.
        cperm (float): Core permeability in mD.
        title (str, optional): Title of the plot. Defaults to "Ward's Plot".
    """
    fzi = calc_fzi(cperm, cpore)
    fzi = fzi[~np.isnan(fzi)]
    sorted_fzi = np.sort(fzi)
    log_fzi = np.log10(sorted_fzi)
    p_log_fzi = (np.arange(1, len(log_fzi) + 1) - .5) / len(log_fzi)
    t = (-2 * np.log(p_log_fzi))**0.5
    zi = abs(t - ((2.30753 + .27061 * t) / (1 + 0.99229 * t + 0.04481 * t**2)))

    # Generate Ward's plot
    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.scatter(log_fzi, zi, marker='s')
    plt.xlabel('log(fzi)')
    plt.ylabel('zi')
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
    plt.grid(True, which='minor', linestyle=':', linewidth='0.4', color='gray')
    plt.xticks(np.arange(min(log_fzi[log_fzi != -np.inf]), max(log_fzi[log_fzi != np.inf]), .2))


def plot_lorenz(cpore, cperm, title="Lorenz's Plot"):
    """Plot Lorenz's plot to identify possible flow units.

    Args:
        cpore (float): Core porosity in fraction.
        cperm (float): Core permeability in mD.
        title (str, optional): Title of the plot. Defaults to "Lorenz's Plot".
    """
    fzi = calc_fzi(cperm, cpore)
    fzi = fzi[~np.isnan(fzi)]
    sorted_fzi = np.sort(fzi)
    p_log_fzi = np.arange(1, len(sorted_fzi) + 1)
    log_fzi = np.log10(sorted_fzi)
    # Generate Lorenz's plot
    plt.figure(figsize=(10, 4))
    plt.title(title)
    plt.scatter(log_fzi, p_log_fzi, marker='s')
    plt.xlabel('log(fzi)')
    plt.ylabel('Count')
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
    plt.grid(True, which='minor', linestyle=':', linewidth='0.4', color='gray')
    plt.xticks(np.arange(min(log_fzi[log_fzi != -np.inf]), max(log_fzi[log_fzi != np.inf]), .2))


def plot_rfn(cpore, cperm, rock_type=None, title='Lucia RFN'):
    """Plot the Rock Fabric Number (RFN) lines on porosity and permeability cross plot. The permeability (mD) is
    calculated based on Lucia-Jenkins, 2003 -
    > k = 10**(9.7892 - 12.0838 * log(RFN) + (8.6711 - 8.2965 * log(RFN)) * log(phi))

    Args:
        cpore (float): Critical porosity in v/v
        cperm (float): Critical permeability in mD
    """
    # Plot the RFN cross plot
    plt.figure(figsize=(5, 4))
    plt.title(title)
    plt.scatter(cpore, cperm, marker='s', c=rock_type, cmap='viridis')
    pore_points = np.linspace(0, .6, 20)
    for rfn in np.arange(.5, 4.5, .5):
        perm_points = 10**(9.7892 - 12.0838 * np.log10(rfn) + (8.6711 - 8.2965 * np.log10(rfn)) * np.log10(pore_points))
        plt.plot(pore_points, perm_points, linestyle='dashed', label=f'RFN={rfn}')

    plt.xlabel('Porosity (frac)')
    plt.xlim(-.05, .5)
    plt.ylabel('Permeability (mD)')
    plt.ylim(0.001, 10000)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.yscale('log')


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
    # Remove high outliers and forward fill missing values
    gr = np.where(gr <= np.nanmean(gr) + 1.5 * np.nanstd(gr), gr, np.nan)
    mask = np.isnan(gr)
    idx = np.where(~mask, np.arange(len(mask)), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    gr = gr[idx]

    # Normalize gamma ray
    if not max_gr or (not min_gr and min_gr != 0):
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
        cut_offs (list, optional): 3 cutoffs to group the curve into 4 rock types. Defaults to [.1, .3, .4].
        higher_is_better (bool, optional): Whether higher value of curve is better quality or not. Defaults to True.

    Returns:
        float: Rock type.
    """
    rock_type = [5, 4, 3, 2, 1] if higher_is_better else [1, 2, 3, 4, 5]
    return np.where(np.isnan(curve), np.nan, np.where(
        curve < cut_offs[0], rock_type[0],
                    np.where(curve < cut_offs[1], rock_type[1],
                             np.where(curve < cut_offs[2], rock_type[2],
                                      np.where(curve < cut_offs[3], rock_type[3], rock_type[4])))))


def train_rock_type(X, y):
    """Train a classification Random Forest model to predict rock type.

    Args:
        X (DataFrame): Dataframe containing features and rock type.
        y (Series): Rock type.

    Returns:
        RandomForestClassifier: Trained model.
    """
    random_seed = 123
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)
    param_dist = {
        'n_estimators': [10, 100, 200],
        'max_depth': [5, 30, None],
        'min_samples_split': [2, 5, 10],
    }
    model = RandomizedSearchCV(RandomForestClassifier(), param_dist, random_state=random_seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Best parameters found:\n', model.best_params_)
    print('Classification Report:\n', classification_report(y_test, y_pred))

    return model


def train_fzi(X, y, stratifier=None):
    """Train a regression Random Forest model to predict FZI.

    Args:
        X (DataFrame): Dataframe containing features and rock type.
        y (Series): Flow Zone Indicator.

    Returns:
        RandomForestRegressor: Trained model.
    """
    random_seed = 123
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=stratifier)
    param_dist = {
        'n_estimators': [10, 100, 200],
        'max_depth': [5, 30, None],
        'min_samples_split': [2, 5, 10],
    }
    model = RandomizedSearchCV(RandomForestRegressor(), param_dist, random_state=random_seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Best parameters found:\n', model.best_params_)
    print('R2 Score:', r2_score(y_test, y_pred))
    print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

    return model
