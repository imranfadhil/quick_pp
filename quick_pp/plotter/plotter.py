import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from quick_pp.utils import line_intersection
from quick_pp import logger
import quick_pp.plotter.well_log as plotter_wells
plotly_log = plotter_wells.plotly_log

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update(
    {
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'legend.fontsize': 'small'
    }
)


def update_fluid_contacts(well_data, well_config: dict):
    logger.debug(f"Updating fluid contacts for well with config: {well_config}")
    """Update fluid flags based on fluid contacts.

    Args:
        well_data (pandas.DataFrame): Pandas dataframe containing well log data.
        well_config (dict): Dictionary containing well sorting and fluid contacts.

    Returns:
        pandas.DataFrame: Pandas dataframe containing updated fluid
    """
    owc = well_config.get('OWC', np.nan)
    odt = well_config.get('ODT', np.nan)
    out = well_config.get('OUT', np.nan)
    goc = well_config.get('GOC', np.nan)
    gdt = well_config.get('GDT', np.nan)
    gut = well_config.get('GUT', np.nan)
    gwc = well_config.get('GWC', np.nan)
    wut = well_config.get('WUT', np.nan)

    well_data = well_data.copy()
    logger.debug(f"well_data columns before update: {well_data.columns.tolist()}")
    well_data['OIL_FLAG'] = np.where(
        ((well_data['DEPTH'] > out) | (well_data['DEPTH'] > goc)) & (
            (well_data['DEPTH'] < odt) | (well_data['DEPTH'] < owc)), 1, 0)

    well_data['GAS_FLAG'] = np.where(
        ((well_data['DEPTH'] < gdt) | (well_data['DEPTH'] < gwc) | (well_data['DEPTH'] < goc)) & (
            well_data['DEPTH'] > gut), 1, 0)

    well_data['WATER_FLAG'] = np.where(
        ((well_data['DEPTH'] > wut) | (well_data['DEPTH'] > owc) | (well_data['DEPTH'] > gwc)), 1, 0)

    well_data['FLUID_FLAG'] = np.where(
        well_data['OIL_FLAG'] == 1, 1, np.where(
            well_data['GAS_FLAG'] == 1, 2, 0))

    return well_data


def generate_zone_config(zones: list = ['ALL']):
    logger.info(f"Generating zone config for zones: {zones}")
    """Generate zone configuration.

    Args:
        zones (list, optional): List of zone names. Defaults to ['ALL'].

    Returns:
        dict: Zone configuration
    """
    zone_config = {}
    for zone in zones:
        zone_config[zone] = {
            'GUT': 0, 'GDT': 0, 'GOC': 0, 'GWC': 0, 'OUT': 0, 'ODT': 0, 'OWC': 0, 'WUT': 0
        }
    return zone_config


def update_zone_config(zone_config: dict, zone: str, fluid_contacts: dict):
    logger.info(f"Updating zone config for zone: {zone} with contacts: {fluid_contacts}")
    """Update zone configuration with fluid contacts.

    Args:
        zone_config (dict): Dictionary containing zone configuration.
        zone (str): Zone name for which configuration is to be updated.
        fluid_contacts (dict): Fluid contacts for the specified zone.

    Returns:
        dict: Updated zone configuration
    """
    if zone in zone_config:
        zone_config[zone].update(fluid_contacts)
    else:
        zone_config[zone] = fluid_contacts
    return zone_config


def generate_well_config(well_names: list = ['X']):
    logger.info(f"Generating well config for wells: {well_names}")
    """Generate well configuration.

    Args:
        well_names (list, optional): List of well names. Defaults to ['X'].

    Returns:
        dict: Well configuration
    """
    well_config = {}
    for i, well in enumerate(well_names):
        well_config[well] = {
            'sorting': i + 1,
            'zones': {
                'ALL': {'GUT': 0, 'GDT': 0, 'GOC': 0, 'GWC': 0, 'OUT': 0, 'ODT': 0, 'OWC': 0, 'WUT': 0},
            }
        }
    return well_config


def update_well_config(
        well_config: dict, well_name: str, zone: str = '', fluid_contacts: dict = {}, sorting: int = 0):
    logger.info(f"Updating well config for well: {well_name}, zone: {zone}, sorting: {sorting}")
    """Update well configuration with fluid contacts.

    Args:
        well_config (dict): Dictionary containing well sorting and fluid contacts.
        well_name (str): Well name for which configuration is to be updated.
        zone (str): Zone name for which configuration is to be updated.
        fluid_contacts (dict): Fluid contacts for the specified zone.
        sorting (int, optional): Sorting of the wells on the stick plot. Defaults to None.

    Returns:
        dict: Updated well configuration
    """
    if zone in well_config[well_name]['zones']:
        well_config[well_name]['zones'][zone].update(fluid_contacts)
    elif zone:
        well_config[well_name]['zones'][zone] = fluid_contacts

    if sorting:
        well_config[well_name]['sorting'] = sorting
    return well_config


def assert_well_config_structure(well_config):
    logger.debug("Asserting well config structure.")
    """Assert well configuration structure.

    Args:
        well_config (dict): Dictionary containing well sorting and fluid contacts.
    """
    required_keys = {'sorting', 'zones'}
    optional_keys = {'GUT', 'GDT', 'GOC', 'GWC', 'OUT', 'ODT', 'OWC', 'WUT'}
    for well, config in well_config.items():
        assert isinstance(config, dict), f"Value for well '{well}' is not a dictionary"
        assert set(config.keys()).intersection(required_keys), f"Well '{well}' does not have the required keys"

        for zone, fluid_contacts in config['zones'].items():
            assert isinstance(fluid_contacts, dict), f"Value for zone '{zone}' in well '{well}' is not a dictionary"
            assert set(fluid_contacts.keys()).intersection(
                optional_keys), f"zone '{zone}' in well '{well}' has invalid keys"


def stick_plot(data, well_config: dict, zone: str = 'ALL'):
    logger.info(f"Generating stick plot for zone: {zone}")
    """Generate stick plot with water saturation and fluid contacts for specified zone.

    Example of well_config:
    ```
    well_config = {
        'X': {
            'sorting': 0,
            'zones': {
                'ALL': {
                    'GUT': 0,
                    'GDT': 0,
                    'GOC': 0,
                    'GWC': 0,
                    'OUT': 0,
                    'ODT': 0,
                    'OWC': 0,
                    'WUT': 0
                },
            }
        }
    }
    ```

    Args:
        data (pandas.DataFrame): Pandas dataframe containing well log data.
        well_config (dict): Dictionary containing well sorting and fluid contacts.
    """
    assert 'SWT' in data.columns, 'SWT column not found in data.'
    assert_well_config_structure(well_config)

    # Create BVO and BVOH columns
    data['BVO'] = data['PHIT'] * (1 - data['SWT'])
    data['BVOH'] = data['PHIT'] * (1 - data['SHF']) if 'SHF' in data.columns else 0

    # Sort well names based on sorting key
    well_names = sorted(data['WELL_NAME'].unique(), key=lambda name: well_config[name]['sorting'])

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(well_names), sharey=True, figsize=(len(well_names) * 2, 15))

    # Plot each well's data
    for ax, well_name in zip(axes, well_names):
        logger.debug(f"Plotting well: {well_name}")
        well_data = data[(data['WELL_NAME'] == well_name) & (data['ZONES'] == zone)].copy()
        well_data = update_fluid_contacts(well_data, well_config[well_name]['zones'][zone])
        ax.plot(well_data['BVO'], well_data['DEPTH'], label=r'$BVO_{Log}$')
        if 'BVOH' in well_data.columns:
            ax.plot(well_data['BVOH'], well_data['DEPTH'], label=r'$BVO_{SHF}$')

        # Fill between based on fluid flag
        ax.fill_betweenx(
            well_data['DEPTH'], 0, 1, where=well_data['FLUID_FLAG'] == 1, color='g', alpha=0.3, label='Oil')
        ax.fill_betweenx(
            well_data['DEPTH'], 0, 1, where=well_data['FLUID_FLAG'] == 2, color='r', alpha=0.3, label='Gas')
        ax.fill_betweenx(
            well_data['DEPTH'], 0, 1, where=well_data['FLUID_FLAG'] == 0, color='b', alpha=0.3, label='Water')

        ax.set_title(f'Well: {well_name}')
        ax.set_xlim(0, .5)
        ax.legend()

    axes[0].set_ylabel('Depth')
    fig.subplots_adjust(wspace=0.3, hspace=0)
    fig.set_facecolor('aliceblue')
    plt.gca().invert_yaxis()
    plt.show()


def neutron_density_xplot(nphi, rhob,
                          dry_min1_point: tuple,
                          dry_clay_point: tuple,
                          fluid_point: tuple = (1.0, 1.0),
                          wet_clay_point: tuple = (),
                          dry_silt_point: tuple = (), **kwargs):
    logger.info("Generating neutron-density crossplot.")
    """Neutron-Density crossplot with lithology lines based on specified end points.

    Args:
        nphi (float): Neutron porosity log.
        rhob (float): Bulk density log.
        dry_min1_point (tuple): Neutron porosity and bulk density of mineral 1 point.
        dry_clay_point (tuple): Neutron porosity and bulk density of dry clay point.
        fluid_point (tuple): Neutron porosity and bulk density of fluid point.
        wet_clay_point (tuple): Neutron porosity and bulk density of wet clay point.
        dry_silt_point (tuple): Neutron porosity and bulk density of dry silt point. Default is None.

    Returns:
        matplotlib.pyplot.Figure: Neutron porosity and bulk density cross plot.
    """
    A = dry_min1_point
    C = dry_clay_point
    D = fluid_point
    E = list(zip(nphi, rhob))
    # Plotting the NPHI-RHOB crossplot
    projected_pt = []
    for i in range(len(nphi)):
        projected_pt.append(line_intersection((A, C), (D, E[i])))
    rockline_from_pt = (A, C)
    min1line_from_pt = (D, A)
    clayline_from_pt = (D, C)

    fig = Figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_title('NPHI-RHOB Crossplot')
    ax.scatter(nphi, rhob, c=np.arange(0, len(nphi)).tolist(), cmap='rainbow', marker='.')
    ax.plot(*zip(*min1line_from_pt), label='Mineral 1 Line', color='blue')

    if dry_silt_point:
        B = dry_silt_point
        siltline_from_pt = (D, B)
        ax.plot(*zip(*siltline_from_pt), label='Silt Line', color='green')
        ax.scatter(dry_silt_point[0], dry_silt_point[1], label='Dry Silt Point', color='orange')

    ax.plot(*zip(*clayline_from_pt), label='Clay Line', color='gray')
    ax.plot(*zip(*rockline_from_pt), label='Rock Line', color='black')
    ax.scatter(*zip(*projected_pt), label='Projected Line', color='purple', marker='.')
    ax.scatter(dry_min1_point[0], dry_min1_point[1],
               label=f'Mineral 1 Point: ({dry_min1_point[0]}, {dry_min1_point[1]})', color='yellow')
    ax.scatter(dry_clay_point[0], dry_clay_point[1],
               label=f'Dry Clay Point: ({dry_clay_point[0]}, {dry_clay_point[1]})', color='black')

    if wet_clay_point:
        ax.scatter(wet_clay_point[0], wet_clay_point[1], label='Wet Clay Point', color='gray')

    ax.scatter(fluid_point[0], fluid_point[1],
               label=f'Fluid Point: ({fluid_point[0]}, {fluid_point[1]})', color='blue')

    ax.set_ylim(3, 0)
    ax.set_ylabel('RHOB')
    ax.set_xlim(-0.15, 1)
    ax.set_xlabel('NPHI')
    ax.legend(loc="upper left", prop={'size': 9})
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
    ax.grid(True, which='minor', linestyle=':', linewidth='0.4', color='gray')
    fig.tight_layout()

    return fig


def sonic_density_xplot():
    logger.info("Called sonic_density_xplot (not implemented)")
    """ TODO: Implement sonic density crossplot
    """
    pass


def sonic_neutron_xplot():
    logger.info("Called sonic_neutron_xplot (not implemented)")
    """ TODO: Implement sonic neutron crossplot
    """
    pass
