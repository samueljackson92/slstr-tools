from pathlib import Path
from image import ImageLoader, Regrid
import pytest

@pytest.fixture()
def product_path():
    """ Fixture to return the path to the local testing product """
    return str(next(Path('data').glob('S3A*')))

def test_create_product(product_path):
    product = ImageLoader(product_path)
    assert isinstance(product, ImageLoader)

def test_load_bts(product_path):
    product = ImageLoader(product_path)
    bts = product.load_bts()

    assert bts.to_array().shape == (3, 1198, 1500) # 1km grid
    assert 'track_offset' in bts.attrs
    assert 'start_offset' in bts.attrs
    assert 'resolution' in bts.attrs

def test_load_radiances(product_path):
    product = ImageLoader(product_path)
    rads = product.load_radiances()

    assert rads.to_array().shape == (6, 1198*2, 1500*2) # .5km grid
    assert 'track_offset' in rads.attrs
    assert 'start_offset' in rads.attrs
    assert 'resolution' in rads.attrs

def test_load_reflectances(product_path):
    product = ImageLoader(product_path)
    refs = product.load_reflectances()

    assert refs.to_array().shape == (6, 1198*2, 1500*2) # .5km grid
    assert 'track_offset' in refs.attrs
    assert 'start_offset' in refs.attrs
    assert 'resolution' in refs.attrs

def test_load_geometry(product_path):
    product = ImageLoader(product_path)
    geo = product.load_geometry()

    assert 'sat_zenith_tn' in geo.data_vars
    assert 'sat_azimuth_tn' in geo.data_vars
    assert 'solar_azimuth_tn' in geo.data_vars
    assert 'solar_zenith_tn' in geo.data_vars

    for name in geo:
        assert geo[name].shape == (1198, 130)

def test_load_met(product_path):
    product = ImageLoader(product_path)
    met = product.load_met()

    columns = ['total_column_water_vapour_tx', 'cloud_fraction_tx',
            'skin_temperature_tx', 'sea_surface_temperature_tx',
            'total_column_ozone_tx', 'soil_wetness_tx', 'snow_albedo_tx',
            'snow_depth_tx', 'sea_ice_fraction_tx', 'surface_pressure_tx']

    for name in columns:
        assert name in met.data_vars
        assert met[name].shape == (1198, 130) # lower resolution meterological grid

def test_load_geodetic(product_path):
    product = ImageLoader(product_path)
    geodetic = product.load_geodetic()

    assert 'latitude_an' in geodetic.data_vars
    assert 'longitude_an' in geodetic.data_vars

    for name in geodetic:
        assert geodetic[name].shape == (2396, 3000)

def test_load_flags(product_path):
    product = ImageLoader(product_path)
    flags = product.load_flags()

    assert 'bayes_in' in flags.data_vars
    assert 'summary_cloud' in flags.data_vars
    assert 'ocean' in flags.data_vars
    assert 'land' in flags.data_vars
    assert 'twilight' in flags.data_vars
    assert 'day' in flags.data_vars

    for name in flags:
        assert flags[name].shape == (1198, 1500)

def test_regrid(product_path):
    product = ImageLoader(product_path)
    bts = product.load_bts()
    met = product.load_met()
    geometry = product.load_geometry()

    regridder = Regrid(bts, geometry)
    met_resized = regridder(met)

    for name in met_resized:
        assert met_resized[name].shape == (1198, 1500)
