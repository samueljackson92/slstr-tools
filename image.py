import os
import numpy as np
import xarray as xr
import threading
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

from src import constants


class Regrid:

    def __init__(self, grid_a, grid_b):
        if grid_a.start_offset == grid_b.start_offset:
            start_offset = 0.0
        else:
            start_offset = grid_a.start_offset * float(grid_a.resolution.split()[2]) / float(grid_b.resolution.split()[2]) - grid_b.start_offset

        x = (np.array(range(grid_a.dims['columns'])) - grid_a.track_offset) * float(grid_a.resolution.split()[1]) / float(grid_b.resolution.split()[1]) + grid_b.track_offset
        y = np.array(range(grid_a.dims['rows'])) * float(grid_a.resolution.split()[2]) / float(grid_b.resolution.split()[2]) + start_offset

        self._x = x
        self._y = y

        self._rows = grid_b.dims['rows']
        self._cols = grid_b.dims['columns']

    def regrid_channel(self, grid):
        f = RectBivariateSpline(np.arange(self._rows),
                                           np.arange(self._cols),
                                           grid[:])
        values = f(self._y, self._x)
        return (('rows', 'columns'), values)

    def __call__(self, dataset):
        channels = {}
        for name in dataset:
            channels[name] = self.regrid_channel(dataset[name])

        return xr.Dataset(channels)

def resize_product(data, resolution=constants.RESOLUTION_1KM):
    h, w = resolution

    def downsample(img):
        if img.dtype == np.float32:
            mode = 'F'
        if img.dtype == np.uint8:
            mode = 'L'
        if img.dtype == np.int32 or img.dtype == np.int64:
            mode = 'I'

        img = resize(img, (h, w), mode=mode)

        return img

    channels = {}
    for channel_name in data:
        channel = data[channel_name]
        attrs = channel.attrs.copy()

        if channel.dtype == np.float64:
            channel = channel.astype(np.float32)

        if channel.dtype == np.uint16:
            channel = channel.astype(np.int32)

        channels[channel_name] = xr.apply_ufunc(downsample, channel,
                                                dask='parallelized',
                                                input_core_dims=[['rows', 'columns']],
                                                output_core_dims=[['rows', 'columns']],
                                                output_dtypes=[channel.dtype],
                                                output_sizes={'columns': w, 'rows': h},
                                                exclude_dims=set(['rows', 'columns']),
                                                keep_attrs=True)

        channels[channel_name] = channels[channel_name].assign_attrs(**attrs)

    return xr.Dataset(channels)


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


class ImageLoader:

    def __init__(self, path):
        self.path = path

    def load_radiances(self, view='an'):
        rads = [
            self.load_radiance_channel(
                self.path,
                i,
                view) for i in range(
                1,
                7)]
        rads = xr.merge(rads)
        return rads

    def load_irradiances(self, view='an'):
        irradiances = {}
        for i in range(1, 7):
            name = 'S{}_solar_irradiance_{}'.format(i, view)
            file_name = os.path.join(
                self.path, 'S{}_quality_{}.nc'.format(
                    i, view))
            irradiance = xr.open_dataset(file_name, engine='h5netcdf')[name][:].data[0]
            irradiances[name] = irradiance
        return irradiances


    def load_reflectance(self, view='an'):
        refs = [
            self.load_reflectance_channel(
                self.path,
                i,
                view) for i in range(
                1,
                7)]
        refs = xr.merge(refs)
        return refs

    def load_reflectance_channel(self, path, channel_num, view='an'):
        rads = self.load_radiance_channel(path, channel_num, view)
        names = {name: name.replace('radiance', 'reflectance')
                 for name in rads}
        rads = rads.rename(names)
        irradiances = self.load_irradiances(view)
        geometry = self.load_geometry()

        solar_zenith = geometry.solar_zenith_tn[:]
        solar_zenith = np.nan_to_num(solar_zenith, 0.0)

        regridder = Regrid(rads, geometry)
        solar_zenith = regridder.regrid_channel(solar_zenith)[1]

        DTOR = 0.017453292
        mu0 = np.where(solar_zenith < 90, np.cos(DTOR * solar_zenith), 1.0)

        name = 'S{}_reflectance_{}'.format(channel_num, view)
        rads[name] = rads[name] / \
            (irradiances[name[:2] + '_solar_irradiance_{}'.format(view)] * mu0) * np.pi
        return rads

    def load_radiance_channel(self, path, channel_num, view='an'):
        excluded_vars = [
            "S{}_exception_{}".format(channel_num, view),
            "S{}_radiance_orphan_{}".format(channel_num, view),
            "S{}_exception_orphan_{}".format(channel_num, view)
        ]

        path = os.path.join(
            path, 'S{}_radiance_{}.nc'.format(
                channel_num, view))
        radiance = xr.open_dataset(
            path, decode_times=False, drop_variables=excluded_vars, engine='h5netcdf')
        return radiance

    def load_bts(self, view='in'):
        bts = [self.load_bt_channel(self.path, i, view) for i in range(7, 10)]
        bts = xr.merge(bts)
        return bts

    def load_bt_channel(self, path, channel_num, view='in'):
        excluded_vars = [
            "S{}_exception_{}".format(channel_num, view),
            "S{}_BT_orphan_{}".format(channel_num, view),
            "S{}_exception_orphan_{}".format(channel_num, view)
        ]

        path = os.path.join(path, 'S{}_BT_{}.nc'.format(channel_num, view))
        bt = xr.open_dataset(path, decode_times=False,
                             drop_variables=excluded_vars, engine='h5netcdf')
        return bt

    def load_flags(self, view='in'):
        flags_path = os.path.join(self.path, 'flags_{}.nc'.format(view))
        excluded = [
            'confidence_orphan_{}',
            'pointing_orphan_{}',
            'pointing_{}',
            'cloud_orphan_{}',
            'bayes_orphan_{}',
            'probability_cloud_dual_{}']
        excluded = [e.format(view) for e in excluded]
        flags = xr.open_dataset(flags_path, decode_times=False,
                                drop_variables=excluded, engine='h5netcdf')

        confidence_var = 'confidence_{}'.format(view)
        flag_masks = flags[confidence_var].attrs['flag_masks']
        flag_meanings = flags[confidence_var].attrs['flag_meanings'].split()
        flag_map = dict(zip(flag_meanings, flag_masks))
        expanded_flags = {}
        for key, bit in flag_map.items():
            msk = flags[confidence_var] & bit
            msk = xr.where(msk > 0, 1, 0)
            expanded_flags[key] = msk
        flags = flags.assign(**expanded_flags)
        return flags

    def load_geometry(self):
        path = os.path.join(self.path, 'geometry_tn.nc')
        geo = xr.open_dataset(path, decode_times=False, engine='h5netcdf')
        return geo

    def load_met(self):
        met_path = os.path.join(self.path, 'met_tx.nc')
        met = xr.open_dataset(met_path, decode_times=False, engine='h5netcdf')
        met = met[['total_column_water_vapour_tx', 'cloud_fraction_tx',
                   'skin_temperature_tx', 'sea_surface_temperature_tx',
                   'total_column_ozone_tx', 'soil_wetness_tx',
                   'snow_albedo_tx', 'snow_depth_tx', 'sea_ice_fraction_tx',
                   'surface_pressure_tx']]
        met = met.squeeze()
        return met

    def load_geodetic(self, view='an'):
        flags_path = os.path.join(self.path, 'geodetic_{}.nc'.format(view))
        excluded = ['elevation_orphan_an', 'elevation_an',
                    'latitude_orphan_an', 'longitude_orphan_an']
        flags = xr.open_dataset(flags_path, decode_times=False,
                                drop_variables=excluded, engine='h5netcdf')
        return flags


class ProductImage:
    def __init__(self, product):
        self._loader = ImageLoader(product)
        self._bts = None
        self._rads = None

    @property
    def bts(self):
        if self._bts is None:
            self._bts = self._loader.load_bts()
        return self._bts

    @property
    def rads(self):
        if self._rads is None:
            self._rads = self._loader.load_radiances()
        return self._rads

class ImagePlotter:

    def __init__(self, image):
        if isinstance(image, str):
            self._image = ProductImage(image)
        else:
            self._image = image

    def plot_bt_channels(self, **kwargs):
        fig, axes = plt.subplots(1, 3)
        axes = axes.flatten()
        for ax, name in zip(axes, self._image.bts):
            ax.matshow(self._image.bts[name], **kwargs)

    def plot_rad_channels(self, **kwargs):
        fig, axes = plt.subplots(2, 3)
        axes = axes.flatten()
        for ax, name in zip(axes, self._image.rads):
            ax.matshow(self._image.rads[name], **kwargs)


def central_crop(img, percent):
    h, w = img.shape[:2]
    dx = int(h * (1 - percent)) // 2
    dy = int(w * (1 - percent)) // 2
    return img[dx:-dx, dy:-dy]


def central_crop_batch(img, input_size, output_size):
    dh, dw = (input_size - output_size) // 2, (input_size - output_size) // 2
    return img[:, dh:-dh, dw:-dw]


def resize(img, shape, mode='L'):
    """Resize an image to the desired size

    This will use the PIL library to quickly resize the image to the new shape.
    The interpolation method used is Lanczos.

    Args:
        img (np.array): the image to resize
        shape (tuple): shape to resize the image to. Should contain two
            elements (height, width).

    Returns:
        np.array: an array with the same size as `shape`.
    """
    img = Image.fromarray(img, mode=mode)
    img = img.resize(reversed(shape), resample=Image.LANCZOS)
    img = np.array(img)
    return img


def resize_spatial(data, h, w, mode='F'):

    def downsample(img):
        if img.dtype == np.float32:
            mode = 'F'
        if img.dtype == np.uint8:
            mode = 'L'
        if img.dtype == np.int32:
            mode = 'I'

        img = resize(img, (h, w), mode=mode)

        return img

    channels = {}
    for channel_name in data:
        channel = data[channel_name]
        attrs = channel.attrs.copy()

        if channel.dtype == np.float64:
            channel = channel.astype(np.float32)

        if channel.dtype == np.uint16:
            channel = channel.astype(np.int32)

        channels[channel_name] = xr.apply_ufunc(downsample, channel,
                                                dask='parallelized',
                                                input_core_dims=[
                                                    ['rows', 'columns']],
                                                output_core_dims=[
                                                    ['rows', 'columns']],
                                                output_dtypes=[channel.dtype],
                                                output_sizes={
                                                    'columns': w, 'rows': h},
                                                exclude_dims=set(
                                                    ['rows', 'columns']),
                                                keep_attrs=True)

        channels[channel_name] = channels[channel_name].assign_attrs(**attrs)

    out = xr.Dataset(channels)
    return out

