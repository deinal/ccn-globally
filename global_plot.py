import xarray as xr
import xgboost as xgb
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
import pandas as pd
import numpy as np
import gif


features = ['co', 'c5h8', 'no', 'no2', 'so2']

# Prepare data
ds = xr.open_dataset('data/global.grib', engine='cfgrib')


@gif.frame
def plot_frame(i, t):
    df = ds.isel(time=i).to_dataframe()
    for v in features:
        df[v] = np.log(df[v].where(df[v] > 0, df[v][df[v] > 0].min()))

    # Load ML model
    params = {'n_estimators': 100, 'reg_lambda': 2, 'learning_rate': 0.3, 'max_depth': 7}
    model = xgb.XGBRegressor(**params)
    model.load_model('models/gases.model')

    # Predict n100 concentration
    df['n100'] = np.exp(model.predict(df[features]))
    result = xr.Dataset.from_dataframe(df)
    result['n100'] = result.n100.assign_attrs(units='$\mathrm{cm}^{-3}$')
    result['n100'] = result.n100.assign_attrs(long_name='Conc.')

    # Plot result

    major_ticks = [10, 100, 1000, 10000]
    fig, ax = plt.subplots(nrows=1, subplot_kw={'projection': ccrs.Robinson()})
    earth = result.n100.plot.contourf(
        ax=ax, transform=ccrs.PlateCarree(), 
        levels=np.geomspace(4.85, 8170, 30), norm=LogNorm(), cmap='viridis', extend='neither',
        cbar_kwargs={'fraction': 0.03, 'ticks': major_ticks}    
    )
    earth.colorbar.ax.minorticks_on()
    ax.set_title(f'Global N100, 2022-05-01T{t}')
    ax.set_global()
    ax.coastlines()

    plt.tight_layout()
    plt.savefig(f'figures/frame_{i}.png')


frames = []
times = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']

for i in range(8):
    print(times[i])
    frame = plot_frame(i, times[i])
    frames.append(frame)
gif.save(frames, 'figures/global_ccn.gif', duration=150)


