import xarray as xr
import xgboost as xgb
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import gif


@gif.frame
def plot_frame(ds, date, time):

    # Load & preprocess data
    df = ds.to_dataframe()
    features = ['co', 'c5h8', 'no', 'no2', 'so2']
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
    vmin, vmax = 5.6, 6760
    major_ticks = [10, 100, 1000, 10000]
    fig, ax = plt.subplots(nrows=1, subplot_kw={'projection': ccrs.Robinson()})
    earth = result.n100.plot.contourf(
        ax=ax, transform=ccrs.PlateCarree(), levels=np.geomspace(vmin, vmax, 30), 
        norm=LogNorm(vmin=vmin, vmax=vmax), cmap='viridis', extend='neither',
        cbar_kwargs={'fraction': 0.03, 'ticks': major_ticks}    
    )
    earth.colorbar.ax.minorticks_on()
    ax.set_title(f'Global N100, {date}T{time}')
    ax.set_global()
    ax.coastlines()

    plt.tight_layout()
    plt.savefig(f'figures/frame_{date}_{time}.png')


frames = []
times = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']

start_date = datetime(2022, 5, 1)
end_date = datetime(2022, 5, 31)
delta = timedelta(days=1)

while start_date <= end_date:
    current_date = str(start_date.date())
    for i, time in enumerate(times):
        # Prepare data
        ds = xr.open_dataset(f'data/global/{current_date}.grib', engine='cfgrib')
        frame = plot_frame(ds.isel(time=i), current_date, time)
        frames.append(frame)
        
    start_date += delta

gif.save(frames, 'figures/global_ccn.gif', duration=150)
