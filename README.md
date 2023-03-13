
# ccn-globally

Model cloud condensation nuclei globally

![](figures/global_ccn.gif)

## Environment

```
conda create -n n100 python=3.7.0
conda install -c conda-forge py-xgboost-gpu -y
conda install -c conda-forge jupyterlab matplotlib -y
conda install -c anaconda pandas jupyter -y
```

```
conda create -n geo
conda install -c conda-forge xarray xgboost cartopy cfgrib
pip install gif
```