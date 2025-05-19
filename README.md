# NDVI Time Series (2014–2018): PROBA-V + Sentinel-3-LFR

## 📌 Objectives

- Build a harmonized NDVI series over **Limuru**
- Combine **PROBA-V NDVI (2014–2017)** with **Sentinel-3 LFR (Land Full Resolution, 2017–2018)**
- Use reflectance bands from LFR `.nc` files for NDVI computation
- Integrate and compare NDVI derived from both **Sentinel-3 LFR (local Level 2)** and **Sentinel-3 OLCI (cloud-based Level 1)** data
- Apply radiance scaling to Sentinel-3 OLCI radiance bands to compute NDVI
- Compare and analyze NDVI outputs from four different processing routes across varying time ranges

## 🛰️ Datasets Used

### 1. **PROBA-V NDVI**

- **Source**: [VITO/PROBAV/C1/S1_TOC_333M](https://developers.google.com/earth-engine/datasets/catalog/VITO_PROBAV_C1_S1_TOC_333M)
- **Period**: 2014–2017
- **Spatial Resolution**: 333m
- **Processing**: GEE-based monthly composites using `.select('NDVI')`

### 2. **Sentinel-3 LFR (Local Level 2)**

- **Product**: `rc_gifapar.nc` file inside zipped `.SEN3` directories
- **Bands Used**:
  - `RC681` (Red)
  - `RC865` (NIR)
- **Period**: 2017–2018
- **Processing**:
  - NDVI = (RC865 - RC681) / (RC865 + RC681)
  - NDVI calculated per file and stacked by date

### 3. **Sentinel-3 OLCI (Cloud-Based Radiance)**

- **Source**: Copernicus OLCI L1 Radiance products via GEE or direct access
- **Bands Used**:
  - Radiance bands mapped to RED and NIR
- **Processing**:
  - Radiance values scaled to reflectance
  - NDVI derived post-scaling and temporally aligned with other sources

## 🛠️ Tools & Libraries

- Google Earth Engine Python API
- `xarray`, `numpy`, `matplotlib`, `scipy`, `pandas`, `glob`, `zipfile`
- `scipy.interpolate` and rolling filters for temporal smoothing and gap-filling
- `matplotlib` for multi-series NDVI visualization across data sources

## ⚙️ Workflow Highlights

- Harmonized NDVI time series constructed from multi-sensor input
- Temporal smoothing applied to handle missing values and ensure consistency across months
- Sentinel-3 NDVI computed from both reflectance (LFR) and radiance-scaled (Level 1) data
- All NDVI results plotted side-by-side to analyze consistency, divergence, and temporal dynamics across sources
