# Predicting Tornado Days from Environmental Parameters

A machine learning project that predicts tornado occurrence in Kansas using environmental parameters from ERA5 reanalysis data.

## Project Overview

This project uses environmental parameters (CAPE, temperature, dewpoint, winds) from ECMWF ERA5 reanalysis data combined with historical tornado records from NOAA's Storm Events Database to train machine learning models that predict tornado days.

### Key Features

- **Data Acquisition**: Fetches ERA5 data via OPeNDAP from NCAR's THREDDS server
- **Multiple ML Models**: Compares Logistic Regression, Random Forest, Gradient Boosting, SVM, and Neural Networks
- **Class Imbalance Handling**: Uses balanced class weights and undersampling for rare tornado events
- **Comprehensive Evaluation**: ROC curves, confusion matrices, cross-validation, and feature importance analysis

## Installation

1. Clone or download this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Open `project.ipynb` in Jupyter Notebook or JupyterLab
2. Run cells sequentially from top to bottom
3. The first run will take several minutes to fetch ERA5 data from remote servers
4. After the first run, the dataset is cached to `tornado_dataset.csv` for faster re-runs

### Data Sources

The notebook automatically downloads data from:
- **Storm Events**: [NCEI Storm Events CSV Files](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/) - Downloaded and filtered for Kansas tornadoes
- **ERA5**: NCAR THREDDS OPeNDAP server - Fetched on-the-fly

## Project Structure

```
Module 8/
├── project.ipynb          # Main Jupyter notebook
├── utilities.py           # Helper functions for data caching
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── proposal.md            # Project proposal
```

## Data Sources

- **ERA5 Reanalysis**: ECMWF ERA5 hourly data accessed via NCAR's THREDDS OPeNDAP server
  - CAPE (Convective Available Potential Energy)
  - 2m Temperature and Dewpoint
  - 10m U/V Wind Components
  - Surface Pressure

- **Storm Events**: [NCEI Storm Events Database](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/)
  - Automatically downloaded from NCEI FTP archive
  - Historical tornado occurrence records
  - County-level location data

## Environmental Parameters Used

| Parameter | Description | Relevance to Tornadoes |
|-----------|-------------|------------------------|
| CAPE | Convective Available Potential Energy | Atmospheric instability measure |
| T2M | 2m Temperature | Surface heating |
| D2M | 2m Dewpoint | Moisture availability |
| U10/V10 | 10m Wind Components | Low-level wind patterns |
| SP | Surface Pressure | Synoptic conditions |
| Wind Speed | Derived from U10/V10 | Storm motion potential |
| Dewpoint Depression | T2M - D2M | Atmospheric moisture content |

## Machine Learning Models

Five classification models are compared:

1. **Logistic Regression** - Baseline linear model
2. **Random Forest** - Ensemble of decision trees
3. **Gradient Boosting** - Sequential ensemble method
4. **SVM (RBF kernel)** - Support vector machine
5. **Neural Network** - Multi-layer perceptron

## Results

The notebook generates:
- Model comparison table (Accuracy, Precision, Recall, F1, ROC-AUC)
- ROC curves for all models
- Confusion matrices
- Feature importance rankings
- Cross-validation analysis

## Limitations

- Spatial aggregation to state-level means loses local variability
- Limited to surface and near-surface parameters
- Tornado events are rare, creating class imbalance challenges
- Does not include wind shear or storm-relative helicity

## Future Improvements

- Add upper-level wind data for shear calculations
- Use county-level spatial resolution
- Incorporate temporal sequences (multi-day patterns)
- Ensemble multiple models for final predictions

## References

- [ECMWF ERA5 Documentation](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
- [NOAA Storm Events Database](https://www.ncdc.noaa.gov/stormevents/)
- [NCAR Research Data Archive](https://rda.ucar.edu/)

## License

This project is for educational purposes as part of AMTS-523 coursework.

