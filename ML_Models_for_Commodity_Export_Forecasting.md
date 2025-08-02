# Machine Learning Models for Illinois Commodity Export Forecasting

## Executive Summary

Based on the analysis of Illinois export data with 110,711 records across multiple commodities, countries, and time periods (with time formats like 'Jul-16', 'Mar-08', 'Jun-24'), this document outlines three advanced machine learning approaches for forecasting commodity exports: **LSTM (Long Short-Term Memory)**, **State Space Models**, and **XGBoost** variants.

## Data Structure Analysis

### Dataset Characteristics:
- **Time Range**: Multiple years with abbreviated format (e.g., '08', '16', '24' representing years)
- **Records**: 110,711 export transactions
- **Time Periods**: 209 unique month-year combinations (e.g., "Jul-16", "Feb-20", "Mar-08")
- **Commodities**: Multiple commodity categories (e.g., "01 Live Animals", agricultural products, manufactured goods)
- **Countries**: Multiple export destinations across different regions
- **Target Variables**: 
  - Export Value ($US)
  - Export Weight (kg)

### Key Features for Modeling:
1. **Temporal Features**: Month, Year, Season
2. **Categorical Features**: Commodity type, Country, Region
3. **Target Variables**: Export value and weight
4. **Derived Features**: Seasonal patterns, trend components

---

## Model 1: LSTM (Long Short-Term Memory) Networks

### Model Architecture

#### **Multivariate LSTM for Commodity-Specific Forecasting**

```python
# Proposed LSTM Architecture
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.layers import Embedding, Concatenate, Input
from tensorflow.keras.models import Model

def build_commodity_lstm_model(sequence_length=12, n_commodities=50, n_countries=100):
    # Time series input (sequence_length, features)
    time_input = Input(shape=(sequence_length, 4))  # [month, year, export_value, export_weight]
    
    # Categorical inputs
    commodity_input = Input(shape=(1,))
    country_input = Input(shape=(1,))
    
    # Embedding layers for categorical features
    commodity_embed = Embedding(n_commodities, 8)(commodity_input)
    country_embed = Embedding(n_countries, 8)(country_input)
    
    # LSTM layers for time series
    lstm1 = LSTM(128, return_sequences=True)(time_input)
    lstm1 = Dropout(0.2)(lstm1)
    lstm2 = LSTM(64, return_sequences=False)(lstm1)
    lstm2 = Dropout(0.2)(lstm2)
    
    # Flatten embeddings
    commodity_flat = tf.keras.layers.Flatten()(commodity_embed)
    country_flat = tf.keras.layers.Flatten()(country_embed)
    
    # Concatenate all features
    combined = Concatenate()([lstm2, commodity_flat, country_flat])
    
    # Dense layers for prediction
    dense1 = Dense(64, activation='relu')(combined)
    dense1 = Dropout(0.2)(dense1)
    dense2 = Dense(32, activation='relu')(dense1)
    
    # Output layers (multi-output for value and weight prediction)
    value_output = Dense(1, activation='linear', name='export_value')(dense2)
    weight_output = Dense(1, activation='linear', name='export_weight')(dense2)
    
    model = Model(inputs=[time_input, commodity_input, country_input], 
                  outputs=[value_output, weight_output])
    
    return model
```

### **Data Preparation for LSTM**

```python
def prepare_lstm_data(df, sequence_length=12):
    # Sort by commodity, country, and time
    df_sorted = df.sort_values(['Commodity', 'Country', 'DateTime'])
    
    # Create sequences for each commodity-country combination
    sequences = []
    targets_value = []
    targets_weight = []
    commodities = []
    countries = []
    
    for commodity in df['Commodity'].unique():
        for country in df['Country'].unique():
            subset = df_sorted[(df_sorted['Commodity'] == commodity) & 
                             (df_sorted['Country'] == country)]
            
            if len(subset) > sequence_length:
                # Create time series sequences
                for i in range(len(subset) - sequence_length):
                    seq = subset.iloc[i:i+sequence_length][['Month', 'Year', 'Vessel Value ($US)', 'Vessel SWT (kg)']].values
                    target_val = subset.iloc[i+sequence_length]['Vessel Value ($US)']
                    target_weight = subset.iloc[i+sequence_length]['Vessel SWT (kg)']
                    
                    sequences.append(seq)
                    targets_value.append(target_val)
                    targets_weight.append(target_weight)
                    commodities.append(commodity)
                    countries.append(country)
    
    return np.array(sequences), np.array(targets_value), np.array(targets_weight), commodities, countries
```

### **LSTM Advantages:**
- **Temporal Dependencies**: Captures long-term seasonal patterns in commodity exports
- **Memory Mechanism**: Remembers important historical events (e.g., trade disruptions, seasonal cycles)
- **Multi-Output**: Can predict both export value and weight simultaneously
- **Commodity-Specific**: Learns unique patterns for different commodity types

### **LSTM Applications:**
1. **Monthly Export Forecasting**: Predict next 1-12 months of exports per commodity
2. **Seasonal Pattern Learning**: Identify agricultural vs. manufactured goods seasonality
3. **Anomaly Detection**: Detect unusual export patterns (e.g., trade disruptions)

---

## Model 2: State Space Models (Kalman Filter Based)

### **Dynamic Linear Model for Commodity Exports**

```python
import pykalman
from statsmodels.tsa.statespace import sarimax
import numpy as np

class CommodityStateSpaceModel:
    def __init__(self, n_commodities, n_countries):
        self.n_commodities = n_commodities
        self.n_countries = n_countries
        self.models = {}
    
    def build_state_space_model(self, commodity_data):
        """
        State Space Model Components:
        - Trend component (random walk with drift)
        - Seasonal component (monthly seasonality)
        - Commodity-specific effects
        - Country-specific effects
        """
        
        # State vector: [level, trend, seasonal_1, ..., seasonal_11, commodity_effect, country_effect]
        n_states = 2 + 11 + 1 + 1  # level, trend, 11 seasonal dummies, commodity, country
        
        # Transition matrix for state evolution
        transition_matrix = np.zeros((n_states, n_states))
        
        # Level and trend (random walk with drift)
        transition_matrix[0, 0] = 1  # level
        transition_matrix[0, 1] = 1  # level += trend
        transition_matrix[1, 1] = 1  # trend persistence
        
        # Seasonal component (rotating seasonal dummies)
        for i in range(11):
            if i == 0:
                transition_matrix[2, 2:13] = -1  # sum to zero constraint
            else:
                transition_matrix[2+i, 1+i] = 1
        
        # Commodity and country effects (random walk)
        transition_matrix[-2, -2] = 1
        transition_matrix[-1, -1] = 1
        
        return transition_matrix
    
    def fit_commodity_model(self, commodity, country, time_series_data):
        """
        Fit state space model for specific commodity-country combination
        """
        model = sarimax.SARIMAX(
            time_series_data,
            order=(1, 1, 1),  # ARIMA component
            seasonal_order=(1, 1, 1, 12),  # Seasonal ARIMA
            trend='ct',  # constant and trend
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_model = model.fit(disp=False)
        self.models[(commodity, country)] = fitted_model
        return fitted_model
    
    def forecast(self, commodity, country, steps=12):
        """
        Generate forecasts for specific commodity-country combination
        """
        if (commodity, country) in self.models:
            model = self.models[(commodity, country)]
            forecast = model.forecast(steps=steps)
            confidence_intervals = model.get_forecast(steps=steps).conf_int()
            return forecast, confidence_intervals
        else:
            return None, None
```

### **Hierarchical State Space Model**

```python
def build_hierarchical_state_space_model():
    """
    Multi-level state space model:
    1. National level (total Illinois exports)
    2. Commodity level (exports by commodity type)
    3. Country level (exports by destination)
    """
    
    # Level 1: National aggregate model
    national_components = {
        'trend': 'random_walk_with_drift',
        'seasonal': 'trigonometric_seasonal(12)',
        'cycle': 'stochastic_cycle(period=24)',  # 2-year business cycle
        'irregular': 'white_noise'
    }
    
    # Level 2: Commodity disaggregation
    commodity_components = {
        'commodity_trend': 'random_walk',
        'commodity_seasonal': 'seasonal_dummy(12)',
        'commodity_specific': 'autoregressive(1)'
    }
    
    # Level 3: Country disaggregation
    country_components = {
        'country_trend': 'random_walk',
        'country_seasonal': 'seasonal_dummy(12)',
        'trade_relationship': 'vector_autoregressive(2)'
    }
    
    return national_components, commodity_components, country_components
```

### **State Space Model Advantages:**
- **Decomposition**: Separates trend, seasonal, and irregular components
- **Uncertainty Quantification**: Provides confidence intervals for forecasts
- **Missing Data Handling**: Can handle irregular time series with missing observations
- **Hierarchical Structure**: Models relationships between different aggregation levels

### **State Space Applications:**
1. **Policy Analysis**: Understand impact of trade policies on export trends
2. **Seasonal Adjustment**: Decompose seasonal patterns from underlying trends
3. **Nowcasting**: Estimate current month exports before official data release

---

## Model 3: XGBoost for Commodity Export Prediction

### **Feature Engineering for XGBoost**

```python
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

def create_xgboost_features(df):
    """
    Comprehensive feature engineering for commodity export prediction
    """
    
    # Convert time column to datetime
    df['DateTime'] = pd.to_datetime(df['Time'], format='%b-%y')
    
    # Temporal features
    df['Year'] = df['DateTime'].dt.year
    df['Month'] = df['DateTime'].dt.month
    df['Quarter'] = df['DateTime'].dt.quarter
    df['DayOfYear'] = df['DateTime'].dt.dayofyear
    df['WeekOfYear'] = df['DateTime'].dt.isocalendar().week
    
    # Cyclical encoding for temporal features
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Quarter_sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
    df['Quarter_cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)
    
    # Lag features (previous months)
    df = df.sort_values(['Commodity', 'Country', 'DateTime'])
    for lag in [1, 2, 3, 6, 12]:
        df[f'export_value_lag_{lag}'] = df.groupby(['Commodity', 'Country'])['Vessel Value ($US)'].shift(lag)
        df[f'export_weight_lag_{lag}'] = df.groupby(['Commodity', 'Country'])['Vessel SWT (kg)'].shift(lag)
    
    # Rolling statistics
    for window in [3, 6, 12]:
        df[f'export_value_rolling_mean_{window}'] = df.groupby(['Commodity', 'Country'])['Vessel Value ($US)'].rolling(window).mean().reset_index(0, drop=True)
        df[f'export_value_rolling_std_{window}'] = df.groupby(['Commodity', 'Country'])['Vessel Value ($US)'].rolling(window).std().reset_index(0, drop=True)
    
    # Commodity-specific features
    commodity_stats = df.groupby('Commodity')['Vessel Value ($US)'].agg(['mean', 'std', 'median']).reset_index()
    commodity_stats.columns = ['Commodity', 'commodity_mean_value', 'commodity_std_value', 'commodity_median_value']
    df = df.merge(commodity_stats, on='Commodity', how='left')
    
    # Country-specific features
    country_stats = df.groupby('Country')['Vessel Value ($US)'].agg(['mean', 'std', 'median']).reset_index()
    country_stats.columns = ['Country', 'country_mean_value', 'country_std_value', 'country_median_value']
    df = df.merge(country_stats, on='Country', how='left')
    
    # Interaction features
    df['commodity_country_interaction'] = df['commodity_mean_value'] * df['country_mean_value']
    df['seasonal_commodity_interaction'] = df['Month'] * df['commodity_mean_value']
    
    # Economic indicators (if available)
    # df['gdp_growth'] = ...  # External economic data
    # df['trade_balance'] = ...  # Bilateral trade data
    # df['exchange_rate'] = ...  # Currency exchange rates
    
    return df

def build_xgboost_model():
    """
    XGBoost model with commodity-specific hyperparameters
    """
    
    # Multi-output XGBoost for value and weight prediction
    model_params = {
        'objective': 'reg:squarederror',
        'eval_metric': ['rmse', 'mae'],
        'max_depth': 8,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'early_stopping_rounds': 50
    }
    
    # Separate models for value and weight prediction
    value_model = xgb.XGBRegressor(**model_params)
    weight_model = xgb.XGBRegressor(**model_params)
    
    return value_model, weight_model

def train_xgboost_pipeline(df):
    """
    Complete training pipeline for XGBoost models
    """
    
    # Feature engineering
    df_features = create_xgboost_features(df)
    
    # Encode categorical variables
    le_commodity = LabelEncoder()
    le_country = LabelEncoder()
    le_region = LabelEncoder()
    
    df_features['commodity_encoded'] = le_commodity.fit_transform(df_features['Commodity'])
    df_features['country_encoded'] = le_country.fit_transform(df_features['Country'])
    df_features['region_encoded'] = le_region.fit_transform(df_features['Region'])
    
    # Select features
    feature_columns = [
        'Year', 'Month', 'Quarter', 'Month_sin', 'Month_cos', 'Quarter_sin', 'Quarter_cos',
        'commodity_encoded', 'country_encoded', 'region_encoded',
        'export_value_lag_1', 'export_value_lag_2', 'export_value_lag_3', 'export_value_lag_6', 'export_value_lag_12',
        'export_weight_lag_1', 'export_weight_lag_2', 'export_weight_lag_3', 'export_weight_lag_6', 'export_weight_lag_12',
        'export_value_rolling_mean_3', 'export_value_rolling_mean_6', 'export_value_rolling_mean_12',
        'export_value_rolling_std_3', 'export_value_rolling_std_6', 'export_value_rolling_std_12',
        'commodity_mean_value', 'commodity_std_value', 'commodity_median_value',
        'country_mean_value', 'country_std_value', 'country_median_value',
        'commodity_country_interaction', 'seasonal_commodity_interaction'
    ]
    
    X = df_features[feature_columns].fillna(0)
    y_value = df_features['Vessel Value ($US)']
    y_weight = df_features['Vessel SWT (kg)']
    
    # Time series split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Train models
    value_model, weight_model = build_xgboost_model()
    
    # Fit models with time series cross-validation
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_value_train, y_value_val = y_value.iloc[train_idx], y_value.iloc[val_idx]
        y_weight_train, y_weight_val = y_weight.iloc[train_idx], y_weight.iloc[val_idx]
        
        value_model.fit(
            X_train, y_value_train,
            eval_set=[(X_val, y_value_val)],
            verbose=False
        )
        
        weight_model.fit(
            X_train, y_weight_train,
            eval_set=[(X_val, y_weight_val)],
            verbose=False
        )
    
    return value_model, weight_model, le_commodity, le_country, le_region
```

### **XGBoost Model Variants**

#### **1. XGBoost Regressor (Standard)**
- **Use Case**: Direct prediction of export values and weights
- **Advantages**: High accuracy, feature importance analysis, handles non-linearities

#### **2. XGBRanker**
- **Use Case**: Ranking commodities by export potential for specific countries
- **Implementation**: Rank commodities within each country-month combination

#### **3. XGBoost Time Series (XGBTSRegressor)**
```python
# Custom XGBoost for time series with built-in temporal features
class XGBoostTimeSeriesRegressor:
    def __init__(self, **params):
        self.model = xgb.XGBRegressor(**params)
        self.time_features = TimeSeriesFeatureGenerator()
    
    def fit(self, X, y, date_column):
        X_with_time_features = self.time_features.fit_transform(X, date_column)
        self.model.fit(X_with_time_features, y)
        return self
    
    def predict(self, X, date_column):
        X_with_time_features = self.time_features.transform(X, date_column)
        return self.model.predict(X_with_time_features)
```

### **XGBoost Advantages:**
- **Feature Importance**: Identifies most predictive factors for commodity exports
- **Non-Linear Relationships**: Captures complex interactions between features
- **Robust to Outliers**: Handles extreme export values well
- **Fast Training**: Efficient gradient boosting implementation

---

## Model Comparison and Selection Strategy

### **Model Performance Metrics**

```python
def evaluate_models(y_true, y_pred_lstm, y_pred_ss, y_pred_xgb):
    """
    Comprehensive model evaluation
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    metrics = {
        'LSTM': {
            'MAE': mean_absolute_error(y_true, y_pred_lstm),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred_lstm)),
            'R2': r2_score(y_true, y_pred_lstm),
            'MAPE': np.mean(np.abs((y_true - y_pred_lstm) / y_true)) * 100
        },
        'State_Space': {
            'MAE': mean_absolute_error(y_true, y_pred_ss),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred_ss)),
            'R2': r2_score(y_true, y_pred_ss),
            'MAPE': np.mean(np.abs((y_true - y_pred_ss) / y_true)) * 100
        },
        'XGBoost': {
            'MAE': mean_absolute_error(y_true, y_pred_xgb),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred_xgb)),
            'R2': r2_score(y_true, y_pred_xgb),
            'MAPE': np.mean(np.abs((y_true - y_pred_xgb) / y_true)) * 100
        }
    }
    
    return metrics
```

### **Recommended Model Selection by Use Case**

| Use Case | Primary Model | Secondary Model | Reasoning |
|----------|---------------|-----------------|-----------|
| **Long-term Strategic Planning** | State Space Model | LSTM | Uncertainty quantification, trend decomposition |
| **Short-term Operational Forecasting** | XGBoost | LSTM | High accuracy, fast predictions |
| **Seasonal Pattern Analysis** | LSTM | State Space | Memory of complex seasonal patterns |
| **Policy Impact Assessment** | State Space Model | XGBoost | Structural break detection, causal inference |
| **Real-time Monitoring** | XGBoost | LSTM | Fast inference, feature importance |

---

## Implementation Roadmap

### **Phase 1: Data Preparation (2-3 weeks)**
1. **Data Cleaning**: Handle missing values, outliers, data type conversions
2. **Feature Engineering**: Create temporal features, lag variables, rolling statistics
3. **Data Validation**: Ensure data quality and consistency

### **Phase 2: Baseline Models (2-4 weeks)**
1. **Simple Models**: Naive forecasts, seasonal naive, linear regression
2. **Traditional Time Series**: ARIMA, exponential smoothing
3. **Performance Benchmarking**: Establish baseline performance metrics

### **Phase 3: Advanced Model Development (4-6 weeks)**
1. **LSTM Implementation**: Build and tune neural network architecture
2. **State Space Models**: Implement Kalman filter-based models
3. **XGBoost Pipeline**: Feature engineering and hyperparameter tuning

### **Phase 4: Model Evaluation and Selection (2-3 weeks)**
1. **Cross-Validation**: Time series split validation
2. **Performance Comparison**: Compare models across different metrics
3. **Ensemble Methods**: Combine best-performing models

### **Phase 5: Deployment and Monitoring (2-4 weeks)**
1. **Model Deployment**: Production-ready prediction pipeline
2. **Monitoring System**: Track model performance over time
3. **Retraining Pipeline**: Automated model updates with new data

---

## Expected Outcomes and Business Value

### **Forecasting Accuracy Targets**
- **Short-term (1-3 months)**: 85-95% accuracy for aggregate exports
- **Medium-term (6-12 months)**: 75-85% accuracy with confidence intervals
- **Long-term (12+ months)**: 60-75% accuracy with trend identification

### **Business Applications**
1. **Supply Chain Optimization**: Plan inventory and logistics based on export forecasts
2. **Market Development**: Identify emerging opportunities in specific countries/commodities
3. **Risk Management**: Predict potential export disruptions and market volatility
4. **Policy Support**: Inform trade policy decisions with data-driven insights

### **Key Success Factors**
- **Data Quality**: Consistent, complete, and timely export data
- **Feature Engineering**: Domain expertise in international trade patterns
- **Model Interpretability**: Ability to explain predictions to stakeholders
- **Continuous Learning**: Regular model updates with new data and feedback

---

## Conclusion

The combination of LSTM, State Space Models, and XGBoost provides a comprehensive approach to forecasting Illinois commodity exports. Each model offers unique strengths:

- **LSTM**: Superior for capturing complex temporal patterns and long-term dependencies
- **State Space Models**: Excellent for uncertainty quantification and structural analysis
- **XGBoost**: Outstanding for high-accuracy predictions and feature importance analysis

The recommended approach is to implement all three models and create an ensemble system that leverages the strengths of each approach, providing robust and accurate forecasts for strategic decision-making in international trade.
