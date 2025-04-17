import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import streamlit as st
from scipy.stats import skew, kurtosis, zscore
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def display_plot():
    fig = plt.gcf()
    st.pyplot(fig)
    plt.clf()

def merge_csv_files(directory, output_file):
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not all_files:
        logging.warning("No CSV files found in the directory.")
        st.warning("No CSV files found in the specified CSV directory.")
        return None
    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(os.path.join(directory, file), encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            logging.warning(f"Encoding issue with {file}, trying latin-1 encoding.")
            df = pd.read_csv(os.path.join(directory, file), encoding='latin-1', low_memory=False)
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
            continue
        df_list.append(df)
    if not df_list:
        logging.error("No CSV files could be loaded successfully.")
        return None
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    logging.info(f"Merged {len(all_files)} CSV files into {output_file} with {merged_df.shape[0]} records and {merged_df.shape[1]} features.")
    st.success(f"Merged {len(all_files)} CSV files into {output_file}")
    return merged_df

def merge_json_files(directory, output_file):
    all_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    if not all_files:
        logging.warning("No JSON files found in the directory.")
        st.warning("No JSON files found in the specified JSON directory.")
        return None
    df_list = []
    for file in all_files:
        try:
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Assuming JSON structure: {"response": {"data": [...]}}
            if "response" in data and "data" in data["response"]:
                df = pd.DataFrame(data["response"]["data"])
                df_list.append(df)
            else:
                logging.warning(f"Unexpected JSON structure in {file}")
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
            continue
    if not df_list:
        logging.error("No JSON files could be loaded successfully.")
        return None
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    logging.info(f"Merged {len(all_files)} JSON files into {output_file} with {merged_df.shape[0]} records and {merged_df.shape[1]} features.")
    st.success(f"Merged {len(all_files)} JSON files into {output_file}")
    return merged_df

def combine_csv_files(file1, file2, output_file):
    try:
        df1 = pd.read_csv(file1, low_memory=False)
        df2 = pd.read_csv(file2, low_memory=False)
    except Exception as e:
        logging.error(f"Error combining files: {e}")
        st.error("Error combining files.")
        return None
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    logging.info(f"Combined {file1} and {file2} into {output_file} with {merged_df.shape[0]} records and {merged_df.shape[1]} features.")
    st.success(f"Combined {file1} and {file2} into {output_file}")
    return merged_df

# Task # 02:  --- Data Preprocessing ---
def preprocess_data(file):
    df = pd.read_csv(file, low_memory=False)
    
    # Data Cleaning and Consistency
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    logging.info("Missing Data Percentage per Column:\n" + str(missing_percentage))
    df = df.dropna(thresh=len(df) * 0.5, axis=1)
    

    # Task # 2(a): Handling missing values: using mode for categorical and median for numeric (assumed MAR)
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
        elif df[col].dtype in ['float64', 'int64']:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
    
    # Task # 2(b): Data Type Conversions and Feature Extraction for 'period'
    if 'period' in df.columns:
        # Convert to datetime, forward-fill, and force datetime64 dtype
        df['period'] = pd.to_datetime(df['period'], errors='coerce').ffill()
        df['period'] = df['period'].astype('datetime64[ns]')
        df['hour'] = df['period'].dt.hour
        df['day'] = df['period'].dt.day
        df['month'] = df['period'].dt.month
        df['year'] = df['period'].dt.year
        df['day_of_week'] = df['period'].dt.dayofweek
    
    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    

    # Task # 2(c): Handling Duplicates and Inconsistencies
    before_duplicates = df.shape[0]
    df.drop_duplicates(inplace=True)
    logging.info(f"Removed {before_duplicates - df.shape[0]} duplicate rows.")
    
    # Task # 2(d): Feature Engineering: Create 'is_weekend'
    if 'day_of_week' in df.columns:
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    df.to_csv(file, index=False)
    logging.info(f"Preprocessed data saved to {file} with {df.shape[0]} records and {df.shape[1]} features.")

    perform_eda(df)

    # Execute remaining tasks before EDA:
    df = detect_handle_outliers(df)
    regression_modeling(df)

    return df

# Task # 3: --- Exploratory Data Analysis ---
def perform_eda(df):
    st.info("Performing Exploratory Data Analysis (EDA)...")

    # Task # 3(a): Summary Statistics
    st.write("### Statistical Summary")
    st.write(df.describe())
    st.write("### Skewness")
    st.write(df.skew(numeric_only=True))
    st.write("### Kurtosis")
    st.write(df.kurtosis(numeric_only=True))
    
    # Task # 3(b): Time Series Analysis
    if 'period' in df.columns and 'value' in df.columns:
        st.write('### Time Series Analysis')

        # Ensure 'period' is in datetime format
        df['period'] = pd.to_datetime(df['period'], dayfirst=True)

        # Aggregate value by period to get total demand
        agg_df = df.groupby('period', as_index=False)['value'].sum()
        agg_df = agg_df.sort_values('period')

        # Calculate rolling average for trend
        agg_df.set_index('period', inplace=True)
        rolling_window = '30D'  # 30-day rolling average
        agg_df['rolling_avg'] = agg_df['value'].rolling(window=rolling_window, min_periods=1).mean()
        agg_df.reset_index(inplace=True)
    
    # Down sample for plotting if necessary
    if len(agg_df) > 10000:
        sample_df = agg_df.iloc[::100]
    else:
        sample_df = agg_df
    
    # Plot 1: Main time series with trend and irregularities
    plt.figure(figsize=(12, 6))
    plt.plot(sample_df['period'], sample_df['value'], label='Total Electricity Demand', alpha=0.5)
    plt.plot(sample_df['period'], sample_df['rolling_avg'], label='30-Day Rolling Average', color='red')
    
    # Highlight irregularities
    agg_df['diff'] = agg_df['value'] - agg_df['rolling_avg']
    std_diff = agg_df['diff'].std()
    threshold = 2 * std_diff  # Define irregularities as 2 standard deviations from trend
    irregularities = agg_df[abs(agg_df['diff']) > threshold]
    if not irregularities.empty:
        for _, row in irregularities.iterrows():
            if row['period'] in sample_df['period'].values:
                plt.scatter(row['period'], row['value'], color='orange', zorder=5, label='Irregularity' if 'Irregularity' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.title("Total Electricity Demand Over Time with Trend and Irregularities")
    plt.xlabel("Time")
    plt.ylabel("Demand (Megawatthours)")
    plt.grid(True)
    plt.legend()
    if len(agg_df) > len(sample_df):
        plt.annotate('Downsampled for efficiency', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, color='gray')
    st.pyplot(plt)  # Assuming display_plot() uses st.pyplot()
    
    # Plot 2: One week to highlight seasonal patterns
    if len(agg_df) >= 168:  # 7 days * 24 hours
        week_df = agg_df.iloc[:168]  # First week
        plt.figure(figsize=(12, 6))
        plt.plot(week_df['period'], week_df['value'], label='Total Electricity Demand')
        
        # Shade nighttime hours (10 PM to 6 AM)
        for day in week_df['period'].dt.date.unique():
            night_start = pd.Timestamp(day) + pd.Timedelta(hours=22)
            night_end = pd.Timestamp(day) + pd.Timedelta(days=1, hours=6)
            if night_start in week_df['period'].values and night_end in week_df['period'].values:
                plt.axvspan(night_start, night_end, color='gray', alpha=0.3, label='Nighttime' if 'Nighttime' not in plt.gca().get_legend_handles_labels()[1] else "")
        
        plt.title("Electricity Demand Over One Week with Nighttime Shaded")
        plt.xlabel("Time")
        plt.ylabel("Demand (Megawatthours)")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)
    
    # Task # 3(c): Univariate Analysis
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns[:5]

    for col in numerical_cols:
        st.write(f"### Analysis for {col}")

        # Handle large datasets by sampling if necessary
        if len(df) > 10000:
            sample_data = df[col].sample(10000, random_state=42)
        else:
            sample_data = df[col]

        # 1. Histogram with KDE
        plt.figure(figsize=(12, 4))
        sns.histplot(sample_data, kde=True)
        plt.title(f"Histogram with KDE for {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)
        st.pyplot(plt)
        plt.clf()  # Clear the figure to avoid overlap

        # 2. Boxplot
        plt.figure(figsize=(12, 4))
        sns.boxplot(x=sample_data)
        plt.title(f"Boxplot for {col}")
        plt.xlabel(col)
        plt.grid(True)
        st.pyplot(plt)
        plt.clf()

        # 3. Density Plot
        plt.figure(figsize=(12, 4))
        sns.kdeplot(sample_data, fill=True)
        plt.title(f"Density Plot for {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.grid(True)
        st.pyplot(plt)
        plt.clf()

        # Calculate descriptive statistics
        mean_val = sample_data.mean()
        median_val = sample_data.median()
        mode_val = sample_data.mode().values[0] if not sample_data.mode().empty else "N/A"
        range_val = sample_data.max() - sample_data.min()
        variance_val = sample_data.var()
        std_dev_val = sample_data.std()
        skewness = skew(sample_data.dropna())
        kurt = kurtosis(sample_data.dropna())

        # Distribution Shape
        st.write(f"**Distribution Shape:**")
        if abs(skewness) < 0.5:
            shape = "approximately normal"
        elif skewness > 0:
            shape = "right-skewed"
        else:
            shape = "left-skewed"
        st.write(f"- The distribution is {shape} (skewness: {skewness:.2f}).")
        st.write(f"- Kurtosis: {kurt:.2f} ({'leptokurtic' if kurt > 0 else 'platykurtic' if kurt < 0 else 'mesokurtic'}).")

        # Central Tendency
        st.write(f"**Central Tendency:**")
        st.write(f"- Mean: {mean_val:.2f}")
        st.write(f"- Median: {median_val:.2f}")
        st.write(f"- Mode: {mode_val}")

        # Dispersion
        st.write(f"**Dispersion:**")
        st.write(f"- Range: {range_val:.2f}")
        st.write(f"- Variance: {variance_val:.2f}")
        st.write(f"- Standard Deviation: {std_dev_val:.2f}")

        # Outliers (based on IQR method)
        q1 = sample_data.quantile(0.25)
        q3 = sample_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = sample_data[(sample_data < lower_bound) | (sample_data > upper_bound)]
        if not outliers.empty:
            st.write(f"- Outliers detected: {len(outliers)} values outside {lower_bound:.2f} to {upper_bound:.2f}.")
        else:
            st.write("- No significant outliers detected.")
    
    # Task # 3(d): Correlation Analysis
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    display_plot()
    
    # Task # 3(e): Advanced Time Series Techniques
    if 'value' in df.columns and len(df) > 1000:
    # Time Series Decomposition
        try:
            # Perform decomposition to isolate trend, seasonal, and residual components
            decomposition = seasonal_decompose(
                df['value'].iloc[:1000],  # Limiting to first 1000 points for performance
                period=24,               # Assuming hourly data with daily seasonality
                model='additive',        # Additive model (could also use 'multiplicative')
                extrapolate_trend='freq' # Handles missing values in trend
            )

            # Visualize the decomposition components
            fig = decomposition.plot()
            plt.suptitle("Seasonal Decomposition of Electricity Demand", y=1.02)
            display_plot()  # Assuming this is a custom function to show plot

            # Optional: Store components for further analysis
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid

        except Exception as e:
            logging.error(f"Seasonal decomposition failed: {e}")
            st.error("Seasonal decomposition failed.")

        # Augmented Dickey-Fuller Test for Stationarity
        try:
            # Conduct ADF test on cleaned data
            result = adfuller(df['value'].dropna())

            # Display results in a formatted way
            st.write("### Augmented Dickey-Fuller Test Results")
            st.write(f"**ADF Test Statistic:** {result[0]:.4f}")
            st.write(f"**p-value:** {result[1]:.4f}")
            st.write("**Critical Values:**")
            for key, value in result[4].items():
                st.write(f"   {key}: {value:.4f}")

            # Interpret the results
            if result[1] < 0.05:
                st.success("The time series is Stationary (p < 0.05)")
            else:
                st.warning("The time series is Non-stationary (p ≥ 0.05)")

        except Exception as e:
            logging.error(f"ADF test failed: {e}")
            st.error("ADF test failed.")

# Task # 4: --- Outlier Detection and Handling ---
def detect_handle_outliers(df):
    # Task # 4(a): Detection Methods
    st.info("Detecting and handling outliers in the 'value' column...")
    if 'value' not in df.columns:
        st.warning("'value' column not found for outlier detection.")
        return df

    # Make a copy of the original 'value' data
    original_values = df['value'].copy()

    # IQR-based detection
    Q1 = original_values.quantile(0.25)
    Q3 = original_values.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound_iqr = Q1 - 1.5 * IQR
    upper_bound_iqr = Q3 + 1.5 * IQR
    outliers_iqr = (original_values < lower_bound_iqr) | (original_values > upper_bound_iqr)
    num_outliers_iqr = outliers_iqr.sum()

    # Z-score detection
    z_scores = zscore(original_values)
    outliers_z = np.abs(z_scores) > 3
    num_outliers_z = outliers_z.sum()

    # Logging the number of outliers detected
    logging.info(f"IQR method detected {num_outliers_iqr} outliers. Z-score detected {num_outliers_z} outliers.")
    st.write(f"**Outlier Detection Results**:")
    st.write(f"- IQR method detected {num_outliers_iqr} outliers.")
    st.write(f"- Z-score method detected {num_outliers_z} outliers.")

    # Evaluate impact of outliers
    st.write("**Evaluating Impact of Outliers**:")
    st.write(f"Mean before handling: {original_values.mean():.2f}")
    st.write(f"Standard deviation before handling: {original_values.std():.2f}")

    # Before handling visualization
    st.write("**Boxplot Before Handling Outliers**:")
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=original_values)
    plt.title("Boxplot of 'value' Before Outlier Handling")
    display_plot()

    # Handling strategy: Decide to cap using IQR bounds
    st.info("Handling Strategy: Capping (Winsorizing) outliers using IQR bounds. Rationale: Capping retains data points while reducing the influence of extremes. IQR is robust to non-normal distributions, which is suitable for this dataset.")
    df['value'] = original_values.clip(lower=lower_bound_iqr, upper=upper_bound_iqr)

    # After handling visualization
    st.write("**Boxplot After Handling Outliers**:")
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df['value'])
    plt.title("Boxplot of 'value' After Outlier Handling (Capped via IQR)")
    display_plot()

    # Evaluate post-handling statistics
    st.write("**Post-Handling Evaluation**:")
    st.write(f"Mean after handling: {df['value'].mean():.2f}")
    st.write(f"Standard deviation after handling: {df['value'].std():.2f}")

    return df

# Task # 5: --- Regression Modeling ---
def regression_modeling(df):
    st.info("Running regression modeling to predict electricity demand...")
    
    # Feature Selection
    potential_features = [
        'hour', 'day', 'month','year', 'day_of_week', 'is_weekend'
    ]
    
    # Check for required columns
    available_features = [feat for feat in potential_features if feat in df.columns]
    if not available_features:
        st.error("No predictor features found in the dataset.")
        return
    if 'value' not in df.columns:
        st.error("Target variable 'value' (electricity demand) not found.")
        return
    
    missing_features = set(potential_features) - set(available_features)
    if missing_features:
        st.warning(f"Missing features: {', '.join(missing_features)}. Model performance may be affected.")
    
    X = df[available_features]
    y = df['value']
    
    # Model Development: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    st.write(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")
    
    # Build and train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    st.write("### Regression Model Evaluation")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.3f}")
    
    # Feature importance
    st.write("### Feature Coefficients")
    for feature, coef in zip(available_features, model.coef_):
        st.write(f"{feature}: {coef:.4f}")
    
    # Visualization: Actual vs Predicted
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(y_test, y_pred, alpha=0.5, label='Predictions')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
    ax.set_xlabel("Actual Demand")
    ax.set_ylabel("Predicted Demand")
    ax.set_title("Actual vs Predicted Electricity Demand")
    ax.legend()
    plt.tight_layout()
    display_plot()
    
    # Residual Analysis
    residuals = y_test - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(residuals, kde=True, ax=ax1)
    ax1.set_title("Residuals Distribution")
    ax1.set_xlabel("Residual Value")
    ax1.set_ylabel("Frequency")
    
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title("Residuals vs Predicted")
    ax2.set_xlabel("Predicted Values")
    ax2.set_ylabel("Residuals")
    plt.tight_layout()
    display_plot()
    
    st.write("### Model Performance Analysis")
    residual_mean = residuals.mean()
    residual_std = residuals.std()
    st.write(f"- Residual Mean: {residual_mean:.4f} (should be close to 0)")
    st.write(f"- Residual Std: {residual_std:.4f}")
    
    commentary = []
    if r2 > 0.7:
        commentary.append("Good model fit (R² > 0.7)")
    elif r2 > 0.5:
        commentary.append("Moderate model fit (0.5 < R² ≤ 0.7)")
    else:
        commentary.append("Poor model fit (R² ≤ 0.5)")
    
    if abs(residual_mean) > rmse * 0.1:
        commentary.append("Potential bias in predictions (residual mean not near zero)")
    if residual_std > rmse:
        commentary.append("High variability in residuals")
    
    st.write("- Observations: " + "; ".join(commentary))
    
    # Log results
    logging.info(f"""
    Regression Model Results:
    - Features used: {', '.join(available_features)}
    - MSE: {mse:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}
    - Residual Mean: {residual_mean:.4f}, Residual Std: {residual_std:.4f}
    """)
    
    return model


# --Streamlit Formation --------------------------------
st.title("Electricity Demand Data Analysis")
st.write("This application merges, preprocesses, and analyzes electricity demand data with an interactive GUI.")

st.sidebar.header("Configuration")
csv_directory = st.sidebar.text_input("CSV Directory", "/path/to/weather_raw_data")
json_directory = st.sidebar.text_input("JSON Directory", "/path/to/electricity_raw_data")
csv_output_file = st.sidebar.text_input("CSV Output File", "merged_weather_data.csv")
json_output_file = st.sidebar.text_input("JSON Output File", "merged_electricity_data.csv")
final_output_file = st.sidebar.text_input("Final Output File", "merged_file.csv")

if st.sidebar.button("Run Full Pipeline"):
    st.info("Running data integration...")
    df_csv = merge_csv_files(csv_directory, csv_output_file)
    df_json = merge_json_files(json_directory, json_output_file)
    
    if df_csv is not None and df_json is not None:
        combined_df = combine_csv_files(csv_output_file, json_output_file, final_output_file)
        if combined_df is not None:
            st.info("Running preprocessing, outlier handling, regression modeling, and EDA...")
            preprocess_data(final_output_file)
        else:
            st.error("Failed to combine CSV and JSON files.")
    else:
        st.error("One or both of the merging steps failed. Please check your directories.")

st.write("Check the logs on the right for progress and details.")