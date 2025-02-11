import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import folium
from statsmodels.tsa.seasonal import seasonal_decompose
def import_data(file_path, file_type):
    """
    Import climate data from various file formats and perform initial data cleaning.
    
    Args:
    file_path (str): Path to the data file
    file_type (str): Type of file ('csv', 'json', 'excel')
    
    Returns:
    pd.DataFrame: Cleaned and processed climate data
    """
    if file_type == 'csv':
        df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    elif file_type == 'json':
        df = pd.read_json(file_path)
    elif file_type == 'excel':
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type")
    
    # Ensure the index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Perform basic data cleaning
    df.dropna(inplace=True)
    
    return df
def analyze_trends(data, variable, time_period):
    """
    Conduct trend analysis on specified climate variables over given time periods.
    
    Args:
    data (pd.DataFrame): Climate data
    variable (str): Climate variable to analyze
    time_period (str): Time period for analysis ('monthly', 'yearly')
    
    Returns:
    dict: Results of trend analysis
    """
    if time_period == 'monthly':
        resampled_data = data[variable].resample('M').mean()
    elif time_period == 'yearly':
        resampled_data = data[variable].resample('Y').mean()
    else:
        raise ValueError("Unsupported time period")
    
    # Perform linear regression
    x = np.arange(len(resampled_data))
    y = resampled_data.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    results = {
        'trend': {'slope': slope, 'intercept': intercept, 'r_value': r_value, 'p_value': p_value},
    }
    
    # Perform seasonal decomposition if enough data is available
    if len(resampled_data) >= 24:  # At least 2 years of monthly data
        decomposition = seasonal_decompose(resampled_data, model='additive', period=12)
        results['seasonal'] = decomposition.seasonal
        results['residual'] = decomposition.resid
    else:
        print(f"Warning: Not enough data for seasonal decomposition. Need at least 24 observations, but only have {len(resampled_data)}.")
    
    return results

def visualize_data(data, plot_type, **kwargs):
    """
    Generate various types of visualizations based on the processed climate data.
    
    Args:
    data (pd.DataFrame or pd.Series): Climate data
    plot_type (str): Type of plot to generate
    **kwargs: Additional arguments for specific plot types
    
    Returns:
    matplotlib.figure.Figure: Generated plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    if plot_type == 'line':
        variable = kwargs.get('variable', data.columns[0])
        ax.plot(data.index, data[variable])
        ax.set_title(f'{variable} over time')
        ax.set_xlabel('Date')
        ax.set_ylabel(variable)
    
    elif plot_type == 'heatmap':
        variable = kwargs.get('variable', data.columns[0])
        pivot_data = data.pivot(columns=data.index.month, values=variable)
        sns.heatmap(pivot_data, ax=ax, cmap='YlOrRd')
        ax.set_title(f'Heatmap of {variable}')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
    
    elif plot_type == 'boxplot':
        variable = kwargs.get('variable', data.columns[0])
        data.boxplot(column=variable, by=data.index.month, ax=ax)
        ax.set_title(f'Monthly distribution of {variable}')
        ax.set_xlabel('Month')
        ax.set_ylabel(variable)
    
    else:
        raise ValueError("Unsupported plot type")
    
    plt.tight_layout()
    return fig
def calculate_climate_index(data, index_type, **params):
    """
    Compute different climate indices based on the input data and specified parameters.
    
    Args:
    data (pd.DataFrame): Climate data
    index_type (str): Type of climate index to calculate
    **params: Additional parameters for specific index calculations
    
    Returns:
    pd.Series: Calculated climate index
    """
    if index_type == 'SPI':
        # Standardized Precipitation Index calculation
        if 'precipitation' not in data.columns:
            raise ValueError("Precipitation data not found in the DataFrame")
        precipitation = data['precipitation']
        spi = (precipitation - precipitation.mean()) / precipitation.std()
        return pd.Series(spi, name='SPI', index=data.index)
    
    elif index_type == 'GDD':
        # Growing Degree Days calculation
        base_temp = params.get('base_temp', 10)
        if 'temperature' in data.columns:
            temp = data['temperature']
            gdd = np.maximum(temp - base_temp, 0).cumsum()
        elif 'temp_max' in data.columns and 'temp_min' in data.columns:
            temp_max = data['temp_max']
            temp_min = data['temp_min']
            gdd = np.maximum(((temp_max + temp_min) / 2) - base_temp, 0).cumsum()
        else:
            raise ValueError("Required temperature data not found in the DataFrame")
        return pd.Series(gdd, name='GDD', index=data.index)
    
    else:
        raise ValueError("Unsupported climate index")


def main():
    file_path = 'larger_sample_climate_data.csv'
    climate_data = import_data(file_path, 'csv')
    
    # Analyze temperature trends
    temp_trends = analyze_trends(climate_data, 'temperature', 'monthly')
    print("Temperature trend analysis:", temp_trends['trend'])
    
    # Visualize temperature data
    temp_plot = visualize_data(climate_data, 'line', variable='temperature')
    temp_plot.savefig('temperature_trend.png')
    
    # Calculate and visualize SPI
    spi = calculate_climate_index(climate_data, 'SPI')
    spi_plot = visualize_data(spi, 'line', variable='SPI')
    spi_plot.savefig('spi_index.png')
    
    print("Analysis complete. Check the generated plots for visual results.")

if __name__ == "__main__":
    main()