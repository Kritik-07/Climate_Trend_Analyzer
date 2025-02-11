import pytest
import pandas as pd
import numpy as np
from climate import import_data, analyze_trends, calculate_climate_index

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
    temperatures = np.random.normal(15, 5, len(dates))
    precipitation = np.random.gamma(2, 2, len(dates))
    return pd.DataFrame({'temperature': temperatures, 'precipitation': precipitation}, index=dates)

def test_import_data(tmp_path):
    # Create a temporary CSV file
    df = pd.DataFrame({'date': ['2022-01-01', '2022-01-02'], 'temperature': [10, 12]})
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    
    # Test the import_data function
    imported_data = import_data(file_path, 'csv')
    assert isinstance(imported_data, pd.DataFrame)
    assert len(imported_data) == 2
    assert 'temperature' in imported_data.columns
    assert imported_data.index.name == 'date'

def test_analyze_trends(sample_data):
    trends = analyze_trends(sample_data, 'temperature', 'monthly')
    assert isinstance(trends, dict)
    assert 'trend' in trends
    assert 'seasonal' in trends
    assert 'residual' in trends
    assert 'slope' in trends['trend']
    assert 'intercept' in trends['trend']
    assert 'r_value' in trends['trend']
    assert 'p_value' in trends['trend']

def test_calculate_climate_index(sample_data):
    # Test SPI calculation
    spi = calculate_climate_index(sample_data, 'SPI')
    assert isinstance(spi, pd.Series)
    assert spi.name == 'SPI'
    assert len(spi) == len(sample_data)
    assert spi.mean() == pytest.approx(0, abs=0.1)
    assert spi.std() == pytest.approx(1, abs=0.1)

    # Test GDD calculation
    gdd = calculate_climate_index(sample_data, 'GDD', base_temp=10)
    assert isinstance(gdd, pd.Series)
    assert gdd.name == 'GDD'
    assert len(gdd) == len(sample_data)
    # assert (gdd.diff() >= 0).all()  # GDD should be non-decreasing

    # Test error for unsupported index type
    with pytest.raises(ValueError):
        calculate_climate_index(sample_data, 'UNSUPPORTED_INDEX')

    # Test error for missing precipitation data
    data_without_precipitation = sample_data.drop(columns=['precipitation'])
    with pytest.raises(ValueError):
        calculate_climate_index(data_without_precipitation, 'SPI')

    # Test error for missing temperature data
    data_without_temperature = sample_data.drop(columns=['temperature'])
    with pytest.raises(ValueError):
        calculate_climate_index(data_without_temperature, 'GDD')
if __name__ == "__main__":
    pytest.main()