import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    """Handles preprocessing steps for battery data, especially from transformer scripts."""

    def __init__(self, voltage_col, current_col, timestamp_col, target_col,
                 window_size_hours=5, sampling_rate_hz=1 / 60, downsample_freq='15T',
                 filter_type='butterworth', butter_order=4, butter_cutoff=0.01,
                 savgol_window=51, savgol_polyorder=3,
                 remove_negatives=True, normalize=True):
        self.voltage_col = voltage_col
        self.current_col = current_col
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.window_size_hours = window_size_hours
        self.sampling_rate_hz = sampling_rate_hz
        self.downsample_freq = downsample_freq
        self.filter_type = filter_type
        self.butter_order = butter_order
        self.butter_cutoff = butter_cutoff  # Adjusted cutoff based on typical low-frequency data
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        self.remove_negatives = remove_negatives
        self.normalize = normalize
        self.scaler = MinMaxScaler() if normalize else None
        # Include other necessary columns if used as features directly or for calculations
        self.feature_cols = ['max_cell_voltage (V)', 'min_cell_voltage (V)',
                             'max_temperature (C)', 'min_temperature (C)',
                             'available_energy (Wh)']  # Example, adjust based on actual data/needs

    def _load_and_clean(self, file_path):
        """Loads data, handles timestamps, drops duplicates, sorts."""
        try:
            data = pd.read_csv(file_path)
            # Check essential columns exist
            required_cols = [self.voltage_col, self.current_col, self.timestamp_col,
                             self.target_col] + self.feature_cols
            if not all(col in data.columns for col in required_cols):
                missing = [col for col in required_cols if col not in data.columns]
                raise ValueError(f"Missing required columns: {missing}")

            data[self.timestamp_col] = pd.to_datetime(data[self.timestamp_col])
            data = data.drop_duplicates(subset=[self.timestamp_col])
            data = data.sort_values(by=self.timestamp_col)
            data = data.set_index(self.timestamp_col)
            return data
        except Exception as e:
            print(f"Error loading/cleaning {file_path}: {e}")
            return pd.DataFrame()  # Return empty dataframe on error

    def _apply_filter(self, data_series):
        """Applies the selected filter type."""
        if self.filter_type == 'butterworth':
            nyquist = 0.5 * self.sampling_rate_hz
            # Ensure cutoff is less than Nyquist frequency
            cutoff_freq = min(self.butter_cutoff, nyquist * 0.99)
            if cutoff_freq <= 0:
                print(
                    f"Warning: Butterworth cutoff frequency ({self.butter_cutoff}) is too low or invalid for Nyquist ({nyquist}). Skipping filter.")
                return data_series
            b, a = butter(self.butter_order, cutoff_freq, btype='low', analog=False, fs=self.sampling_rate_hz)
            return filtfilt(b, a, data_series)
        elif self.filter_type == 'savgol':
            # Ensure window length is odd and less than data length
            window = min(self.savgol_window, len(data_series) - 1)
            if window % 2 == 0: window -= 1
            if window < self.savgol_polyorder + 1:
                print(
                    f"Warning: Savgol filter window ({window}) too small for polyorder ({self.savgol_polyorder}). Skipping filter.")
                return data_series
            return savgol_filter(data_series, window, self.savgol_polyorder)
        elif self.filter_type == 'moving_average':
            window_samples = int(self.window_size_hours * 3600 * self.sampling_rate_hz)  # Convert hours to samples
            window_samples = max(1, window_samples)  # Ensure window size is at least 1
            return data_series.rolling(window=window_samples, min_periods=1).mean()
        else:
            return data_series  # No filter or unknown type

    def _remove_negatives(self, data):
        """Removes negative values from voltage and current."""
        for col in [self.voltage_col, self.current_col]:
            if col in data.columns:
                data[col] = data[col].apply(lambda x: max(x, 0) if pd.notna(x) else x)
        return data

    def _downsample(self, data):
        """Downsamples the data."""
        if self.downsample_freq:
            # Handle potential non-numeric columns before resampling mean
            numeric_cols = data.select_dtypes(include=np.number).columns
            return data[numeric_cols].resample(self.downsample_freq).mean()
        return data

    def _normalize_data(self, data):
        """Applies Min-Max normalization."""
        if self.normalize and self.scaler:
            # Fit scaler only once or on training data ideally
            # For simplicity here, fitting per file - might cause issues
            # Select only numeric columns for scaling
            numeric_cols = data.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
                return data, self.scaler
            else:
                print("Warning: No numeric columns found for normalization.")
                return data, None
        return data, None

    def process_file(self, file_path):
        """Processes a single data file."""
        data = self._load_and_clean(file_path)
        if data.empty:
            return pd.DataFrame(), None

        # Apply preprocessing steps
        for col in [self.voltage_col, self.current_col] + self.feature_cols + [self.target_col]:
            if col in data.columns:
                # Handle NaNs before filtering/processing
                data[col] = data[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
                if data[col].isnull().any():  # If still NaNs, fill with 0
                    data[col] = data[col].fillna(0)
                data[col] = self._apply_filter(data[col])

        if self.remove_negatives:
            data = self._remove_negatives(data)

        data = self._downsample(data)
        data = data.dropna()  # Drop rows with NaNs after downsampling/processing

        if data.empty:
            print(f"Warning: Dataframe empty after preprocessing steps for {file_path}")
            return pd.DataFrame(), None

        data_normalized, scaler_used = self._normalize_data(data)

        # Reset index to make timestamp a column again if needed downstream
        data_normalized = data_normalized.reset_index()

        return data_normalized, scaler_used


# --- Functions from original preprocess.py (can be integrated or kept separate) ---
# These seem more like general-purpose functions than part of the class above.

def remove_outliers(data):
    # Placeholder - implement robust outlier detection (e.g., IQR, Z-score)
    print("Warning: Outlier removal not implemented.")
    return data


def set_voltage_limits(data, voltage_col='voltage', initial_hours=50):
    """Sets voltage limits based on initial data (if applicable)."""
    # This logic might be specific to a particular dataset/problem
    # Ensure 'timestamp' is available or adjust logic
    try:
        if pd.api.types.is_datetime64_any_dtype(data.index):
            initial_cutoff = data.index[0] + pd.Timedelta(hours=initial_hours)
            initial_data = data[data.index <= initial_cutoff]
            if not initial_data.empty and voltage_col in initial_data.columns:
                Umax = initial_data[voltage_col].max()
                Umin = 0.9 * Umax
                return Umax, Umin
    except Exception as e:
        print(f"Warning: Could not set voltage limits: {e}")
    return None, None  # Return None if limits cannot be determined

