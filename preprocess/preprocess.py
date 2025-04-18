import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataPreprocessor:
    """
    Handles preprocessing of battery data including filtering, cleaning, downsampling, and normalization.
    """

    def __init__(self,
                 voltage_col='voltage',
                 current_col='current',
                 timestamp_col='timestamp',
                 window_size_hours=5,
                 sampling_rate_hz=1,  # Assuming initial data is sampled roughly every second
                 downsample_freq='15T',
                 filter_type=None,  # 'butterworth', 'savgol', or None
                 butterworth_order=4,
                 butterworth_cutoff_frac=0.1,
                 # Cutoff frequency as a fraction of the Nyquist frequency after downsampling
                 savgol_polyorder=2,
                 remove_negatives=True,
                 normalize=True):
        """
        Initializes the DataPreprocessor.

        Args:
            voltage_col (str): Name of the voltage column.
            current_col (str): Name of the current column.
            timestamp_col (str): Name of the timestamp column.
            window_size_hours (int): Window size in hours for moving average.
            sampling_rate_hz (int): Approximate original sampling rate in Hz.
            downsample_freq (str): Pandas frequency string for downsampling (e.g., '15T').
            filter_type (str, optional): Type of filter to apply ('butterworth', 'savgol'). Defaults to None.
            butterworth_order (int): Order for the Butterworth filter.
            butterworth_cutoff_frac (float): Cutoff frequency fraction for Butterworth filter.
            savgol_polyorder (int): Polynomial order for Savitzky-Golay filter.
            remove_negatives (bool): Whether to clip negative voltage/current values at 0.
            normalize (bool): Whether to apply Min-Max scaling.
        """
        self.voltage_col = voltage_col
        self.current_col = current_col
        self.timestamp_col = timestamp_col
        # Calculate window size in samples based on approximate original sampling rate
        self.window_size_samples = int(window_size_hours * 3600 * sampling_rate_hz)
        self.downsample_freq = downsample_freq
        self.filter_type = filter_type
        self.butterworth_order = butterworth_order
        self.butterworth_cutoff_frac = butterworth_cutoff_frac
        self.savgol_polyorder = savgol_polyorder
        self.remove_negatives = remove_negatives
        self.normalize = normalize
        self.scaler = MinMaxScaler() if normalize else None
        self.downsample_fs_hz = 1 / (pd.Timedelta(downsample_freq).total_seconds())  # Sampling freq after downsampling

        if self.filter_type not in [None, 'butterworth', 'savgol']:
            raise ValueError("filter_type must be 'butterworth', 'savgol', or None")

        logging.info(
            f"Preprocessor initialized with window_size_samples={self.window_size_samples}, downsample_freq={self.downsample_freq}")

    def _apply_moving_average(self, series):
        """Applies moving average filter."""
        return series.rolling(window=self.window_size_samples, min_periods=1).mean()

    def _apply_butterworth_filter(self, series):
        """Applies Butterworth low-pass filter."""
        nyquist = 0.5 * self.downsample_fs_hz
        cutoff = self.butterworth_cutoff_frac * nyquist
        if cutoff <= 0 or cutoff >= nyquist:
            logging.warning(
                f"Butterworth cutoff frequency ({cutoff} Hz) is outside the valid range (0, {nyquist} Hz). Skipping filter.")
            return series
        b, a = butter(self.butterworth_order, cutoff, btype='low', analog=False, fs=self.downsample_fs_hz)
        # Apply filter only if series is long enough
        if len(series) > len(b) * 3:  # Rule of thumb: signal length > 3 * filter order
            return filtfilt(b, a, series)
        else:
            logging.warning(
                f"Series too short ({len(series)} points) for Butterworth filter order {len(b)}. Skipping filter.")
            return series

    def _apply_savgol_filter(self, series):
        """Applies Savitzky-Golay filter."""
        window_length = self.window_size_samples // (3600 * sampling_rate_hz * pd.Timedelta(
            self.downsample_freq).total_seconds())  # Adjust window length based on downsampling factor
        window_length = max(5, int(window_length) | 1)  # Ensure window length is odd and at least 5
        if window_length >= len(series):
            logging.warning(
                f"Window length ({window_length}) >= series length ({len(series)}). Skipping SavGol filter.")
            return series
        if self.savgol_polyorder >= window_length:
            logging.warning(
                f"SavGol polyorder ({self.savgol_polyorder}) >= window length ({window_length}). Skipping filter.")
            return series
        return savgol_filter(series, window_length=window_length, polyorder=self.savgol_polyorder)

    def process_dataframe(self, df):
        """
        Applies the full preprocessing pipeline to a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with timestamp, voltage, and current columns.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
            MinMaxScaler: The fitted scaler object (or None if normalize=False).
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        processed_df = df.copy()

        # Ensure timestamp is datetime type
        if not pd.api.types.is_datetime64_any_dtype(processed_df[self.timestamp_col]):
            try:
                processed_df[self.timestamp_col] = pd.to_datetime(processed_df[self.timestamp_col])
            except Exception as e:
                logging.error(f"Failed to convert timestamp column '{self.timestamp_col}' to datetime: {e}")
                raise

        # 1. Remove negative values
        if self.remove_negatives:
            logging.info("Clipping negative values for voltage and current.")
            if self.voltage_col in processed_df.columns:
                processed_df[self.voltage_col] = processed_df[self.voltage_col].clip(lower=0)
            if self.current_col in processed_df.columns:
                processed_df[self.current_col] = processed_df[self.current_col].clip(lower=0)

        # 2. Remove duplicates and sort by timestamp
        initial_rows = len(processed_df)
        processed_df = processed_df.drop_duplicates().sort_values(self.timestamp_col)
        rows_removed = initial_rows - len(processed_df)
        if rows_removed > 0:
            logging.info(f"Removed {rows_removed} duplicate rows.")

        # Check for out-of-order timestamps
        time_diffs = processed_df[self.timestamp_col].diff().dt.total_seconds()
        out_of_order = time_diffs[time_diffs < 0]
        if not out_of_order.empty:
            logging.warning(f"Found {len(out_of_order)} out-of-order timestamps after sorting. Check data integrity.")
            # Optional: remove out-of-order points? For now, we just warn.

        # 3. Apply moving average (before downsampling)
        logging.info("Applying moving average...")
        if self.voltage_col in processed_df.columns:
            processed_df[self.voltage_col] = self._apply_moving_average(processed_df[self.voltage_col])
        if self.current_col in processed_df.columns:
            processed_df[self.current_col] = self._apply_moving_average(processed_df[self.current_col])

        # Set timestamp as index for resampling
        processed_df = processed_df.set_index(self.timestamp_col)

        # 4. Downsample data
        logging.info(f"Downsampling to {self.downsample_freq}...")
        processed_df = processed_df.resample(
            self.downsample_freq).mean().dropna()  # Use mean for aggregation, drop rows where all values are NaN after resampling
        initial_rows_after_downsample = len(processed_df)
        if initial_rows_after_downsample == 0:
            logging.warning("DataFrame is empty after downsampling and dropping NaNs.")
            return processed_df, self.scaler  # Return empty df

        # 5. Apply additional filtering (after downsampling)
        if self.filter_type:
            logging.info(f"Applying {self.filter_type} filter...")
            if self.voltage_col in processed_df.columns:
                if self.filter_type == 'butterworth':
                    processed_df[self.voltage_col] = self._apply_butterworth_filter(processed_df[self.voltage_col])
                elif self.filter_type == 'savgol':
                    processed_df[self.voltage_col] = self._apply_savgol_filter(processed_df[self.voltage_col])
            if self.current_col in processed_df.columns:
                if self.filter_type == 'butterworth':
                    processed_df[self.current_col] = self._apply_butterworth_filter(processed_df[self.current_col])
                elif self.filter_type == 'savgol':
                    processed_df[self.current_col] = self._apply_savgol_filter(processed_df[self.current_col])

        # 6. Normalize data
        feature_cols = [col for col in processed_df.columns if
                        col != self.timestamp_col]  # Normalize all except timestamp
        if self.normalize and feature_cols:
            logging.info("Applying Min-Max normalization...")
            processed_df[feature_cols] = self.scaler.fit_transform(processed_df[feature_cols])

        logging.info(f"Preprocessing complete. Final shape: {processed_df.shape}")
        return processed_df.reset_index(), self.scaler  # Reset index to bring timestamp back as column

    def process_file(self, file_path):
        """
        Loads a CSV file and applies the preprocessing pipeline.

        Args:
            file_path (str): Path to the input CSV file.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
             MinMaxScaler: The fitted scaler object (or None if normalize=False).
        """
        try:
            logging.info(f"Processing file: {file_path}")
            df = pd.read_csv(file_path)
            if self.timestamp_col not in df.columns:
                raise ValueError(f"Timestamp column '{self.timestamp_col}' not found in {file_path}")
            return self.process_dataframe(df)
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            raise


# Example usage (optional, can be run if script is executed directly)
if __name__ == '__main__':
    # Create dummy data for demonstration
    np.random.seed(0)
    dates = pd.date_range('2023-01-01', periods=10000, freq='H')
    data = pd.DataFrame({
        'timestamp': dates,
        'voltage': 4.2 - np.cumsum(np.random.randn(10000) * 0.001) + np.random.randn(10000) * 0.05,
        'current': 5 + np.random.randn(10000) * 0.1,
        'temperature': 25 + np.cumsum(np.random.randn(10000) * 0.0001)
    })
    # Introduce some negatives and duplicates
    data.loc[::100, 'voltage'] *= -1
    data.loc[::200, 'current'] *= -1
    data = pd.concat([data, data.iloc[::50]], ignore_index=True)

    dummy_file = 'dummy_battery_data.csv'
    data.to_csv(dummy_file, index=False)

    logging.info("--- Running Preprocessing Example ---")

    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        voltage_col='voltage',
        current_col='current',
        timestamp_col='timestamp',
        window_size_hours=5,
        sampling_rate_hz=1 / 3600,  # Original data is hourly
        downsample_freq='4H',  # Downsample to 4 hours
        filter_type='butterworth',
        butterworth_cutoff_frac=0.2,
        remove_negatives=True,
        normalize=True
    )

    try:
        processed_df, fitted_scaler = preprocessor.process_file(dummy_file)
        print("\nProcessed DataFrame head:")
        print(processed_df.head())
        if fitted_scaler:
            print(f"\nScaler min: {fitted_scaler.min_}, data min: {fitted_scaler.data_min_}")
            print(f"Scaler scale: {fitted_scaler.scale_}, data range: {fitted_scaler.data_range_}")

        # Example of inverse transform (if needed later)
        # original_data = fitted_scaler.inverse_transform(processed_df[['voltage', 'current', 'temperature']])
        # print("\nInverse transformed head:")
        # print(original_data[:5])

    except Exception as e:
        print(f"An error occurred during example processing: {e}")
    finally:
        # Clean up dummy file
        import os

        if os.path.exists(dummy_file):
            os.remove(dummy_file)
            logging.info(f"Removed dummy file: {dummy_file}")

# End of script