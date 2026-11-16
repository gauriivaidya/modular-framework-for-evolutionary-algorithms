import scipy.io as scio
import os


class DataLoader:
    """
    A class for loading data files with built-in caching.

    This class loads data from the 'data' directory and caches the results
    """
    _instance = None
    _cache = {}

    def __new__(cls):
        # Singleton pattern to ensure only one instance of the loader exists
        if cls._instance is None:
            cls._instance = super(DataLoader, cls).__new__(cls)
        return cls._instance

    def load_mat_data(self, filename):
        """
        Loads data from a .mat file, using a cache to prevent re-loading.

        Args:
            filename (str): The name of the .mat file to load.

        Returns:
            dict: The dictionary of data loaded from the .mat file.

        Raises:
            FileNotFoundError: If the file is not found in the 'data' directory.
        """
        if filename in self._cache:
            return self._cache[filename]

        data_dir = "data"
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}. "
                                    f"Please ensure all .mat files are in the '{data_dir}' directory.")

        data = scio.loadmat(file_path)
        self._cache[filename] = data
        return data


data_loader = DataLoader()
