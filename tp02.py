## Exercise 1: Build the `CSVReader` Class (OOP Foundation)

import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Protocol
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class CSVReader:
    

    def __init__(self, file_path):
        self.file_path = file_path
        self._data = None  

    def read(self):
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        self._data = pd.read_csv(self.file_path)
        return self._data

    def preview(self, n=5):
        
        if self._data is None:
            self.read()
        print(f"Top {n} rows of '{self.file_path}':")
        print(self._data.head(n))

    @property
    def data(self):
       
        return self._data

    def __repr__(self):
        status = "loaded" if self._data is not None else "not loaded"
        return f"CSVReader(file_path='{self.file_path}', status={status})"
    
    
# Exercise 2: Strategy Pattern for Missing Values Handling
    
class MissingValueStrategy(ABC):
    """Abstract base class for missing value handling strategies."""
    
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataframe.
        
        Args:
            df: Input dataframe with missing values
            
        Returns:
            DataFrame with missing values handled
        """
        pass
    
    def __str__(self):
        return self.__class__.__name__
    
class DropMissing(MissingValueStrategy):
    """Strategy to drop rows with any missing values."""
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop all rows containing missing values."""
        result = df.dropna()
        print(f"Dropped {len(df) - len(result)} rows with missing values.")
        return result
    
class FillMean(MissingValueStrategy):
    """Strategy to fill missing values with the mean of numeric columns."""
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with column means for numeric columns."""
        result = df.copy()
        
        # Get numeric columns only
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if result[col].isnull().any():
                mean_value = result[col].mean()
                result[col].fillna(mean_value, inplace=True)
                print(f"Filled {col} with mean: {mean_value:.2f}")
        
        # For non-numeric columns, we could drop rows or use a different strategy
        # Here we'll just leave them as is
        return result
    
class FillMode(MissingValueStrategy):
    """Strategy to fill missing values with the mode (most frequent value)."""
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with column mode."""
        result = df.copy()
        
        for col in result.columns:
            if result[col].isnull().any():
                mode_value = result[col].mode()
                if len(mode_value) > 0:
                    result[col].fillna(mode_value[0], inplace=True)
                    print(f"Filled {col} with mode: {mode_value[0]}")
        
        return result
    
class DataCleaner:
    """Context class that applies a missing value handling strategy."""
    
    def __init__(self, strategy: MissingValueStrategy):
        """Initialize with a specific strategy.
        
        Args:
            strategy: A MissingValueStrategy instance
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy: MissingValueStrategy):
        """Change the strategy at runtime.
        
        Args:
            strategy: A new MissingValueStrategy instance
        """
        self._strategy = strategy
        print(f"Strategy changed to: {strategy}")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataframe using the current strategy.
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        print(f"\nApplying strategy: {self._strategy}")
        print("-" * 50)
        return self._strategy.handle(df)
    
## Exercise 3

# 1. Decorator for logging
def log_action(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[START] {func.__name__}")
        result = func(*args, **kwargs)
        print(f"[FINISH] {func.__name__}")
        return result
    return wrapper

# 2. Decorator for execution time
def log_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

        print(f"[TIME] {func.__name__}: {end - start:.6f} seconds")
        return result
    return wrapper

# 3. CSVReader class from Exercise 1
class CSVReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    @log_action
    @log_time
    def read(self):
        self.df = pd.read_csv(self.file_path)
        return self.df

    @log_action
    def preview(self, n=5):
        if self.df is None:
            self.read()
        print(self.df.head(n))
        

## Exercise 4: Implement Factory Pattern for Data Transformations

# 1. Abstract Class
class DataTransform(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
# 2. Subclasses
class NormalizeColumns(DataTransform):
    """Makes column names lowercase and replaces spaces with underscores."""
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        return df
    
class RemoveDuplicates(DataTransform):
    """Removes duplicate rows from the DataFrame."""
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates().reset_index(drop=True)
    
class StandardizeText(DataTransform):
    """Trims extra spaces and converts text values to lowercase."""
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        text_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in text_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()
        return df
    
# 3. Factory Class
class TransformFactory:
    @staticmethod
    def create_transform(transform_type: str) -> DataTransform:
        transforms = {
            "normalize_columns": NormalizeColumns,
            "remove_duplicates": RemoveDuplicates,
            "standardize_text": StandardizeText,
        }
        key = transform_type.strip().lower()
        if key not in transforms:
            raise ValueError(f"Unknown transformation: '{transform_type}'")
        return transforms[key]()
    
### Exercise 5: Combine and implement the reusable code from previous exercises

class DataPipeline(ABC):

    def __init__(self, source: Path, destination: Path) -> None:
        self.source = source
        self.destination = destination
        self.df: pd.DataFrame | None = None

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def clean(self) -> None: ...

    @abstractmethod
    def transform(self) -> None: ...

    @abstractmethod
    def save(self) -> None: ...

    @log_action
    @log_time
    def run(self) -> None:
        LOGGER.info("PIPELINE START: %s to %s", self.source, self.destination)
        self.load()
        self.clean()
        self.transform()
        self.save()
        LOGGER.info("PIPELINE END: result saved in %s", self.destination)
        
class CSVDataPipeline(DataPipeline):

    def __init__(self, source: Path, destination: Path,
                 cleaner: DataCleaner, transforms: list[DataTransform]) -> None:
        super().__init__(source, destination)
        self.reader = CSVReader(source)
        self.cleaner = cleaner
        self.transforms = list(transforms)

    def load(self) -> None:
        self.df = self.reader.read()
        self.reader.preview(3)

    def clean(self) -> None:
        self.df = self.cleaner.clean(self.df)

    def transform(self) -> None:
        for transform in self.transforms:
            LOGGER.info("Transform: %s", type(transform).__name__)
            self.df = transform.apply(self.df)

    def save(self) -> None:
        self.df.to_csv(self.destination, index=False)
        
def build_pipeline(strategy: MissingValueStrategy, transform_names: list[str]) -> CSVDataPipeline:
    return CSVDataPipeline(
        source=INPUT_PATH,
        destination=OUTPUT_PATH,
        cleaner=DataCleaner(strategy),
        transforms=[TransformFactory.create(name) for name in transform_names],
    )