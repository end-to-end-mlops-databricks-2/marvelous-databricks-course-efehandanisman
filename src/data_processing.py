import pandas as pd
from src.config import ProjectConfig
import datetime
from sklearn.model_selection import train_test_split
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from datetime import datetime
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

class DataProcessor:
    def __init__(self,
                 df,
                 config: ProjectConfig):

        self.df = df
        self.config = config

    def remove_empty_rows(self):
        """
        Drop empty rows from csv
        """
        self.df = self.df.dropna(how="all")

    def select_relevant_columns(self, config: ProjectConfig):
        """
        Drop unnecessary columns
        """
        self.df = self.df.select(self.config.num_features + config.cat_features + [config.target])

    def spark_train_test_split(self,
                               column_name,
                               test_size: float = 0.2,
                               seed: int = 42):
        """
        Split a PySpark DataFrame by a column value into training and testing sets.

        Returns:
        - train_df: PySpark DataFrame, the training set
        - test_df: PySpark DataFrame, the test set
        """
        # Get the distinct values of the column to split by
        unique_values = [row[column_name] for row in self.df.select(column_name).distinct().collect()]

        # Initialize DataFrames for train and test splits
        self.train_df = None
        self.test_df = None

        # Split the data for each value of the specified column
        for value in unique_values:
            # Filter the DataFrame for the current value
            value_df = self.df.filter(F.col(column_name) == value)

            # Perform the split (using random sampling)
            value_train_df, value_test_df = value_df.randomSplit([1 - test_size, test_size], seed=seed)

            # Concatenate the train and test DataFrames
            self.train_df = value_train_df if self.train_df is None else self.train_df.union(value_train_df)
            self.test_df = value_test_df if self.test_df is None else self.test_df.union(value_test_df)

    def save_to_catalog(self, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = self.train_df.withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))

        test_set_with_timestamp = self.test_df.withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set")

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set")

        spark.sql(f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set"
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        spark.sql(f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
