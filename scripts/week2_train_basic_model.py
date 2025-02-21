import mlflow
from pyspark.sql import SparkSession

from src.airline_reviews.models.basic_model import BasicModel
from src.config import ProjectConfig, Tags

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "4ce0950880b6fdade547501027c83efd6bc5ed86", "branch": "efehan_week2"}
tags = Tags(**tags_dict)

# Initialize model
basic_model = BasicModel(config=config, tags=tags, spark=spark)
# Create feature table

basic_model.load_data()
basic_model.prepare_features()
basic_model.train()
basic_model.log_model()


basic_model.register_model()
