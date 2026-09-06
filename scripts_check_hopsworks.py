from src.config import SETTINGS
from src.hopsworks_io import get_project

project = get_project()
fs = project.get_feature_store()
project.get_model_registry()
print("Hopsworks login PASSED")
print("Project:", project.name)
print("Feature Store:", fs.name)
print("Raw Feature Group will be:", SETTINGS.raw_feature_group_name)
print("Engineered Feature Group will be:", SETTINGS.feature_group_name)
print("Feature View will be:", SETTINGS.feature_view_name)
print("Model Registry model will be:", SETTINGS.model_name)
