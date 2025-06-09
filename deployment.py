# Determine Databricks resources to specify for automatic auth passthrough at deployment time
import mlflow
from agent import RESTAURANTS_GENIE_SPACE_ID, LLM_ENDPOINT_NAME, tools
from databricks_langchain import UnityCatalogTool, VectorSearchRetrieverTool
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
)
from pkg_resources import get_distribution

# TODO: Manually include underlying resources if needed. See the TODO in the markdown above for more information.
resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
    DatabricksGenieSpace(genie_space_id=RESTAURANTS_GENIE_SPACE_ID)
]
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent.py",
        input_example=input_example,
        extra_pip_requirements=[f"databricks-connect=={get_distribution('databricks-connect').version}"],
        resources=resources,
    )