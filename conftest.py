import pytest

def df_plugin():
  return None

# Creating a Dataframe object 'pytest.df' in Namespace
def pytest_configure():
  pytest.df = df_plugin()
