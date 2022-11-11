import pytest
import logging
from churn_library import *


# Config python logging module - But we will use pytest live logging instead
logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

######################### FIXTURES ##################################

"""
Fixture - The test function test_import() will 
use the return of path() as an argument
"""
@pytest.fixture(scope="module")
def path():
    return "./data/bank_data.csv"


"""
Fixture - The test functions will 
use the return of dataframe() as an argument
"""
@pytest.fixture(scope="module")
def dataframe(path):
    return import_data(path)


"""
Fixture - The test function test_encoder_helper will 
use the parameters returned by encoder_params() as arguments
"""
@pytest.fixture(scope="module", params=[['Gender','Education_Level','Marital_Status','Income_Category','Card_Category'],
										['Gender','Education_Level','Marital_Status','Income_Category'],
										['Gender','Education_Level','Marital_Status','Income_Category','Card_Category','Not_a_column'],
										])
def encoder_params(request):
  cat_features = request.param
  # get dataset from pytest Namespace
  df = pytest.df.copy()
  return df, cat_features


"""
Fixture - The test function test_train_models will 
use the datasets returned by input_train() as arguments
"""
@pytest.fixture(scope="module")
def input_train():
	# get dataset from pytest Namespace
	df = pytest.df.copy()
	return perform_feature_engineering(df)

######################### UNIT TESTS ##################################

@pytest.mark.parametrize("filename",["./data/bank_data.csv", "./data/no_file.csv"])
def test_import(filename):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	two_test_level = False

	try:
		df = import_data(filename)
		logging.info("Testing import_data from file: %s - SUCCESS", filename)
		
		# store dataframe into pytest namespace for re-use in other test functions
		pytest.df = df
		two_test_level = True

	except FileNotFoundError as err:
		logging.error("Testing import_data from file: %s: The file wasn't found", filename)
		
	if two_test_level:
		try:
			assert df.shape[0] > 0
			assert df.shape[1] > 0
			logging.info("Returned dataframe with shape: %s", df.shape)

		except AssertionError as err:
			logging.error("The file doesn't appear to have rows and columns")
	

def test_eda():
	'''
	test perform eda function
	'''
	# get dataset from pytest Namespace
	df = pytest.df
	
	try:
		perform_eda(df)
		logging.info("Testing perform_eda - SUCCESS")

	except Exception as err:
		logging.error("Testing perform_eda failed - Error type %s", type(err))


def test_encoder_helper(encoder_params):
	'''
	test encoder helper
	'''
	two_test_level = False
	df, cat_features = encoder_params
	
	try:
		newdf = encoder_helper(df, cat_features)
		logging.info("Testing encoder_helper with %s - SUCCESS", cat_features)
		two_test_level = True

	except KeyError:
		logging.error("Testing encoder_helper with %s failed: Check for categorical features not in the dataset", cat_features)

	except Exception as err:
		logging.error("Testing encoder_helper failed - Error type %s", type(err))


	if two_test_level:
		
		try:
			assert newdf.select_dtypes(include='object').columns.tolist() == []
			logging.info("All categorical columns were encoded")

		except AssertionError as err:
			logging.error("At least one categorical columns was NOT encoded - Check categorical features submitted")



def test_perform_feature_engineering():
	'''
	test perform_feature_engineering
	'''

	two_test_level = False
	try:
		# get dataset from pytest Namespace
		df = pytest.df
		X_train, X_test, y_train, y_test = perform_feature_engineering(df)
		logging.info("Testing perform_feature_engineering - SUCCESS")
		two_test_level = True

	except Exception as err:
		logging.error("Testing perform_feature_engineering failed - Error type %s", type(err))

	if two_test_level:
		try:
			assert X_train.shape[0] > 0
			assert X_train.shape[1] > 0
			assert X_test.shape[0] > 0
			assert X_test.shape[1] > 0
			assert y_train.shape[0] > 0
			assert y_test.shape[0] > 0
			logging.info("perform_feature_engineering returned Train / Test set of shape %s %s", X_train.shape,X_test.shape)

		except AssertionError as err:
			logging.error("The returned train / test datasets do not appear to have rows and columns")


def test_train_models(input_train):
	'''
	test train_models
	'''
	try:
		train_models(*input_train)
		logging.info("Testing train_models: SUCCESS")
	except Exception as err:
		logging.error("Testing train_models failed - Error type %s", type(err))



if __name__ == "__main__":
	'''test_import("./data/bank_data.csv")
	test_import("./data/no_file.csv")
	test_perform_feature_engineering(import_data("./data/bank_data.csv"))
	test_encoder_helper(import_data("./data/bank_data.csv"), 
		['Gender','Education_Level','Marital_Status','Income_Category'])
	test_encoder_helper(import_data("./data/bank_data.csv"), 
		['Gender','Education_Level','Marital_Status','Income_Category','Card_Category'])'''
	pass








