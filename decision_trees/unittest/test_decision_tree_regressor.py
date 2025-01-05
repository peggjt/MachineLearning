import unittest
from decision_trees.decision_tree_regressor import decision_tree_regressor

class TestDecisionTreeRegressor(unittest.TestCase):

    def test_basic(self):

        features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
        model, mae = decision_tree_regressor(
            csv_path="test_data.csv",
            features=features,
            prediction_target="SalePrice",
            max_leaf_nodes=None,
        )

        self.assertEqual(mae, 32410.824657534245)

if __name__ == "__main__":
    unittest.main()

