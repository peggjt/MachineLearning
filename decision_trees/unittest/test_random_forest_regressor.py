import unittest
from decision_trees.random_forest_regressor import random_forest_regressor


class TestRandomForestRegressor(unittest.TestCase):

    def test_basic(self):

        features = [
            "LotArea",
            "YearBuilt",
            "1stFlrSF",
            "2ndFlrSF",
            "FullBath",
            "BedroomAbvGr",
            "TotRmsAbvGrd",
        ]
        model, mae = random_forest_regressor(
            csv_path="test_data.csv",
            features=features,
            prediction_target="SalePrice",
            max_leaf_nodes=None,
        )

        self.assertEqual(mae, 23093.063676581867)


if __name__ == "__main__":
    unittest.main()
