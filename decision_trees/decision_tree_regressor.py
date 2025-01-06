import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def decision_tree_regressor(
    csv_path: str, features: list, prediction_target: str, max_leaf_nodes: int = None
):
    """Decision Tree Regressor.

    Args:
        csv_path (str): The csv file path.
        features (list): The features list.
        prediction_target (str): The prediction target.
        max_leaf_nodes (int): The maximum number of leaf nodes.

    Returns:
        model: The trained model.
        mae (float): The models mean absolute error.
    """
    # Read CSV data.
    data = pd.read_csv(csv_path)

    # Find data.
    X = data[features]
    y = data[prediction_target]
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)

    # Form model.
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)

    # Validate model.
    prediction = model.predict(test_X)
    mae = mean_absolute_error(prediction, test_y)

    return model, mae
