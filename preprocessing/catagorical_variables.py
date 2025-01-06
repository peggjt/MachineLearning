import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


class CatagoricalVariables:

    def find_catagorical_variables(self, X):
        """Find Catagorical Variables."""
        object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

        print("Catagorical Varbials:")
        print(object_cols)

        return object_cols

    def drop(self, X):

        X = X.select_dtypes(exclude=["object"], axis=1)

        return X

    def ordinal_encoder(self, X):
        """Ordinal Encoder.

        If the categories do not have an inherent order, the model may mistakenly
        interpret the numeric values as having a meaningful relationship.
        """
        object_cols = self.find_catagorical_variables(X)

        ordinal_encoder = OrdinalEncoder()
        X = ordinal_encoder.fit_transform(X[object_cols])

        return X

    def one_hot_encoding(self):
        """One-Hot Encoder.

        One-hot encoding creates new columns indicating the presence
        (or absence) of each possible value in the original data.
        """
        index = X.index
        object_cols = self.find_catagorical_variables(X)

        # Apply one-hot encoder to each column with categorical data.
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        X_one_hot = pd.DataFrame(one_hot_encoder.fit_transform(X[object_cols]))

        # Reinstate index.
        X_one_hot.index = index

        # Remove categorical columns.
        # Then add one-hot encoded columns to numerical features.
        X_numerical = self.drop(X)
        X = pd.concat([X_numerical, X_one_hot], axis=1)

        # Ensure all columns have string type
        X.columns = OH_X_train.columns.astype(str)

        return X
