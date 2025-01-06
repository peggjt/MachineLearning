"""
Helper functions for cleaning data.

"""

from sklearn.impute import SimpleImputer


class NumericalVariable:

    def drop_columns(self, X):
        """Drop columns.

        From the dataset, drops columns with incomplete data.
        """
        # Find incomplete column names.
        incomplete_columns = [col for col in X.columns if X_train[col].isnull().any()]

        # Drop incomplete columns.
        X_reduced = X.drop(incoomplete_columns, axis=1)

        return X

    def imputation_basic(self, X):
        """Imputation basic.

        Imputation offers a means to extrapolate missing data.
        However, imputed data is not tracked.
        """
        # Find columns names.
        columns = X.columns

        # Impute data.
        imputer = SimpleImputer()
        X_imputed = pd.DataFrame(imputer.fit_transform(X))

        # After imputation, reinstante columns names.
        X.columns = columns

        return X

    def imputation_track(self, X):
        """Imputation track.

        Imputation offers a means to extrapolate missing data.
        Here, an addtional column tracks imputed data.
        """
        # Find incomplete column names.
        incomplete_columns = [col for col in X.columns if X_train[col].isnull().any()]

        # By adding an addtional column, track data for imputation.
        for i in incomplete_columns:
            X[i + "_was_missing"] = X[i].isnull()

        # Run imputation.
        X = imputation_basic(X)

        return X
