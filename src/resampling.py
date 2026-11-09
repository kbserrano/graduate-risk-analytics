"""
graduate-risk-analytics
src/resampling.py

ValidCategoryADASYN: ADASYN whose synthetic rows are snapped back to
valid values for one-hot and 0/1 columns.

ADASYN (unlike SMOTE, which has SMOTE-NC) has no categorical-aware
variant: it interpolates every column, so a synthetic graduate can come
out as e.g. 0.4 BEED / 0.6 BSEd Math, or 0.3 "working student". This
subclass runs plain ADASYN, then, for the synthetic rows only:

    - in each one-hot group (columns starting with a given prefix),
      sets the largest value to 1 and the rest to 0 — the category the
      synthetic point is closest to;
    - rounds each 0/1 column to 0 or 1.

Original rows and continuous columns are left untouched. Requires
imbalanced-learn; import this module only where it's needed.
"""

import numpy as np
import pandas as pd

from imblearn.over_sampling import ADASYN


class ValidCategoryADASYN(ADASYN):
    """
    ADASYN + snapping of synthetic rows to valid categories.

    onehot_prefixes: column-name prefixes identifying one-hot groups
        (e.g. ("program_",)).
    binary_columns: names of 0/1 columns.

    X must be a pandas DataFrame with those column names (e.g. a
    ColumnTransformer with set_output(transform="pandas")).
    """

    _parameter_constraints = {
        **ADASYN._parameter_constraints,
        "onehot_prefixes": [tuple, list],
        "binary_columns": [tuple, list],
    }

    def __init__(self, *, sampling_strategy="auto", random_state=None, n_neighbors=5,
                 onehot_prefixes=(), binary_columns=()):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            n_neighbors=n_neighbors,
        )
        self.onehot_prefixes = onehot_prefixes
        self.binary_columns = binary_columns

    def fit_resample(self, X, y, **params):

        if not isinstance(X, pd.DataFrame):
            raise TypeError("ValidCategoryADASYN needs a DataFrame with column names.")

        n_original = len(X)
        X_res, y_res = super().fit_resample(X, y, **params)

        # imbalanced-learn appends synthetic rows after the originals
        X_res = X_res.copy()
        synthetic = X_res.iloc[n_original:]

        for prefix in self.onehot_prefixes:
            cols = [c for c in X_res.columns if str(c).startswith(prefix)]
            if not cols:
                continue
            values = synthetic[cols].to_numpy()
            snapped = np.zeros_like(values)
            snapped[np.arange(len(values)), values.argmax(axis=1)] = 1
            X_res.iloc[n_original:, X_res.columns.get_indexer(cols)] = snapped

        for col in self.binary_columns:
            idx = X_res.columns.get_loc(col)
            X_res.iloc[n_original:, idx] = np.clip(np.round(synthetic[col].to_numpy()), 0, 1)

        return X_res, y_res
