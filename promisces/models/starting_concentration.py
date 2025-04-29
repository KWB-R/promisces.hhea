import os
import numpy as np
import pandas as pd
import dataclasses as dtc

from promisces.models.array_container import ArrayContainer
from promisces.models.matrix import Matrix
from promisces.models.substance import Substance


@dtc.dataclass
class StartingConcentration(ArrayContainer):
    arr: np.ndarray
    substance: Substance
    matrix: Matrix

    attrs = ["substance", "matrix"]

    @staticmethod
    def from_lit(substance: Substance, matrix: Matrix) -> "StartingConcentration":
        df = pd.read_csv(os.path.join("data", "starting_concentration.csv"),
                         encoding='cp1252',
                         sep=';',
                         na_values="",
                         keep_default_na=False,
                         dtype={"removal_percent": float}
                         )
        dff = df.loc[
            (df.substance_id == substance.id) &
            (df.matrix_id == matrix.id)
            ]
        lit_values = np.concatenate((
            dff.min_value_ng_l.values,
            dff.point_value_ng_l.values,
            dff.max_value_ng_l.values)
        )
        return StartingConcentration(lit_values[~np.isnan(lit_values)], substance, matrix)

    def n_uniform_samples(self, n_samples: int):
        return np.sort(np.random.uniform(self.arr.min(), self.arr.max(), n_samples))[::-1]
