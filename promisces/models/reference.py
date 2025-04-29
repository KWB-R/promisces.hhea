import dataclasses as dtc
import os
import warnings

import pandas as pd

from promisces.models.matrix import Matrix, Matrices
from promisces.models.substance import Substance, Substances


@dtc.dataclass
class Reference:
    id: str
    substance: Substance
    matrix: Matrix
    ref_value_ng_l: float
    year: int
    comments: str

    @staticmethod
    def from_lit(output_matrix: Matrix, substance: Substance) -> "Reference":
        df = pd.read_csv(
            os.path.join("data", "reference_lit.csv"),
            encoding='cp1252',
            sep=';',
            dtype={
                "substance_id": str,
                "matrix_id": str,
                "reference_value": float,
                "reference_id": str,
                "year": pd.Int64Dtype(),
                "comments": str}
        )
        df = df[df.substance_id.eq(substance.id) & df.matrix_id.eq(output_matrix.id)]
        if len(df):
            if len(df) < 1:
                warnings.warn("more than one literature reference found. returning only the first one")
            df = next(df.itertuples(index=False))
            return Reference(
                df.reference_id,
                getattr(Substances, df.substance_id),
                getattr(Matrices, df.matrix_id),
                df.reference_value_ng_l,
                df.year, df.comments
            )
        # TODO: fake ref?
        return Reference("dummy", substance, output_matrix, 1, 2024, "Not a true reference")