import dataclasses as dtc

import numpy as np
import pandas as pd

from promisces.models.substance import Substance


@dtc.dataclass
class Matrix:
    id: str
    name: str
    description: str

    def __repr__(self):
        return f"InputMatrix({self.id}, {self.name})"

    def substance_concentration_range_from_lit(self, substance: Substance) -> tuple[float, float]:
        df = pd.read_csv(os.path.join("data", "starting_concentration.csv"),
                         encoding='cp1252',
                         sep=';',
                         na_values="",
                         keep_default_na=False,
                         dtype={"removal_percent": float}
                         )
        dff = df.loc[
            (df.substance_id == substance.id) &
            (df.matrix_id == self.id)
            ]
        lit_values = np.concatenate((
            dff.min_value_ng_l.values,
            dff.point_value_ng_l.values,
            dff.max_value_ng_l.values)
        )

        lit_values = lit_values[~np.isnan(lit_values)]
        if not np.any(lit_values):
            raise RuntimeError(f"No concentration found for substance {substance.id} in matrix {self.id}")
        return lit_values.min(), lit_values.max()


class Matrices:
    iww = Matrix("iww", "Industrial wastewater", "Raw wastewater from industrial sources")
    hww = Matrix("hww", "Household wastewater", "Raw wastewater from households")
    lww = Matrix("lww", "Landfill leachate", "Landfill leachate")
    rww = Matrix("rww", "Raw wastewater", "Raw wastewater from various sources (dominated by houshold wastewater)")
    tiw = Matrix("tiw", "Treated industrial wastewater",
                 "Industrial or landfill leachate after wastewater treatment (not further specified, only used for site specific data)")
    tww = Matrix("tww", "Treated wastewater", "Wastewater after primary and secondary treatment")
    stw = Matrix("stw", "Stormwater runoff", "Urban stormwater runoff")
    raw = Matrix("raw", "Rainwater", "Rainwater without runoff")
    suw = Matrix("suw", "Surface water", "Surfacewater")
    grw = Matrix("grw", "Groundwater", "Groundwater")
    pow = Matrix("pow", "Soil pore water", "Soil leachate")
    bfw = Matrix("bfw", "Bank filtrate", "Bank filtrate")
    drw = Matrix("drw", "Drinking water", "Drinking water after treatment")
    sdg = Matrix("sdg", "Sludge",
                 "Thickened sludge from treated wastewater (as part of the secondary wastewater treatment)")
    no_change = None
