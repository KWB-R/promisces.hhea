import dataclasses as dtc
import warnings

import numpy as np

from promisces.models.mixture import Mixture
from promisces.models.reference import Reference
from promisces.models.starting_concentration import StartingConcentration
from promisces.models.removal_percent import RemovalPercent
from promisces.models.matrix import Matrix, Matrices
from promisces.models.substance import Substance, Substances
from promisces.models.treatment import Treatments, TreatmentTrain


@dtc.dataclass
class CaseStudy:
    name: str
    mixtures: list[Mixture]
    removal_percents: list[RemovalPercent]
    references: list[Reference]
    starting_concentration: list[StartingConcentration]

    def get_removal_percents_for(self, treatment_train: TreatmentTrain, substance: Substance) -> list[RemovalPercent]:
        by_substance = tuple(r for r in self.removal_percents if substance.id == r.substance.id)
        out = []
        for treatment in treatment_train:
            matches = tuple(r for r in by_substance if r.treatment.id == treatment.id)
            if len(matches) > 0:
                out += [matches[0]]
            else:
                out += [RemovalPercent(np.array([]))]
        return out

    def get_mixtures_for(self, treatment_train: TreatmentTrain, substance: Substance, scenario: str) -> list[Mixture]:
        by_substance = tuple(m for m in self.mixtures if substance.id == m.substance.id and scenario == m.scenario)
        out = []
        for treatment in treatment_train:
            matches = tuple(m for m in by_substance if m.treatment.id == treatment.id)
            if len(matches) > 0:
                out += [matches[0]]
            else:
                out += [None]
        return out

    def get_starting_concentration_for(self, substance: Substance, matrix: Matrix) -> StartingConcentration | None:
        matches = tuple(s for s in self.starting_concentration
                        if s.substance.id == substance.id and s.matrix.id == matrix.id and len(s.arr) > 0)
        if len(matches) == 0:
            return None
        if len(matches) > 1:
            warnings.warn("More than one starting concentration matched! Returning only the first one")
        return matches[0]


groundwater_remediation = CaseStudy(
    "Groundwater Remediation",
    mixtures=[
        Mixture(0, 0, 0, 0, Treatments.dilgw, Substances.pfoa, "persulfate"),
        Mixture(0, 0, 0, 0, Treatments.dilgw, Substances.pfoa, "ultra_cavitation")
    ],
    removal_percents=[],
    references=[
        Reference("TrinkwV 2023 + DOI 10.21945/RIVM-2018-0070",
                  Substances.pfoa, Matrices.drw,
                  ref_value_ng_l=5.0,
                  year=2023,
                  comments="Use of relative potency factors: (20 / 4 / RPF = 1)")
    ],
    starting_concentration=[
        StartingConcentration(
            # https://pd.lubw.de/54057
            # Abbildung 5.1
            np.array([1000., 3000.]), Substances.pfoa, Matrices.grw
        )]
)

nutrient_recovery = CaseStudy(
    "Nutrient Recovery",
    mixtures=[
        Mixture(0.875, 0.038265306, 45, 22.44897959,
                Treatments.dilww, Substances.pfoa, "standard",
                "x entered as range between 0.8 and 0.95, c entered as range between 1.0 and 89.0"),
        Mixture(0.21, 0.015306122, 0, 0,
                Treatments.sepev, Substances.pfoa, "standard",
                "x entered as range between 0.18 and 0.24"),
        Mixture(0.16, 0.016, 19180, 2290,
                Treatments.dilpr, Substances.pfoa, "standard",
                "x standard deviation assumed to be 10% of mean value,"
                " process water characteristics based on D4.5 table 12"),
        Mixture(0.99948, 9.18E-05, 0, 0,
                Treatments.dilrw, Substances.pfoa, "standard",
                "x entered as range between 0.9993 and 0.99966"),
        Mixture(0.875, 0.038265306, 45, 22.44897959,
                Treatments.dilww, Substances.pfoa, "cont_irrigation",
                "x entered as range between 0.8 and 0.95, c entered as range between 1.0 and 89.0"),
        Mixture(0.21, 0.015306122, 0, 0,
                Treatments.sepev, Substances.pfoa, "cont_irrigation",
                "x entered as range between 0.18 and 0.24"),
        Mixture(0.16, 0.016, 19180, 2290,
                Treatments.dilpr, Substances.pfoa, "cont_irrigation",
                "x standard deviation assumed to be 10% of mean value,"
                " process water characteristics based on D4.5 table 12"),
        Mixture(0.99948, 9.18e-5, 10, 5,
                Treatments.dilrw, Substances.pfoa, "cont_irrigation",
                "x entered as range between 0.9993 and 0.99966"),
    ],
    removal_percents=[
        RemovalPercent(np.array([100]), Treatments.wwro, Substances.pfoa)
    ],
    references=[
        Reference("",
                  Substances.pfoa, Matrices.pow,
                  ref_value_ng_l=7.36,
                  year=-1,
                  comments="Based on EFSA PFAS-4 TWI,"
                           " BCF and 70kg Body weights,"
                           " 0.26 lettuce consumption and 10% other sources safety factor")
    ],
    starting_concentration=[
        StartingConcentration(
            # PROMISCES D4.5
            # Plant No.3 (Table 12), minimum and maximum from mean +- 1.96 sd
            np.array([3600., 5900.]), Substances.pfoa, Matrices.lww
        )]
)

water_cycle_b = CaseStudy(
    name="Water Cycle B",
    mixtures=[
        Mixture(0.715, 0.0586734693877551, 0, 0,
                Treatments.dilsw, Substances.pfos, "standard",
                "x entered as range between 0.6 and 0.83"),
        Mixture(0.225, 0.0127551020408163, 0, 0,
                Treatments.dilgw, Substances.pfos, "standard",
                "x entered as range between 0.2 and 0.25"),
        Mixture(0.715, 0.0586734693877551, 0, 0,
                Treatments.dilsw, Substances.pfoa, "standard",
                "x entered as range between 0.6 and 0.83"),
        Mixture(0.225, 0.0127551020408163, 0, 0,
                Treatments.dilgw, Substances.pfoa, "standard",
                "x entered as range between 0.2 and 0.25"),
        Mixture(0.715, 0.0586734693877551, 0, 0,
                Treatments.dilsw, Substances.pfna, "standard",
                "x entered as range between 0.6 and 0.8"),
        Mixture(0.225, 0.0127551020408163, 0, 0,
                Treatments.dilgw, Substances.pfna, "standard",
                "x entered as range between 0.2 and 0.25"),
        Mixture(0.715, 0.0586734693877551, 0, 0,
                Treatments.dilsw, Substances.pfhxs, "standard",
                "x entered as range between 0.6 and 0.83"),
        Mixture(0.225, 0.0127551020408163, 0, 0,
                Treatments.dilgw, Substances.pfhxs, "standard",
                "x entered as range between 0.2 and 0.25"),
        Mixture(0.715, 0.0586734693877551, 0, 0,
                Treatments.dilsw, Substances.benzo, "standard",
                "x entered as range between 0.6 and 0.83"),
        Mixture(0.225, 0.0127551020408163, 0, 0,
                Treatments.dilgw, Substances.benzo, "standard",
                "x entered as range between 0.2 and 0.25"),
        Mixture(0.715, 0.0586734693877551, 0, 0,
                Treatments.dilsw, Substances._14_d, "standard",
                "x entered as range between 0.6 and 0.83"),
        Mixture(0.225, 0.012755102040816323, 0, 0,
                Treatments.dilgw, Substances._14_d, "standard",
                "x entered as range between 0.2 and 0.25")
    ],
    removal_percents=[],
    references=[
        Reference("TrinkwV 2023 + DOI 10.21945/RIVM-2018-0070", Substances.pfos, Matrices.drw, 3.07,
                  2023,
                  "Use of relative potency factors: (20 / 4 / RPF = 2)"),
        Reference("TrinkwV 2023 + DOI 10.21945/RIVM-2018-0070", Substances.pfoa, Matrices.drw, 6.13,
                  2023,
                  "Use of relative potency factors: (20 / 4 / RPF = 1)"),
        Reference("TrinkwV 2023 + DOI 10.21945/RIVM-2018-0070", Substances.pfna, Matrices.drw, 0.61,
                  2023,
                  "Use of relative potency factors: (20 / 4 / RPF = 10)"),
        Reference("TrinkwV 2023 + DOI 10.21945/RIVM-2018-0070", Substances.pfhxs, Matrices.drw, 10.18,
                  2023,
                  "Use of relative potency factors: (20 / 4 / RPF = 0.6)"),
        Reference("UBA GOW (july 2020)", Substances.benzo, Matrices.drw, 3000.0,
                  -1, ""),
        Reference("CA State Water Board 14dioxane", Substances._14_d, Matrices.drw, 1000.0,
                  -1, "")
    ],
    starting_concentration=[
        # Benedikt Master Thesis
        StartingConcentration(np.array([0.0, 1793.0]), Substances._14_d, Matrices.rww),
        StartingConcentration(np.array([17000.0, 44000.0]), Substances.benzo, Matrices.rww),
    ]
)

water_cycle_let = CaseStudy(
    name="Water Cycle LET",
    mixtures=[
        Mixture(0.715, 0.0586734693877551, 0, 0,
                Treatments.dilsw, Substances.pfos, "small_low_none",
                "x entered as range between 0.6 and 0.83"),
        Mixture(0.225, 0.0127551020408163, 0, 0,
                Treatments.dilgw, Substances.pfos, "small_low_none",
                "x entered as range between 0.2 and 0.25"),
        Mixture(0.715, 0.0586734693877551, 0, 0,
                Treatments.dilsw, Substances.pfos, "small_low_data",
                "x entered as range between 0.6 and 0.83"),
        Mixture(0.225, 0.0127551020408163, 0, 0,
                Treatments.dilgw, Substances.pfos, "small_low_data",
                "x entered as range between 0.2 and 0.25"),
        Mixture(0.925, 0.0127551020408163, 0, 0,
                Treatments.dilsw, Substances.pfos, "large_low_data",
                "x entered as range between 0.9 and 0.95"),
        Mixture(0.225, 0.0127551020408163, 0, 0,
                Treatments.dilgw, Substances.pfos, "large_low_data",
                "x entered as range between 0.2 and 0.25"),
        Mixture(0.925, 0.0127551020408163, 20, 2,
                Treatments.dilsw, Substances.pfos, "large_high_data",
                "x entered as range between 0.9 and 0.95"),
        Mixture(0.225, 0.0127551020408163, 0, 0,
                Treatments.dilgw, Substances.pfos, "large_high_data",
                "x entered as range between 0.2 and 0.25")
    ],
    removal_percents=[
        RemovalPercent(np.array([75, 80, 85]), Treatments.dwac, Substances.pfos)
    ],
    references=[
        Reference("TrinkwV 2023 + DOI 10.21945/RIVM-2018-0070", Substances.pfos, Matrices.drw, 2.5,
                  2023,
                  "Use of relative potency factors: (20 / 4 / RPF = 2)")
    ],
    starting_concentration=[
        # Point value for simple interpretation
        StartingConcentration(np.array([100.]), Substances.pfos, Matrices.rww)
    ]
)

water_reuse = CaseStudy(
    name="Water Reuse",
    mixtures=[
        Mixture(0.2, 0.0510204081632653, 0, 0,
                Treatments.dilrw, Substances.pfos, "standard",
                "x entered as range between o and 0.3"),
        Mixture(0.21, 0.015306122448979591, 0, 0,
                Treatments.sepev, Substances.pfos, "standard",
                "x entered as range between o and 0.24"),
    ],
    removal_percents=[],
    references=[
        Reference("Based on TWI 44 ng/kg bw (EFSA)", Substances.pfos, Matrices.pow, 0.39,
                  2020, "")
    ],
    starting_concentration=[]
)
