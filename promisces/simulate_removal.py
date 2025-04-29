from typing import Optional
import dataclasses as dtc
import numpy as np
import pandas as pd

from promisces.models.case_study import CaseStudy
from promisces.models.matrix import Matrix
from promisces.models.reference import Reference
from promisces.models.removal_percent import RemovalPercent
from promisces.models.starting_concentration import StartingConcentration
from promisces.models.substance import Substance
from promisces.models.treatment import TreatmentTrain
from promisces.removal_processes import ProcessResult, apply_generic_process, apply_mixture_process, \
    apply_separation_process, apply_separation_sludge_process


@dtc.dataclass
class SimulationResult:
    treatment_train: TreatmentTrain
    substance: Substance
    input_matrix: Matrix
    n_runs: int
    rmv_factor_resolution: int
    starting_concentration: np.ndarray
    intermediate_results: list[ProcessResult]
    case_study: CaseStudy | None = None
    reference: Reference | None = None
    scenario: str = ""

    @property
    def final_concentration(self) -> np.ndarray:
        return self.intermediate_results[-1].output_concentration

    @property
    def output_c_df(self):
        df = dict(input=self.starting_concentration)
        for result, treatment in zip(self.intermediate_results, self.treatment_train):
            df.update({treatment.id: result.output_concentration})
        return pd.DataFrame(df)

    @property
    def rmv_factor_df(self):
        return pd.DataFrame({
            treatment.id: result.rmv_factors
            for result, treatment in zip(self.intermediate_results, self.treatment_train)
        })

    @property
    def treatment_df(self):
        in_mat, out_mat = [self.input_matrix], []
        for treatment in self.treatment_train:
            out_mat += [treatment.get_output_matrix(in_mat[-1])]
            in_mat += [out_mat[-1]]
        in_mat = in_mat[:-1]
        return pd.DataFrame(dict(
            treatment_id=[t.id for t in self.treatment_train],
            average_out=[r.average_out for r in self.intermediate_results],
            input_matrix=[m.name for m in in_mat],
            output_matrix=[m.name for m in out_mat],
            dominant_data=[r.dominant_distribution.value for r in self.intermediate_results],
            substance=self.substance.name,
            n_runs=self.n_runs,
            removal_factor_resolution=self.rmv_factor_resolution,

        ))

    def export_excel(self, filename):
        with pd.ExcelWriter(filename) as excel_writer:
            self.treatment_df.to_excel(excel_writer, sheet_name='info', index=False)
            self.output_c_df.describe(
                percentiles=[0.5, 0.75, 0.9, 0.95, 0.975, 0.99]
            ).T.to_excel(excel_writer, sheet_name='output_c', index=False)
            self.rmv_factor_df.describe(
                percentiles=[0.5, 0.75, 0.9, 0.95, 0.975, 0.99]
            ).T.to_excel(excel_writer, sheet_name='removal', index=False)


def simulate_removal(
        treatment_train: TreatmentTrain,
        substance: Substance,
        input_matrix: Matrix,
        n_runs: int = 10000,
        rmv_factor_resolution: int = 1000,
        case_study: Optional[CaseStudy] = None,
        reference: Reference | None = None,
        scenario: str = ""
) -> SimulationResult:
    treatment_train.validate_matrices(input_matrix)

    if case_study is not None:
        starting_c = case_study.get_starting_concentration_for(substance, input_matrix)
        cs_removals = case_study.get_removal_percents_for(treatment_train, substance)
        mixtures = case_study.get_mixtures_for(treatment_train, substance, scenario)

    else:
        starting_c = None
        cs_removals = [RemovalPercent(np.array([]))] * len(treatment_train)
        mixtures = [None] * len(treatment_train)

    if starting_c is None:
        starting_c = StartingConcentration.from_lit(substance, input_matrix)
        if len(starting_c) == 0:
            raise RuntimeError("No literature starting concentration found for"
                               f" substance {substance.id} and input matrix {input_matrix.id}")
    start_c = input_c = starting_c.n_uniform_samples(n_runs)
    lit_removals = [RemovalPercent.from_lit(treatment, substance) for treatment in treatment_train]

    # validate / prompt for mixture data
    for i, treatment in enumerate(treatment_train):
        if treatment.requires_mixture and mixtures[i] is None:
            # todo prompt for input ?
            raise ValueError(f"treatment {treatment} needs mixture data")

    results = []
    for i, (treatment, cs_rmv, lit_rmv, mixture) in enumerate(
            zip(treatment_train, cs_removals, lit_removals, mixtures)
    ):
        # print(treatment.id)
        # fix input_c based on treatment id
        if treatment.id != "wwsl":
            input_c = np.flip(input_c)
        # dispatch to removal functions:
        if treatment.id.startswith("dil"):
            result = apply_mixture_process(input_c, **mixture.asdict())
        elif treatment.id == "sepev":
            result = apply_separation_process(input_c, **mixture.asdict())
        elif treatment.id == "wwsl":
            result = apply_separation_sludge_process(
                input_c,
                lit_rmv,
                cs_rmv,
                rmv_factor_resolution,
            )
        else:
            result = apply_generic_process(
                input_c,
                lit_rmv,
                cs_rmv,
                rmv_factor_resolution,
                n_runs=n_runs
            )

        results += [result]
        input_c = np.sort(result.output_concentration)

    if reference is None:
        # try to find one
        out_mat = treatment_train.output_matrix(input_matrix)
        if case_study is not None:
            matches = tuple(ref for ref in case_study.references
                            if ref.substance.id == substance.id and ref.matrix.id == out_mat.id)
        else:
            matches = tuple()
        if any(matches):
            reference = matches[0]
        else:
            # todo: assuming this **always** returns a ref
            reference = Reference.from_lit(out_mat, substance)
    return SimulationResult(
        treatment_train,
        substance,
        input_matrix,
        n_runs,
        rmv_factor_resolution,
        start_c,
        results,
        case_study,
        reference,
        scenario
    )


