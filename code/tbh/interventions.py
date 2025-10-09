from numpy import log
from summer2.functions import time as stf
from summer2.parameters import Parameter


class ScreeningTools:

    # TB Infection screening and TPT
    TST = {
        "sensitivities": {
            "incipient": Parameter('prev_se_incipient'),
            "contained": Parameter('prev_se_contained'),
        },
        "dest_comp": "cleared",
        "success_prop": Parameter('tpt_completion_perc') / 100.      # probability of completing TPT if screened positive 
    }

    # TB Disease screening
    CXR = {
        "sensitivities": {
            "subclin_noninf": Parameter('prev_se_subclin_noninf_cxr'),
            "clin_noninf": Parameter('prev_se_clin_noninf_cxr'),
            "subclin_inf": Parameter('prev_se_subclin_inf_cxr'),
            "clin_inf": Parameter('prev_se_clin_inf_cxr')
        },
        "dest_comp": "treatment",
        "success_prop": 1.  # probability of getting started on treatment if screened positive 
    }

    Xpert_topup = {
        # sensitivities need to be seen as additional sensitivity compared to screening based on CXR alone
        "sensitivities": {
            "subclin_noninf": Parameter('prev_se_subclin_noninf_pearl') - Parameter('prev_se_subclin_noninf_cxr'),
            "clin_noninf": Parameter('prev_se_clin_noninf_pearl') - Parameter('prev_se_clin_noninf_cxr'),
            "subclin_inf": Parameter('prev_se_subclin_inf_pearl') - Parameter('prev_se_subclin_inf_cxr'),
            "clin_inf": Parameter('prev_se_clin_inf_pearl') - Parameter('prev_se_clin_inf_cxr')
        },
        "dest_comp": "treatment",
        "success_prop": 1.  # probability of getting started on treatment if screened positive 
    }


class ScreeningProgram:
    def __init__(self, name, start_time, end_time, total_coverage_perc, strata_coverage_multipliers, scr_tool):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.total_coverage_perc = total_coverage_perc
        self.strata_coverage_multipliers = strata_coverage_multipliers
        self.scr_tool = scr_tool

        self.get_raw_screening_func()

    def get_raw_screening_func(self):
        """
        Calculate the constant per-time-unit screening rate that would yield the 
        observed total coverage percentage over the campaign duration.
        Uses the formula: rate = -log(1 - coverage_prop) / duration

        Returns
        -------
        float
            The estimated per-time-unit screening rate.
        """
        duration = self.end_time - self.start_time
        assert duration > 0, "End time must be after Start time"
                
        scr_rate = - log(1 - self.total_coverage_perc / 100.) / duration

        # FIXME: Not sure about the timeseries below
        self.raw_screening_func = stf.get_linear_interpolation_function(
            [self.start_time - 0.01, self.start_time, self.end_time - 0.01, self.end_time], 
            [0., scr_rate, scr_rate, 0.]
        )


class Scenario:
    def __init__(self, sc_id, sc_name, scr_prgs, params_ow={}):
        self.sc_id = sc_id
        self.sc_name = sc_name
        self.scr_prgs = scr_prgs
        self.params_ow = params_ow


example_scr_program = ScreeningProgram(
    name="betio_cxr_screening",
    start_time=2026,
    end_time=2027,
    total_coverage_perc=85.,
    strata_coverage_multipliers={
        "age": {
            "0": 0.
        }
    },
    scr_tool=ScreeningTools.CXR
)

example_xpert_program = ScreeningProgram(
    name="betio_xpert_screening",
    start_time=2024,
    end_time=2026,
    total_coverage_perc=85,
    strata_coverage_multipliers={
        "age": {
            "0": 0.,
            "3": 0.
        }
    },
    scr_tool=ScreeningTools.Xpert_topup
)

example_tpt_program = ScreeningProgram(
    name="betio_tst_screening",
    start_time=2024,
    end_time=2026,
    total_coverage_perc=85,
    strata_coverage_multipliers={
        "age": {
            "0": 0.,
        }
    },
    scr_tool=ScreeningTools.TST
)