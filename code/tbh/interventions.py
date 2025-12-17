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
    SSX = { # symptom screening
        "sensitivities": {
            tb_state: Parameter(f"prev_se_{tb_state}_ssx") for tb_state in [
                "subclin_lowinf", "clin_lowinf", "subclin_inf", "clin_inf"
            ]
        },
        "dest_comp": "treatment",
        "success_prop": 1.  # probability of getting started on treatment if screened positive
    }
    
    CXR = { # chest X-ray in addition to symptom screening
        "sensitivities": {
            tb_state: Parameter(f"prev_se_{tb_state}_cxr") for tb_state in [
                "subclin_lowinf", "clin_lowinf", "subclin_inf", "clin_inf"
            ]
        },
        "dest_comp": "treatment",
        "success_prop": 1.  # probability of getting started on treatment if screened positive 
    }

    PLTS = { # Pluslife Tongue Swab in addition to symptom screening and CXR
        "sensitivities": {
            tb_state: Parameter(f"prev_se_{tb_state}_plts") for tb_state in [
                "subclin_lowinf", "clin_lowinf", "subclin_inf", "clin_inf"
            ]
        },
        "dest_comp": "treatment",
        "success_prop": 1.  # probability of getting started on treatment if screened positive 
    }

    PEARL = { # Xpert MTB/RIF Ultra in addition to symptom screening and CXR (Xpert done for about 35%, other 65% get CXR only)
        "sensitivities" : {
            tb_state: .35 * Parameter(f"prev_se_{tb_state}_pearl") + .65 * Parameter(f"prev_se_{tb_state}_cxr") for tb_state in [
                "subclin_lowinf", "clin_lowinf", "subclin_inf", "clin_inf"
            ]
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
    def __init__(self, sc_id, sc_name, scr_prgs, desc="", params_ow={}):
        self.sc_id = sc_id
        self.sc_name = sc_name
        self.scr_prgs = scr_prgs
        self.params_ow = params_ow
        self.description = desc


example_cxr_program = ScreeningProgram(
    name="betio_cxr_screening",
    start_time=2026,
    end_time=2027,
    total_coverage_perc=85.,
    strata_coverage_multipliers={
        "age": {
            "0": 0.,
            "3": 0.,
            "5": 0.
        }
    },
    scr_tool=ScreeningTools.CXR
)

example_xpert_program = ScreeningProgram(
    name="betio_xpert_screening",
    start_time=2026,
    end_time=2027,
    total_coverage_perc=85,
    strata_coverage_multipliers={
        "age": {
            "0": 0.,
            "3": 0.,
            "5": 0.
        }
    },
    scr_tool=ScreeningTools.PEARL
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