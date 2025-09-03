from jax import numpy as jnp

class ScreeningTools:

    # TB Infection screening and TPT
    TST = {
        "sensitivities": {
            "incipient": .8,
            "contained": .8,
        },
        "dest_comp": "cleared",
        "success_prop": .70       # probability of completing TPT if screened positive 
    }

    IGRA = {
        "sensitivities": {
            "incipient": .9,
            "contained": .9,
        },
        "dest_comp": "cleared",
        "success_prop": .70       # probability of completing TPT if screened positive 
    }

    # TB Disease screening
    CXR = {
        "sensitivities": {
            "subclin_noninf": 0.,
            "clin_noninf": .5,
            "subclin_inf": 0.,
            "clin_inf": .9
        },
        "dest_comp": "treatment",
        "success_prop": 1.  # probability of getting started on treatment if screened positive 
    }

    Xpert = {
        "sensitivities": {
            "subclin_noninf": .5,
            "clin_noninf": .7,
            "subclin_inf": .8,
            "clin_inf": .9
        },
        "dest_comp": "treatment",
        "success_prop": 1.  # probability of getting started on treatment if screened positive 
    }


class ScreeningProgram:
    def __init__(self, name, start_time, end_time, total_coverage_perc, included_strata, scr_tool):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.total_coverage_perc = total_coverage_perc
        self.included_strata = included_strata
        self.scr_tool = scr_tool

        self.get_screening_rate()

    def get_screening_rate(self):
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
                
        return - jnp.log(1 - self.total_coverage_perc / 100.) / duration
    

example_scr_program = ScreeningProgram(
    name="betio_xpert_screening",
    start_time=2024,
    end_time=2026,
    total_coverage_perc=85,
    included_strata={"age": 0},
    scr_tool=ScreeningTools.Xpert
)