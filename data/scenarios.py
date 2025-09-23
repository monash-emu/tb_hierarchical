from tbh.interventions import Scenario, ScreeningProgram, ScreeningTools


scenario_1 = Scenario(
    sc_id="scenario_1",
    sc_name="Betio-like screening",
    scr_prgs=[
        ScreeningProgram(
            name="cxr_3+_85perc",
            start_time=2026,
            end_time=2027,
            total_coverage_perc=85.,
            strata_coverage_multipliers={
                "age": {
                    "0": 0.
                }
            },
            scr_tool=ScreeningTools.CXR
        ),
        ScreeningProgram( 
            name="xpert_10+_35perc",
            start_time=2026,
            end_time=2027,
            total_coverage_perc=35,
            strata_coverage_multipliers={
                "age": {
                    "0": 0.,
                    "3": 0.
                }
            },
            scr_tool=ScreeningTools.Xpert_topup
        )
    ]
)
