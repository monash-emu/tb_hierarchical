from tbh.interventions import Scenario, ScreeningProgram, ScreeningTools

"""
    Scenario 1
    Continue as current at a rate of around 300/week (15,000/year)
    Screen 35,000 people in total by end 2026 for TB and TBI  (20,000 people unscreened)
    85% coverage of everyone >3yrs
    All people screened with CXR (30-40% of people >10yrs screened with universal sputum Xpert as well)
    70% TPT completion
    95% TB treatment success in identified cases
"""
scenario_1 = Scenario(
    sc_id="scenario_1",
    sc_name="Betio-like",
    scr_prgs=[
        ScreeningProgram(
            name="cxr_3+_75perc",
            start_time=2026,
            end_time=2027,
            total_coverage_perc=75.,
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
            total_coverage_perc=35 * .75, # 30-40% of those screened
            strata_coverage_multipliers={
                "age": {
                    "0": 0.,
                    "3": 0.
                }
            },
            scr_tool=ScreeningTools.Xpert_topup
        ),
        ScreeningProgram(
            name="tst_3+_75perc",
            start_time=2026,
            end_time=2027,
            total_coverage_perc=75.,
            strata_coverage_multipliers={
                "age": {
                    "0": 0.,
                }
            },
            scr_tool=ScreeningTools.TST
        ),
    ]
)


"""
    Scenario 2
    Continue as current at a rate of around 400/week (20,000/year)
    Screen 40,000 people in total by end 2026 for TB and TBI  (15,000 people unscreened)
    85% coverage of everyone >3yrs
    70% TPT completion
    95% TB treatment success in identified cases
"""
scenario_2 = Scenario(
    sc_id="scenario_2",
    sc_name="Betio-like, higher rate",
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
            total_coverage_perc=35. * .85,
            strata_coverage_multipliers={
                "age": {
                    "0": 0.,
                    "3": 0.
                }
            },
            scr_tool=ScreeningTools.Xpert_topup
        ),
        ScreeningProgram(
            name="tst_3+_85perc",
            start_time=2026,
            end_time=2027,
            total_coverage_perc=85.,
            strata_coverage_multipliers={
                "age": {
                    "0": 0.,
                }
            },
            scr_tool=ScreeningTools.TST
        ),
    ]
)

"""
    Scenario 3
    Continue as current, but drop sputum Xpert in those >10yrs (CXR with AI reading as main screen) at a rate of 500/week (25,000/year)
    Screen 45,000 people in total by end 2026 for TB and TBI  (10,000 people unscreened)
    85% coverage of everyone >3yrs (30-40% of people >10yrs screened with universal sputum Xpert as well)
    70% TPT completion
    95% TB treatment success in identified cases
"""
scenario_3 = Scenario(
    sc_id="scenario_3",
    sc_name="Drop sputum, much higher rate",
    scr_prgs=[
        ScreeningProgram(
            name="cxr_3+_95perc",
            start_time=2026,
            end_time=2027,
            total_coverage_perc=95.,
            strata_coverage_multipliers={
                "age": {
                    "0": 0.
                }
            },
            scr_tool=ScreeningTools.CXR
        ),
        ScreeningProgram(
            name="tst_3+_85perc",
            start_time=2026,
            end_time=2027,
            total_coverage_perc=95.,
            strata_coverage_multipliers={
                "age": {
                    "0": 0.,
                }
            },
            scr_tool=ScreeningTools.TST
        ),
    ]
)

"""
    Scenario 4
    Continue as current, but drop sputum Xpert in those >10yrs and stop screening children <10yrs (CXR with AI reading as main screen) at a rate of 600/week (30,000/year)
    Screen 50,000 people in total by end 2026 for TB and TBI  (5,000 people unscreened)
    85% coverage of everyone >10yrs (30-40% of people >10yrs screened with universal sputum Xpert as well)
    70% TPT completion
    95% TB treatment success in identified cases
"""

"""
    Scenario 5
    Continue as current, but drop sputum Xpert in those >10yrs (CXR with AI reading as main screen) at a rate of 700/week (30,000/year) AND stop TST (TBI screening + TPT, unless recent HH TB contact)
    Screen 55,000 people in total by end 2026 for TB only  (no people unscreened)
    85% coverage of everyone >3yrs (30-40% of people >10yrs screened with universal sputum Xpert as well)
    95% TB treatment success in identified cases
"""