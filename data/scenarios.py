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
    sc_name="1. Current / Low",
    scr_prgs=[
        ScreeningProgram(
            name="cxr_3+",
            start_time=2026,
            end_time=2027,
            total_coverage_perc=85. * 15/35,
            strata_coverage_multipliers={
                "age": {
                    "0": 0.
                }
            },
            scr_tool=ScreeningTools.CXR
        ),
        ScreeningProgram( 
            name="xpert_10+",
            start_time=2026,
            end_time=2027,
            total_coverage_perc=85. * 0.35 * 15/35, # 30-40% of those screened
            strata_coverage_multipliers={
                "age": {
                    "0": 0.,
                    "3": 0.
                }
            },
            scr_tool=ScreeningTools.Xpert_topup
        ),
        ScreeningProgram(
            name="tst_3+",
            start_time=2026,
            end_time=2027,
            total_coverage_perc=85. * 15/35,
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
    sc_name="2. Current / Med",
    scr_prgs=[
        ScreeningProgram(
            name="cxr_3+",
            start_time=2026,
            end_time=2027,
            total_coverage_perc=85. * 20/35,
            strata_coverage_multipliers={
                "age": {
                    "0": 0.
                }
            },
            scr_tool=ScreeningTools.CXR
        ),
        ScreeningProgram( 
            name="xpert_10+",
            start_time=2026,
            end_time=2027,
            total_coverage_perc=85. * 0.35 * 20/35, # 30-40% of those screened
            strata_coverage_multipliers={
                "age": {
                    "0": 0.,
                    "3": 0.
                }
            },
            scr_tool=ScreeningTools.Xpert_topup
        ),
        ScreeningProgram(
            name="tst_3+",
            start_time=2026,
            end_time=2027,
            total_coverage_perc=85. * 20/35,
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
    sc_name="3. Drop Xpert / High",
    scr_prgs=[
        ScreeningProgram(
            name="cxr_3+",
            start_time=2026,
            end_time=2027,
            total_coverage_perc=85. * 25/35,
            strata_coverage_multipliers={
                "age": {
                    "0": 0.
                }
            },
            scr_tool=ScreeningTools.CXR
        ),
        ScreeningProgram(
            name="tst_3+",
            start_time=2026,
            end_time=2027,
            total_coverage_perc=85. * 25/35,
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
scenario_4 = Scenario(
    sc_id="scenario_4",
    sc_name="4. Drop Xpert - Scr 10+ / VHigh",
    scr_prgs=[
        ScreeningProgram(
            name="cxr_10+",
            start_time=2026,
            end_time=2027,
            total_coverage_perc=85. * 30/35,
            strata_coverage_multipliers={
                "age": {
                    "0": 0.,
                    "3": 0.
                }
            },
            scr_tool=ScreeningTools.CXR
        ),
        ScreeningProgram(
            name="tst_3+",
            start_time=2026,
            end_time=2027,
            total_coverage_perc=85. * 30/35,
            strata_coverage_multipliers={
                "age": {
                    "0": 0.,
                    "3": 0.
                }
            },
            scr_tool=ScreeningTools.TST
        ),
    ]
)

"""
    Scenario 5
    Continue as current, but drop sputum Xpert in those >10yrs (CXR with AI reading as main screen) at a rate of 700/week (30,000/year) AND stop TST (TBI screening + TPT, unless recent HH TB contact)
    Screen 55,000 people in total by end 2026 for TB only  (no people unscreened)
    85% coverage of everyone >3yrs (30-40% of people >10yrs screened with universal sputum Xpert as well)
    95% TB treatment success in identified cases
"""
scenario_5 = Scenario(
    sc_id="scenario_5",
    sc_name="5. Drop Xpert & TST / Max",
    scr_prgs=[
        ScreeningProgram(
            name="cxr_3+",
            start_time=2026,
            end_time=2027,
            total_coverage_perc=85. * 35/35,
            strata_coverage_multipliers={
                "age": {
                    "0": 0.
                }
            },
            scr_tool=ScreeningTools.CXR
        ),
    ]
)