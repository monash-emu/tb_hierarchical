from tbh.interventions import Scenario, ScreeningProgram, ScreeningTools

COVERAGE = {
    "low": 0.55,
    "med": 0.65,
    "high": 0.75,
    "vhigh": 0.85,
    "max": 0.95,
}
cov_list = list(COVERAGE.keys())

EFF_ENROLMENT_PERC = 100.
START_TIME, END_TIME = 2026, 2027

# HELPER FUNCTION
def make_scr_program(scr_tool, name, coverage, ages_excluded=None):
    if ages_excluded:
        strata_coverage_multipliers={
            "age": {age: 0. for age in ages_excluded}
        }
    else:
        strata_coverage_multipliers={}

    return ScreeningProgram(
        name=name,
        start_time=START_TIME,
        end_time=END_TIME,
        total_coverage_perc=coverage * EFF_ENROLMENT_PERC,
        strata_coverage_multipliers=strata_coverage_multipliers,
        scr_tool=scr_tool
    )

SCENARIOS = []

# Scenarios 1-5 are based on current program but varying coverage (low, med, high, vhigh, max)
desc = "PEARL algorithm: CXR-Xpert for 35% of 10+yr olds (CXR only for other 65%), SSX and TST for 3+yr olds"
for sc_num, coverage_key in enumerate(cov_list, start=1):
    coverage = COVERAGE[coverage_key]
    scr_prgs = [
        make_scr_program(
            scr_tool=ScreeningTools.PEARL,
            name="pearl_10+",
            coverage=coverage,
            ages_excluded=["0", "3", "5"]
        ),
        make_scr_program(
            scr_tool=ScreeningTools.SSX, # symptom screen only for 3-9yr olds
            name="ssx_3_9",
            coverage=coverage, 
            ages_excluded=["0", "10", "15", "18", "40", "65"]
        ),
        make_scr_program(
            scr_tool=ScreeningTools.TST, # TST for 3+yr olds
            name="tst_3+",
            coverage=coverage,
            ages_excluded=["0"]
        ),
    ]

    SCENARIOS.append(
        Scenario(
            sc_id=f"scenario_{sc_num}",
            sc_name=f"{sc_num}. PEARL / {coverage_key.capitalize()}",
            scr_prgs=scr_prgs,
            desc=desc
        )
    )

# Scenarios 6-10 consider dropping Xpert with varying coverage
desc = "Dropping Xpert: CXR for 10+yr olds, SSx and TST for 3+yr olds"
for sc_num, coverage_key in enumerate(cov_list, start=6):
    coverage = COVERAGE[coverage_key]
    scr_prgs = [
        make_scr_program(
            scr_tool=ScreeningTools.CXR,
            name="cxr_10+",
            coverage=coverage,
            ages_excluded=["0", "3", "5"]
        ),
        make_scr_program(
            scr_tool=ScreeningTools.SSX, # symptom screen only for 3-9yr olds
            name="ssx_3_9",
            coverage=coverage, 
            ages_excluded=["0", "10", "15", "18", "40", "65"]
        ),
        make_scr_program(
            scr_tool=ScreeningTools.TST, # TST for 3+yr olds
            name="tst_3+",
            coverage=coverage,
            ages_excluded=["0"]
        ),
    ]
    SCENARIOS.append(
        Scenario(
            sc_id=f"scenario_{sc_num}",
            sc_name=f"{sc_num}. CXR-TST / {coverage_key.capitalize()}",
            scr_prgs=scr_prgs,
            desc=desc
        )
    )

# Scenario 11-15 consider dropping Xpert and stopping screening children <10yrs with varying coverage
desc = "Dropping Xpert and stop screening <10yrs: CXR, SSx and TST for 10+yr olds only"
for sc_num, coverage_key in enumerate(cov_list, start=11):
    coverage = COVERAGE[coverage_key]
    scr_prgs = [
        make_scr_program(
            scr_tool=ScreeningTools.CXR,
            name="cxr_10+",
            coverage=coverage,
            ages_excluded=["0", "3", "5"]
        ),
        make_scr_program(
            scr_tool=ScreeningTools.TST, 
            name="tst_10+",
            coverage=coverage,
            ages_excluded=["0", "3", "5"]
        ),
    ]
    SCENARIOS.append(
        Scenario(
            sc_id=f"scenario_{sc_num}",
            sc_name=f"{sc_num}. CXR-TST 10+yrs / {coverage_key.capitalize()}",
            scr_prgs=scr_prgs,
            desc=desc
        )
    )     

# Scenario 16-20 consider dropping Xpert and TST and stopping screening children <10yrs with varying coverage
desc = "Dropping Xpert and TST, stop screening <10yrs: CXR and SSx for 10+yr olds only"
for sc_num, coverage_key in enumerate(cov_list, start=16):
    coverage = COVERAGE[coverage_key]
    scr_prgs = [
        make_scr_program(
            scr_tool=ScreeningTools.CXR,
            name="cxr_10+",
            coverage=coverage,
            ages_excluded=["0", "3", "5"]
        ),
    ]
    SCENARIOS.append(
        Scenario(
            sc_id=f"scenario_{sc_num}",
            sc_name=f"{sc_num}. CXR 10+yrs / {coverage_key.capitalize()}",
            scr_prgs=scr_prgs,
            desc=desc
        )
    )   

# Scenarios 21-25 consider CXR TST and PLTS with varying coverage
desc = "Use Tongue Swabs (PLTS): CXR and PLTS for 10+yr olds, SSx and TST for 3+yr olds"
for sc_num, coverage_key in enumerate(cov_list, start=21):
    coverage = COVERAGE[coverage_key]
    scr_prgs = [
        make_scr_program(
            scr_tool=ScreeningTools.PLTS,
            name="plts_10+",
            coverage=coverage,
            ages_excluded=["0", "3", "5"]
        ),
        make_scr_program(
            scr_tool=ScreeningTools.SSX, # symptom screen only for 3-9yr olds
            name="ssx_3_9",
            coverage=coverage, 
            ages_excluded=["0", "10", "15", "18", "40", "65"]
        ),
        make_scr_program(
            scr_tool=ScreeningTools.TST, # TST for 3+yr olds  
            name="tst_3+",
            coverage=coverage,
            ages_excluded=["0"]
        ),
    ]
    SCENARIOS.append(
        Scenario(
            sc_id=f"scenario_{sc_num}",
            sc_name=f"{sc_num}. CXR-PLTS-TST / {coverage_key.capitalize()}",
            scr_prgs=scr_prgs,
            desc=desc
        )
    )