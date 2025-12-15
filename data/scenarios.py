from tbh.interventions import Scenario, ScreeningProgram, ScreeningTools

COVERAGE = {
    "low": 0.60,
    "med": 0.70,
    "high": 0.80,
    "vhigh": 0.90,
    "max": 0.95,
}
EFF_ENROLMENT_PERC = 85.
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

# Scenarios 1, 2, 3 are based on current program but varying coverage (low, med, high)
for sc_num, coverage_key in enumerate(["low", "med", "high"], start=1):
    coverage = COVERAGE[coverage_key]
    scr_prgs = [
        make_scr_program(
            scr_tool=ScreeningTools.CXR,
            name="cxr_3+",
            coverage=coverage,
            ages_excluded=["0"]
        ),
        make_scr_program(
            scr_tool=ScreeningTools.Xpert_topup,
            name="xpert_10+",
            coverage=coverage * 0.35,  # 30-40% of those screened
            ages_excluded=["0", "3"]
        ),
        make_scr_program(
            scr_tool=ScreeningTools.TST,
            name="tst_3+",
            coverage=coverage,
            ages_excluded=["0"]
        ),
    ]

    SCENARIOS.append(
        Scenario(
            sc_id=f"scenario_{sc_num}",
            sc_name=f"{sc_num}. Current / {coverage_key.capitalize()}",
            scr_prgs=scr_prgs
        )
    )

# Scenarios 4, 5, 6 consider dropping Xpert with varying coverage (med, high, vhigh)
for sc_num, coverage_key in enumerate(["med", "high", "vhigh"], start=4):
    coverage = COVERAGE[coverage_key]
    scr_prgs = [
        make_scr_program(
            scr_tool=ScreeningTools.CXR,
            name="cxr_3+",
            coverage=coverage,
            ages_excluded=["0"]
        ),
        make_scr_program(
            scr_tool=ScreeningTools.TST,
            name="tst_3+",
            coverage=coverage,
            ages_excluded=["0"]
        ),
    ]
    SCENARIOS.append(
        Scenario(
            sc_id=f"scenario_{sc_num}",
            sc_name=f"{sc_num}. Drop Xpert / {coverage_key.capitalize()}",
            scr_prgs=scr_prgs
        )
    )

# Scenario 7, 8, 9 consider dropping Xpert and stopping screening children <10yrs with varying coverage (high, vhigh, max)
for sc_num, coverage_key in enumerate(["high", "vhigh", "max"], start=7):
    coverage = COVERAGE[coverage_key]
    scr_prgs = [
        make_scr_program(
            scr_tool=ScreeningTools.CXR,
            name="cxr_10+",
            coverage=coverage,
            ages_excluded=["0", "3"]
        ),
        make_scr_program(
            scr_tool=ScreeningTools.TST,
            name="tst_10+",
            coverage=coverage,
            ages_excluded=["0", "3"]
        ),
    ]
    SCENARIOS.append(
        Scenario(
            sc_id=f"scenario_{sc_num}",
            sc_name=f"{sc_num}. Drop Xpert - Scr 10+ / {coverage_key.capitalize()}",
            scr_prgs=scr_prgs
        )
    )   

# Scenario 10, 11, 12 consider dropping Xpert and TST and stopping screening children <10yrs with varying coverage (high, vhigh, max)
for sc_num, coverage_key in enumerate(["high", "vhigh", "max"], start=10):
    coverage = COVERAGE[coverage_key]
    scr_prgs = [
        make_scr_program(
            scr_tool=ScreeningTools.CXR,
            name="cxr_10+",
            coverage=coverage,
            ages_excluded=["0", "3"]
        ),
    ]
    SCENARIOS.append(
        Scenario(
            sc_id=f"scenario_{sc_num}",
            sc_name=f"{sc_num}. Drop Xpert & TST / {coverage_key.capitalize()}",
            scr_prgs=scr_prgs
        )
    )   
