library(conmat)
setwd("/Users/rrag0004/Documents/Code/tb_hierarchical/data/Rscript")
polymod_contact_data <- get_polymod_contact_data(setting="all")
polymod_survey_data <- get_polymod_population()
contact_model <- fit_single_contact_model(
  contact_data = polymod_contact_data,
  population = polymod_survey_data
)

# Load population data for Kiribati (South Tarawa)
kiribati_pop_df <- read.csv("KIR_pop_2025.csv")
kiribati_population = conmat_population(kiribati_pop_df, "lower.age.limit", "population")

synthetic_KIR_predictions <- predict_contacts(
  model = contact_model,
  population = kiribati_population,
  age_breaks = c(kiribati_pop_df$lower.age.limit, Inf)
)
write.csv(synthetic_KIR_predictions, "conmat_all_KIR.csv")

matrix = predictions_to_matrix(synthetic_KIR_predictions)
autoplot(matrix)
