library(tidyverse)
library(vroom)
library(forecast)
library(patchwork)
library(embed)
library(tidymodels)

kobetest <- vroom("/kaggle/input/kobe-bryant-shot-selection/data.csv.zip")

## Converting it to polar coordinates

dist <- sqrt((kobetest$loc_x/10)^2 + (kobetest$loc_y/10)^2)
kobetest$shot_distance <- dist

#Creating angle column
kobetest$angle_to_basket <- atan2(kobetest$loc_y, kobetest$loc_x)

# Create one time variable
kobetest$time_remaining = (kobetest$minutes_remaining*60)+kobetest$seconds_remaining

# Home and Away
kobetest$matchup = ifelse(str_detect(kobetest$matchup, 'vs.'), 'Home', 'Away')

# Season
kobetest['season'] <- substr(str_split_fixed(kobetest$season, '-',2)[,2],2,2)

#Game Number
kobetest$game_num <- as.numeric(kobetest$game_date)

# First and Last 2 Seasons
kobetest$f2l2 <- ifelse(kobetest$game_num > 1452 | kobetest$game_num < 150, 1, 0)


#In final 2 minutes of 4th quarter and overtime
kobetest$is_clutch <- ifelse(kobetest$seconds_remaining <= 120 & kobetest$period >= 4, 1, 0)

# period into a factor
kobetest$period <- as.factor(kobetest$period)



kobe_f <- kobetest %>%
  select(-c('shot_id', 'team_id', 'team_name', 'shot_zone_range', 'lon', 'lat',
            'seconds_remaining', 'minutes_remaining', 'game_event_id',
            'game_id', 'game_date','shot_zone_area',
            'shot_zone_basic', 'loc_x', 'loc_y'))

# Train
train <- kobe_f %>%
  filter(!is.na(shot_made_flag))

# Test
test <- kobe_f %>%
  filter(is.na(shot_made_flag))

# Make the response variable into a factor
train$shot_made_flag <- as.factor(train$shot_made_flag)

recipe <- recipe(shot_made_flag ~ ., data = train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors())

test.id <- kobetest %>% filter(is.na(shot_made_flag)) %>% select(shot_id)

library(doParallel)

num_cores <- parallel::detectCores()

cl <- makePSOCKcluster(num_cores)

registerDoParallel(cl)

## Create a workflow with model & recipe

my_mod_kf <- rand_forest(mtry = tune(),
                         min_n=tune(),
                         trees=800) %>%
  set_engine("ranger") %>%
  set_mode("classification")


kobe_workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(my_mod_kf)

## Set up grid of tuning values

tuning_grid <- grid_regular(mtry(range = c(1,(ncol(train)-1))),
                            min_n(),
                            levels = 3)

folds <- vfold_cv(train, v = 3, repeats=1)

CV_results <- kobe_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

best_tune_rf <- CV_results %>%
  select_best()

## Finalize workflow and predict

final_wf <- kobe_workflow %>%
  finalize_workflow(best_tune_rf) %>%
  fit(data=train)

kobe_predictions_rf <- final_wf %>% predict(new_data=test,
                                            type="prob")
                            
kobe_rf_submit <- as.data.frame(cbind(test.id, as.character(kobe_predictions_rf$.pred_1)))

colnames(kobe_rf_submit) <- c("shot_id", "shot_made_flag")

write_csv(kobe_rf_submit, "kobe_rf_submit3.csv")

stopCluster(cl)





                            
