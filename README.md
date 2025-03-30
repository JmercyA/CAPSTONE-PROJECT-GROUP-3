# Predicting Traffic Incident Severity 

![Alt text](https://github.com/JmercyA/CAPSTONE-PROJECT-GROUP-3/blob/main/images/0_ipvJpQLmhlrch90Q.jpg?raw=true)


## Table of Contents

- [Project Description](#Description)
- [Business Understanding](#business_understanding) 
- [Data Loading and data understanding)](#Data_Loading_and_data_understanding)  
- [Data cleaning](#Data_cleaning)  
- [EDA](#Exploratory_Data_analysis)  
- [Modelling](#Modelling)  
- [Feature Importance](#feature-importance)  
- [Model tuning](#model_tuning)  
- [Recommendations](#model_tuning)    

## Project description.

The goal of this project is to develop a predictive model that can estimate the severity of traffic incidents based on various factors such as weather conditions (rain, snow, fog, etc.) and time-related elements (time of day, day of the week, and holidays). The model will utilize historical traffic incident data from San Fransisco open data source, weather patterns, and temporal factors to predict the likelihood of incidents leading to Mild and severe outcomes, such as injuries or fatalities, rather than minor disruptions. This will help transportation authorities, emergency services, and city planners make data-driven decisions, optimize response strategies, and improve public safety.


## Business Understanding

aiming to reduce the devastating consequences of road traffic accidents,this project employs machine learning to predict accident severity. The model, trained on data including vehicle type,casualty details,the type of the road, the location, the weather at the time of the accident just to mention a few, that can be used to inform proactive strategies aimed at reducing fatalities and injuries.

Traffic incidents contribute significantly to congestion, injuries, fatalities, and economic losses. Understanding the factors that influence the severity of these incidents can help reduce the overall impact on society. By leveraging data science to predict the severity of traffic incidents based on weather and time related variables, transportation agencies can:

- ```Improve Safety```: Predicting severe incidents allows for timely interventions, such as dispatching emergency services more effectively.


- ```Optimize Resource Allocation```: Traffic management and emergency responders can allocate resources in advance based on predicted severity, ensuring quicker response times in critical situations.

- ```Enhance Traffic Management```: Better understanding of incident severity can guide traffic signal optimization, road closures, and detour planning to minimize disruptions.

- ```Promote Public Awareness```: Through predictive insights, authorities can inform drivers about weather-related risks and encourage safer driving practices during high-risk periods.

This project aims to create a solution that not only reduces the severity of traffic incidents but also improves overall traffic flow and safety.



## Stakeholders
|Stakeholder	            |      Interest                                        |
|---------------------------|------------------------------------------------------|
|Transportation Authorities | to optimize response times                           |
|Emergency Services         | prepare resources and prioritize high-risk incidents.|
|Public and Drivers         | increased safety, fewer severe accidents             |
|Insurance Companies        | to optimize their pricing models, assess risk        |

## Data Loading and Data Understanding

- Data Source: data sourced from https://data.sfgov.org/Public-Safety/Traffic-Crashes-Resulting-in-Injury/ubvf-ztfx/about_data

- the traffic data set has 61229 rows and 63 columns (before cleaning)

### Dataset Description

#### Overview
This dataset contains information about traffic accidents, with a total of 61,229 entries. It includes various attributes related to the accident details, such as location, time, collision type, and injury data. The dataset is valuable for analysis related to traffic safety, accident severity, and patterns across different areas and conditions.

#### Features
The dataset consists of the following columns:

- **unique_id**: Unique identifier for each accident (int64).
- **cnn_intrsctn_fkey**: Foreign key linking to intersection data (float64).
- **cnn_sgmt_fkey**: Foreign key linking to segment data (float64).
- **case_id_pkey**: Primary key for case ID (object).
- **tb_latitude**: Latitude of the accident (float64).
- **tb_longitude**: Longitude of the accident (float64).
- **geocode_source**: Source of geocoding data (object).
- **geocode_location**: Geocoded location information (object).
- **collision_datetime**: Date and time of the collision (object).
- **collision_date**: Date of the collision (object).
- **collision_time**: Time of the collision (object).
- **accident_year**: Year of the accident (int64).
- **month**: Month of the accident (object).
- **day_of_week**: Day of the week when the accident occurred (object).
- **time_cat**: Time category of the accident (object).
- **juris**: Jurisdiction (object).
- **officer_id**: ID of the officer reporting the accident (object).
- **reporting_district**: District from which the report originated (object).
- **beat_number**: Beat number assigned to the report (object).
- **primary_rd**: Primary road involved in the accident (object).
- **secondary_rd**: Secondary road involved in the accident (object).
- **distance**: Distance from the reference point (float64).
- **direction**: Direction of travel (object).
- **weather_1**: Primary weather condition at the time of the accident (object).
- **weather_2**: Secondary weather condition (object).
- **collision_severity**: Severity of the collision (object).
- **type_of_collision**: Type of collision (object).
- **mviw**: Motor vehicle involvement (object).
- **ped_action**: Pedestrian action (object).
- **road_surface**: Condition of the road surface (object).
- **road_cond_1**: Primary road condition (object).
- **road_cond_2**: Secondary road condition (object).
- **lighting**: Lighting condition at the time of the accident (object).
- **control_device**: Type of traffic control device at the intersection (object).
- **intersection**: Whether the accident occurred at an intersection (object).
- **vz_pcf_code**: Code for the Vehicle-Zone Potential Conflict (object).
- **vz_pcf_group**: Grouping of the Vehicle-Zone Potential Conflict (object).
- **vz_pcf_description**: Description of the Vehicle-Zone Potential Conflict (object).
- **vz_pcf_link**: Link to more information about the potential conflict (object).
- **number_killed**: Number of people killed in the accident (float64).
- **number_injured**: Number of people injured in the accident (int64).
- **street_view**: Street view image URL of the accident location (object).
- **dph_col_grp**: Department of Public Health collision group (object).
- **dph_col_grp_description**: Description of the Department of Public Health collision group (object).
- **party_at_fault**: Party at fault for the accident (float64).
- **party1_type**: Type of the first party involved in the accident (object).
- **party1_dir_of_travel**: Direction of travel for the first party (object).
- **party1_move_pre_acc**: Movement of the first party before the accident (object).
- **party2_type**: Type of the second party involved in the accident (object).
- **party2_dir_of_travel**: Direction of travel for the second party (object).
- **party2_move_pre_acc**: Movement of the second party before the accident (object).
- **point**: Geospatial point data for the accident (object).
- **data_as_of**: Date the data was last updated (object).
- **data_updated_at**: Timestamp when the data was last updated (object).
- **data_loaded_at**: Timestamp when the data was loaded (object).
- **analysis_neighborhood**: Neighborhood where the accident occurred (object).
- **supervisor_district**: Supervisor district where the accident occurred (float64).
- **police_district**: Police district where the accident occurred (object).
- **Current Police Districts**: Police districts for the current data (float64).
- **Current Supervisor Districts**: Supervisor districts for the current data (float64).
- **Analysis Neighborhoods**: Neighborhoods relevant for analysis (float64).
- **Neighborhoods**: Neighborhoods associated with the accident (float64).
- **SF Find Neighborhoods**: Neighborhoods found by the SF Find algorithm (float64).

- **Missing values**: Some columns contain missing values. For example, `cnn_sgmt_fkey`, `party_at_fault`, and `party2_type` have missing data for some records.

#### Usage
This dataset will be used for various analyses related to traffic accidents, including:
- Accident severity analysis
- Geospatial analysis of accident hotspots
- Study of weather, road conditions, and lighting effects on accidents
- Machine learning for predicting accident severity based on features like weather, location, and vehicle type

## Data Cleaning
 1. droping the unnecessary columns
   - Columns such as accident id, police district, Neighboorhoods were droped as they are not necessary in achieving the objective
 2. Dealing with null values
   - droping columns with a higher percentage of null values
   - Dropping rows in columns with few missing values
   - Filling with median for numerical columns and 'unknown' for categorical columns.
 3. Handling outliers 
   - Outliers were dealt with by capping them to the upper and lower bound
 4. Replacing less informative or rare values with a more general category 
   - e.g for columns such as weather_1, category such as (Other: NOT ON SCENE) replaced with (Other) same applies to (Other: NOT AT SCENE) 
   - the target colums cateogies are reduced from 5 to 2 i.e Mild and Severe.

## Exploratory Data Analysis
 - Bar plot for categorical columns:
   Frequency Distribution

![alt text](https://raw.githubusercontent.com/JmercyA/CAPSTONE-PROJECT-GROUP-3/main/images/barplotforcat_columns.png)

 - Collision severity across time categories: 
   Time category "2:01 pm to 6:00 pm" is the most critical period
  
![alt text](https://github.com/JmercyA/CAPSTONE-PROJECT-GROUP-3/blob/main/images/collision%20severity%20accross%20time%20categories.png?raw=true)

 - Direction of parties involved in accidents

  Across all directions of travel (West, East, North, South), "Mild" is the most common collision severity

![alt text](https://github.com/JmercyA/CAPSTONE-PROJECT-GROUP-3/blob/main/images/Direction%20of%20travel%20for%20party%201%20by%20collision%20severity.png?raw=true)

- Effect of pedestrian action on collision severity

  No Pedestrian Involved dominates the data

![alt text](https://github.com/JmercyA/CAPSTONE-PROJECT-GROUP-3/blob/main/images/Effect%20of%20pedestrian%20actions%20on%20collision%20severity.png?raw=true)

- Distribution of accidents over the year.

  There appears to be a noticeable increase in accidents in the later years (around 2017-2019)

  There's a visible drop in accidents in the most recent years (2020-2023)

![alt text](https://github.com/JmercyA/CAPSTONE-PROJECT-GROUP-3/blob/main/images/Distribution%20of%20accidents%20by%20month,year%20and%20week.png?raw=true)

- weather and collision severity

  While seemingly safe, clear weather can still lead to a high number of collisions

![alt text](https://github.com/JmercyA/CAPSTONE-PROJECT-GROUP-3/blob/main/images/Collision%20severity%20by%20weather%20conditions.png?raw=true)


## Hypothesis testing
Chi-Square Test for Weather and Collision Severity;

-  Weather significantly affects collision severity.

ANOVA for Road Surface and Number of Injuries;

- Road surface conditions affect the number of injuries.



## Modeling

### Models used
 - RandomForest Model
 - Logistic Regression Model
 - SVC model
 - KNeighbours Classifier
 - XGBoost Model

### Defining the target and the features

- Collision_severity is the target columns

### Label Encoding the target column

- Collision_severity values 'Mild' and 'Severe' are encoded to 0 and 1

### Preprocessing Pipeline;

- scaling the data
- Onehot encoding the categorical columns

### Model training
- spliting data into training and test set 80% for training and 20% for testing
- helper function to train and evaluate the models


### metrics used:

|Metric	    |                                               Reason                                          |
|-----------|-----------------------------------------------------------------------------------------------|
|Precission | Important when the cost of false positives is high (e.g., predicting "Severe" incorrectly).   |
|Recall 	  |Important for ensuring minority class detection (e.g., identifying "Severe" collisions).       | 
|F1-Score	  |Balance precision and recall for imbalanced data.                                              |

### Results;

This table summarizes the **Accuracy**, **Precision**, **Recall**, and **F1-Score** for various models used in predicting collision severity. These metrics were chosen to evaluate the models effectively, especially in cases of class imbalance.

| Model                        | Accuracy | Precision (Mild) | Recall (Mild) | F1-Score (Mild) | Precision (Severe) | Recall (Severe) | F1-Score (Severe) |
|------------------------------|----------|------------------|---------------|-----------------|--------------------|-----------------|-------------------|
| Random Forest                 | 0.64     | 0.68             | 0.82          | 0.74            | 0.49               | 0.31            | 0.38              |
| Logistic Regression           | 0.66     | 0.68             | 0.89          | 0.77            | 0.56               | 0.24            | 0.34              |
| Support Vector Classifier     | 0.66     | 0.67             | 0.91          | 0.77            | 0.55               | 0.20            | 0.30              |
| k-Nearest Neighbors           | 0.61     | 0.67             | 0.76          | 0.72            | 0.44               | 0.33            | 0.38              |
| XGBoost                       | 0.66     | 0.69             | 0.87          | 0.77            | 0.55               | 0.29            | 0.38              |


### Feature Importance

- Since both Random Forest and XGBoost are tree-based models, they inherently provide a measure of feature importance based on how much each feature contributes to reducing uncertainty
- After reviewing the top features from both models, we selected the following features for further modeling: 'distance', 'number_injured', 'day_of_week', 'party1_dir_of_travel', 'number_killed', 'type_of_collision', and 'party2_dir_of_travel'. 

### Results after feature importance:

| Model               | Accuracy | Precision (Mild) | Recall (Mild) | F1-Score (Mild) | Precision (Severe) | Recall (Severe) | F1-Score (Severe) |
|---------------------|----------|------------------|---------------|-----------------|--------------------|-----------------|-------------------|
| Logistic Regression | 0.66     | 0.66             | 0.95          | 0.78            | 0.60               | 0.13            | 0.22              |
| XGBoost             | 0.65     | 0.67             | 0.91          | 0.77            | 0.53               | 0.17            | 0.26              |


- Overall, both Logistic Regression and XGBoost perform better with Mild collision predictions but still miss a significant number of Severe collisions. Improvements in identifying Severe cases could involve tuning the models further or using specialized techniques for class imbalance, such as oversampling Severe cases or using weighted loss functions.

### Hyperparameter tuning
 - Gridsearch

 - Logistic Regression - Best Parameters from GridSearchCV: {'model__C': 1, 'model__max_iter': 500, 'model__solver': 'lbfgs'}

 - XGBoost - Best Parameters from GridSearchCV: {'model__learning_rate': 0.01, 'model__max_depth': 3, 'model__n_estimators': 50, 'model__subsample': 0.8}

### Results:

| Model               | Accuracy | Precision (Mild) | Recall (Mild) | F1-Score (Mild) | Precision (Severe) | Recall (Severe) | F1-Score (Severe) |
|---------------------|----------|------------------|---------------|-----------------|--------------------|-----------------|-------------------|
| Logistic Regression | 0.66     | 0.66             | 0.95          | 0.78            | 0.60               | 0.13            | 0.22              |
| XGBoost             | 0.66     | 0.66             | 0.95          | 0.78            | 0.60               | 0.13            | 0.22              |


- both Logistic Regression and XGBoost perform similarly, with good performance on Mild collisions but significant difficulty with Severe collisions. The models could be improved for Severe collision prediction with further tuning, resampling techniques, or alternative modeling approaches.

### Recommendations and Next steps

- Advanced Feature Engineering: Investigate potential feature transformations, interaction terms, and non-linear features that might better represent the underlying patterns in the data. Creating new features or aggregating features could improve model performance.

- try more sophisticated, non-linear methods like tree-based models or ensembles.

- Tuning Regularization: For some models (like neural networks or linear models), experiment with L1/L2 regularization to control overfitting.

- Feature Transformation: Experiment with non-linear transformations of your features (e.g., log transformations, polynomial features, or PCA for dimensionality reduction) to improve how the model learns from the data.

- onclusion Based on EDA.

We have noted that environmental factors have a limited contribution to occurrence of accidents and to the severity. Human behaviour or features influenced by or correlated with human behaviour. e.g. month (seasonality influence on human behaviour), type of collision, pedestrian action and intersection are a major determinant of accident occurrence and accident severity.

- Recommendations based on EDA.

Jurisdictions should therefore invest more in human behaviour change strategies in order to reduce occurrence of accidents and their severity. The may include pedestrian and driver sensitization on road use, fines and penalties on risky behaviours and structuring driver courses and /or refresher courses to emphasise on safe road use.





