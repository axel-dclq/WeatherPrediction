# Weather Prediction GUI Application
## Overview
This GUI application allows users to predict weather conditions based on a Bayesian Network model trained on historical weather data. The application provides an interactive map interface, where users can set a location marker for weather prediction. Users can input specific weather parameters such as temperature, humidity, cloud cover, and wind speed to get predictions for different weather codes.

## Prerequisites
Make sure you have the required dependencies installed. You can install them using the following:

```bash
pip install tk customtkinter tkcalendar pillow pandas requests pgmpy scikit-learn numpy matplotlib
```

## Getting Started
To run the application, execute the following command in your terminal:

```bash
python main.py
```

## User Interface
The application consists of the following components:

### Map Interface
Users can interact with the map to set a location marker.
- Three map types are available: Google normal, Google satellite, and OpenStreetMap. Users can switch between these options using the "Type of map" dropdown.
- The "Set Marker" button allows users to manually set a marker on the map.
- A double left click or right click and "Set marker" also allows you to set a marker on the map.
- The search bar enables users to search for a specific location by address.
### Date Selection
- Users can select a date using the calendar widget. The historical weather data for the selected month will be used to train the prediction model.
### Weather Parameters
- Users can input weather parameters such as temperature, humidity, cloud cover, and wind speed.
- The "Run" button initiates the weather prediction based on the provided parameters.
### Weather Prediction Results
- The application displays the predicted weather code along with a percentage indicating the likelihood of the prediction.
- The "View Pie" button on the image opens a pie chart showing the distribution of weather predictions.
###Testing the Model
The Weather Prediction App includes a testing functionality to evaluate the accuracy of the weather prediction model.
1. Click the "Test Model" Button: By clicking this button, you initiate the process of testing the model.

2. Observe the Result: Once the test is performed, the application displays the percentage of accurate predictions compared to the total number of tests conducted. This information is updated in real-time.

3. Check the Details: For more detailed insights into the test results, you can click on the text to show a pie chart. This chart illustrates the distribution of exact predictions, predictions present in the propositions, and predictions that did not correspond.

4. Test Anytime: You can press the "Test Model" button at any time, even after modifying the date or moving the marker on the map. This allows you to see how the model performs with different data.


### Appearance Options
- Users can switch between dark, light and system appearance modes.
- Users can choose the appearance mode and map type according to their preferences.

## How It Works
1. Map Interface: Users interact with the map interface to set a location marker. User also selects a date to set the search month. The application provides various map options for visualization.

2. Weather Prediction: Users input weather parameters, and click on "Run".

3. Data Generation: Historical weather data is fetched from the Open Meteo API for the specified location and month. This data is processed to create a dataset for training the Bayesian Network model.

4. Model Training: The Bayesian Network model is trained using the historical weather data. The model takes into account temperature, humidity, cloud cover, and wind speed to predict weather codes.

5. Results Visualization: The predicted weather code and its likelihood are displayed. Users can view a pie chart for a detailed distribution of weather predictions.


## How the generation works
#### 1. Dataset Generation
The algorithm starts by fetching historical weather data from the Open Meteo API. It specifies the latitude (lat), longitude (long), month, and date range for which the weather data is required. The API provides hourly data for parameters such as temperature, humidity, cloud cover, and wind speed. The fetched data is then organized into a structured dataset.

#### 2. Data Preprocessing
The retrieved dataset undergoes preprocessing to ensure it is suitable for training a predictive model. This involves handling missing data, removing unnecessary columns, and preparing the data for transformation.

#### 3. Quantile Calculation
Quantiles are computed for each weather variable in the dataset. Quantiles divide the range of values into discrete groups, facilitating the subsequent training of the Decision Tree model. These quantiles are then used to discretize continuous data, transforming it into categorical values.

#### 4. Model Training
A Decision Tree model is trained using the ID3 (Iterative Dichotomiser 3) algorithm. The Decision Tree learns patterns and relationships within the historical weather data, making it capable of predicting weather conditions based on input features such as temperature, humidity, cloud cover, and wind speed. A function that computes all the possible combinaison possible depending and how our dataset is built and how the variables are splitted is implemented. Then, we can calculate the probability of each combinaison and use it in the Bayesian Network.

#### 5. Bayesian Network Creation
The trained Decision Tree serves as the foundation for constructing a Bayesian Network model. Conditional Probability Distributions (CPDs) are defined for each weather variable, establishing the relationships between them. The Bayesian Network encapsulates probabilistic dependencies and interactions among the weather features.

#### 6. Model Validation
The algorithm checks the validity of the Bayesian Network model. This ensures that the constructed model is consistent with the principles of Bayesian Networks and is suitable for making reliable weather predictions.

## Notes
The application may take some time to generate the model based on the historical data.
Ensure that your system has an active internet connection for fetching historical weather data.

## Authors
- Mathéo BEGIS
- Axel DECLERCQ
