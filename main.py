import time
import customtkinter
from customtkinter import ThemeManager
from tkintermapview import TkinterMapView, convert_coordinates_to_city
from PIL import Image
from tkcalendar import Calendar
import pandas
import requests
import json
import itertools
from pgmpy.inference import VariableElimination
from sklearn.tree import DecisionTreeClassifier
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
import matplotlib.pyplot as mp
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


def get_model(month, lat, long, details=3, start_year=2000, end_year=2022):
    """
    Get the weather prediction model based on historical data.

    :param month: String of the month for which to generate the model.
    :param lat: Latitude (integer) of the location.
    :param long: Longitude (integer) of the location.
    :param details: Number of groups to separate values (default is 3).
    :param start_year: First year of the historical data query (default is 2000).
    :param end_year: Last year of the historical data query (default is 2022).

    :return: model - Bayesian Network model for weather prediction.

    :raises ConnectionError: If there is an issue with the API connection.
    :raises Exception: If the Bayesian Network model is not valid.

    Note:
    The function retrieves historical weather data from the Open Meteo API, processes the data,
    and uses it to train a Bayesian Network model for weather prediction.

    Example:
    >>> model = get_model(month="01", lat=40, long=-73, details=4, start_year=2010, end_year=2020)
    """
    global list_quantile, data_table

    print("Generation of dataset...")

    # Initialize an empty DataFrame for storing historical weather data
    data_table = pandas.DataFrame()

    # Define the number of days in each month
    month_number = {
        "01": 31,
        "02": 28,
        "03": 31,
        "04": 30,
        "05": 31,
        "06": 30,
        "07": 31,
        "08": 31,
        "09": 30,
        "10": 31,
        "11": 30,
        "12": 31
    }

    # Iterate over the specified range of years to fetch historical weather data
    for i in range(start_year, end_year + 1):
        start_date = str(i) + "-" + month + "-01"
        end_date = str(i) + "-" + month + "-{}".format(month_number[month])
        aPI_address = "https://archive-api.open-meteo.com/v1/archive?latitude=" + str(lat) + "&longitude=" + str(
            long) + "&start_date=" + start_date + "&end_date=" + end_date + \
                      "&hourly=weather_code,temperature_2m,relativehumidity_2m,cloud_cover,wind_speed_10m"

        # Make a request to the Open Meteo API
        answer = requests.get(aPI_address)

        # Check if the request was successful (status code 200)
        if answer.status_code == 200:
            data = answer.text
            data = json.loads(data)
            d = pandas.DataFrame(data["hourly"])
            data_table = data_table.append(d, ignore_index=True)
        else:
            # Raise a ConnectionError if there is an issue with the API connection
            return ConnectionError

    print("Dataset generated")

    data_table = data_table.drop(columns=['time'])

    # Rename columns for clarity
    data_table = data_table.rename(
        columns={"temperature_2m": "temperature", "relativehumidity_2m": "relativehumidity",
                 "cloud_cover": "cloudcover", "wind_speed_10m": "wind", "weather_code": "weathercode"})

    # Create a temporary dataset to generate quantiles
    df = data_table.drop(columns=['weathercode'])

    # Transform data into a range of values using quantiles
    list_quantile = []
    for cat in df.columns:
        min_val = data_table[cat].min()
        max_val = data_table[cat].max()
        threshold = []
        for d in range(1, details):
            threshold.append(min_val + d * (max_val - min_val) / details)
        list_quantile.append((min_val, *threshold, max_val))
        data_table[cat] = pandas.cut(data_table[cat], bins=[min_val, *threshold, max_val],
                                     labels=list(range(details)))

    data_table = data_table.dropna()

    # Generate all possible values for categorical variables
    encode_dict = {
        'Outlook': tuple(range(details)),
        'Temperature': tuple(range(details)),
        'Humidity': tuple(range(details)),
        'Wind': tuple(range(details))
    }
    values_lists = [list(values) for values in encode_dict.values()]
    all_combinations = list(itertools.product(*values_lists))

    # Separate features (x) and target variable (y)
    x = data_table[['temperature', 'relativehumidity', 'cloudcover', 'wind']]
    y = data_table['weathercode']

    # Create a Decision Tree model using ID3 algorithm
    clf = DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(x.values, y)
    res = clf.predict_proba(all_combinations)
    proba = res.T

    # Create a Bayesian Network model
    weather = BayesianNetwork(
        [('temperature', 'weathercode'), ('relativehumidity', 'weathercode'), ('cloudcover', 'weathercode'),
         ('wind', 'weathercode')])
    cpd = TabularCPD(
        'weathercode',
        data_table['weathercode'].nunique(),
        proba,
        evidence=['temperature', 'relativehumidity', 'cloudcover', 'wind'],
        evidence_card=[data_table['temperature'].nunique(),
                       data_table['relativehumidity'].nunique(),
                       data_table['cloudcover'].nunique(),
                       data_table['wind'].nunique()])
    weather.add_cpds(cpd)

    # Create conditional probability distributions (CPDs) for individual variables
    for feature in ['temperature', 'relativehumidity', 'cloudcover', 'wind']:
        cpd = TabularCPD(
            feature,
            data_table[feature].nunique(),
            [[i] for i in data_table[feature].value_counts(normalize=True).sort_index().tolist()],
            state_names={feature: sorted(data_table[feature].unique().tolist())}
        )
        weather.add_cpds(cpd)

    # Check if the Bayesian Network model is valid
    if weather.check_model():
        return weather
    else:
        # Raise an exception if the model is not valid
        return Exception


def get_weather(weather, temp, hum, cloud, wind):
    """
    Get weather predictions based on a given weather model.

    :param weather: Bayesian Network model for weather prediction.
    :param temp: Temperature in degrees Celsius (float).
    :param hum: Humidity percentage (float).
    :param cloud: Cloud cover percentage (float).
    :param wind: Wind speed in kilometers per hour (float).

    :return: result (dict) : Dictionary of predicted weather probabilities.

    Note:
    The function takes a Bayesian Network weather prediction model, along with current weather
    parameters (temperature, humidity, cloud cover, and wind speed), and returns a dictionary
    containing the predicted probabilities for different weather codes.

    Example:
    >>> model = get_model(month="01", lat=40, long=-73, details=4, start_year=2010, end_year=2020)
    >>> weather_prediction = get_weather(model, temp=20, hum=60, cloud=30, wind=15)
    """
    global list_quantile, data_table

    # Initialize VariableElimination for Bayesian Network inference
    inference = VariableElimination(weather)

    # Prepare a dictionary of current weather parameters
    var = {
        'temperature': temp,
        'relativehumidity': hum,
        'cloudcover': cloud,
        'wind': wind,
    }

    # Convert normal values into a range of values like the dataset
    list_inf = [var['temperature'], var['relativehumidity'], var['cloudcover'], var['wind']]
    list_inf_modified = [[]] * len(list_quantile)

    # Convert each parameter value to the corresponding range based on quantiles
    for i in range(len(list_quantile)):
        list_inf_modified[i] = pandas.cut([list_inf[i]], bins=list_quantile[i],
                                          labels=list(range(len(list_quantile[0]) - 1)))
        list_inf_modified[i] = list_inf_modified[i][0]

        # Remove nan value into min or max
        if list_inf[i] <= min(list_quantile[i]):
            list_inf_modified[i] = 0
        elif list_inf[i] > max(list_quantile[i]):
            list_inf_modified[i] = len(list_quantile[0]) - 2

    # Update the variable dictionary with modified values
    var = {
        'temperature': list_inf_modified[0],
        'relativehumidity': list_inf_modified[1],
        'cloudcover': list_inf_modified[2],
        'wind': list_inf_modified[3],
    }

    # Perform Bayesian Network inference to get predicted weather probabilities
    predicted = inference.query(variables=['weathercode'], evidence=var)
    key = sorted(data_table['weathercode'].unique().tolist())
    value = predicted.values.tolist()

    # Create a dictionary of predicted weather probabilities
    result = dict(zip(key, value))

    return result


def test_model(model, step=10):
    """
    Test the accuracy of a weather prediction model.

    This function tests the accuracy of a given weather prediction model by iterating through the dataset,
    making predictions, and comparing them with the actual weather codes. The test is performed in steps.

    :param model: The weather prediction model (Bayesian Network) to be tested.
    :param step: The step size for iterating through the dataset (default is 10).
    :return: A dictionary containing the counts of different types of test results.
             Keys represent the result categories ('Exact', 'In propositions', 'Not corresponding').
    """
    global data_table

    # List to store the results of each test
    total = []

    # Iterate through the dataset in steps
    for i in range(0, len(data_table), step):
        print(f"{i} / {len(data_table)} ({round(i * 100 / len(data_table))}%)")

        # Extract the test data for the current iteration
        test = data_table.iloc[i]

        # Get weather predictions using the model
        result = get_weather(model, test['temperature'], test['relativehumidity'], test['cloudcover'], test['wind'])

        # Remove results with value 0
        updated_result = {key: value for key, value in result.items() if value != 0}

        # Find the maximum result code
        max_key = max(updated_result, key=lambda k: updated_result[k])

        # Check the accuracy of the prediction and update the result list
        if max_key == test['weathercode']:
            total.append("Exact")
        elif test['weathercode'] in updated_result:
            total.append("In propositions")
        else:
            total.append("Not corresponding")

    # Convert the result list to a pandas Series and get value counts
    series = pandas.Series(total)
    result_dict = series.value_counts().to_dict()

    return result_dict


customtkinter.set_default_color_theme("green")


class App(customtkinter.CTk):
    """
    CustomTkinter-based GUI application for weather prediction.

    Attributes:
        APP_NAME (str): The name of the application.
        WIDTH (int): The width of the application window.
        HEIGHT (int): The height of the application window.
    """
    APP_NAME = "Weather prediction"
    WIDTH = 800
    HEIGHT = 600

    def __init__(self, *args, **kwargs):
        """
         Initialize the App class.

         :param args: Additional arguments for the parent class.
         :param kwargs: Additional keyword arguments for the parent class.
         """
        super().__init__(*args, **kwargs)

        self.title(App.APP_NAME)
        self.geometry(str(App.WIDTH) + "x" + str(App.HEIGHT))
        self.minsize(App.WIDTH, App.HEIGHT)
        self.iconbitmap("img/icon.ico")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.bind("<Command-q>", self.on_closing)
        self.bind("<Command-w>", self.on_closing)
        self.createcommand('tk::mac::Quit', self.on_closing)

        self.marker = None
        self.lat = 0
        self.long = 0
        self.lat_mem = 0
        self.long_mem = 0

        self.temp_pos = (0, 0)  # double click on map init

        self.model = None
        self.month = None
        self.result = None
        self.result_test = {"Exact":0,
                            "In propositions": 0,
                            "Not corresponding": 0}

        # ============ create two CTkFrames ============

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        self.frame_top = customtkinter.CTkFrame(master=self, width=150, corner_radius=0, fg_color=None)
        self.frame_top.grid(row=0, column=1, rowspan=1, columnspan=3, padx=0, pady=0, sticky="nsew")

        self.frame_left = customtkinter.CTkFrame(master=self, width=150, corner_radius=0, fg_color=None)
        self.frame_left.grid(row=1, column=0, rowspan=2, padx=0, pady=0, sticky="nsew")

        self.frame_right = customtkinter.CTkFrame(master=self, corner_radius=0)
        self.frame_right.grid(row=1, column=1, rowspan=2, pady=0, padx=0, sticky="new")

        # ============ frame_left ============

        self.button_test = customtkinter.CTkButton(master=self.frame_left,
                                                text="Test model",
                                                width=90,
                                                command=self.run_test)
        self.button_test.pack(side='top')
        self.label_test_model_string = customtkinter.StringVar(value="")
        self.label_test_model = customtkinter.CTkButton(self.frame_left, textvariable=self.label_test_model_string,
                                                        state='disabled', command=self.get_pie_test,
                                                        fg_color='transparent')
        self.label_test_model.pack(side='top')

        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.frame_left,
                                                                       values=["Dark", "Light", "System"],
                                                                       command=customtkinter.set_appearance_mode)
        self.appearance_mode_optionemenu.pack(side='bottom', padx=(20, 20), pady=3)

        self.appearance_mode_label = customtkinter.CTkLabel(self.frame_left, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.pack(side='bottom', padx=(20, 20))

        self.map_option_menu = customtkinter.CTkOptionMenu(self.frame_left, values=["Google normal", "Google satellite",
                                                                                    "OpenStreetMap"],
                                                           command=self.change_map)
        self.map_option_menu.pack(side='bottom', padx=(20, 20), pady=3)

        self.map_label = customtkinter.CTkLabel(self.frame_left, text="Type of map:", anchor="w")
        self.map_label.pack(side='bottom', padx=(20, 20))

        self.cal = Calendar(self.frame_left, selectmode='day', locale='en_US', disabledforeground='red',
                            cursor="hand2", background=ThemeManager.theme["CTkFrame"]["fg_color"][1],
                            selectbackground=ThemeManager.theme["CTkButton"]["fg_color"][1])
        self.cal.pack(side="bottom", expand=True, padx=10, pady=10)

        # ============ frame_right ============

        self.frame_right.grid_columnconfigure(0, weight=1)
        self.frame_right.grid_columnconfigure(1, weight=0)
        self.frame_right.grid_columnconfigure(2, weight=0)

        self.button_1 = customtkinter.CTkButton(master=self.frame_right,
                                                text="Set Marker",
                                                command=self.set_marker_event)
        self.button_1.grid(row=0, column=2, sticky="we", padx=(12, 0), pady=12)

        self.map_widget = TkinterMapView(self.frame_right, width=850, height=600, corner_radius=0)
        self.map_widget.grid(row=1, rowspan=1, column=0, columnspan=3, sticky="nwe", padx=(0, 0), pady=(0, 0))

        self.map_widget.add_left_click_map_command(self.left_click_pos)
        self.map_widget.add_right_click_menu_command("Set marker", self.set_marker_pos, pass_coords=True)
        self.entry = customtkinter.CTkEntry(master=self.frame_right,
                                            placeholder_text="Type address")
        self.entry.grid(row=0, column=0, sticky="we", padx=(12, 0), pady=12)
        self.entry.bind("<Return>", self.search_event)

        self.button_5 = customtkinter.CTkButton(master=self.frame_right,
                                                text="Search",
                                                width=90,
                                                command=self.search_event)
        self.button_5.grid(row=0, column=1, sticky="w", padx=(12, 0), pady=12)

        # ============ TOP ============
        self.frame_info = customtkinter.CTkFrame(master=self, corner_radius=0)
        self.frame_info.grid(row=0, column=0, sticky="nsew")

        self.label_title = customtkinter.CTkLabel(self.frame_info, text="Weather prediction:", anchor="w")
        self.label_title.pack(side='top', pady=5)

        self.image = customtkinter.CTkImage(light_image=Image.open("img/n-a.png"),
                                            dark_image=Image.open("img/n-a.png"),
                                            size=(100, 100))
        self.image_button = customtkinter.CTkButton(self.frame_info, image=self.image, text="", fg_color="transparent",
                                                    state="disabled", command=self.view_pie)
        self.image_button.pack(side='top')
        self.label_info_weather_string = customtkinter.StringVar(value="")
        self.label_info_weather = customtkinter.CTkLabel(self.frame_info, textvariable=self.label_info_weather_string,
                                                         anchor="w")
        self.label_info_weather.pack(side='top')

        self.label_info_string = customtkinter.StringVar(value="Parameters selected: /")
        self.label_info = customtkinter.CTkLabel(self.frame_top, textvariable=self.label_info_string, anchor="w")
        self.label_info.pack(side='top', pady=25)

        # ============ PARAMETERS ============

        self.frame_parameter = customtkinter.CTkFrame(master=self.frame_top, corner_radius=0, fg_color='transparent')
        self.frame_parameter.pack(side='top', padx=(20, 20))

        self.label_temp = customtkinter.CTkLabel(master=self.frame_parameter, text="Temperature (°C)", anchor="w")
        self.label_temp.grid(row=0, column=0, padx=5)
        self.entry_temp = customtkinter.CTkEntry(master=self.frame_parameter, placeholder_text="°C", width=45)
        self.entry_temp.grid(row=0, column=1)
        self.entry_temp.bind("<Return>", self.run)

        self.label_hum = customtkinter.CTkLabel(master=self.frame_parameter, text="Humidity (%)", anchor="w")
        self.label_hum.grid(row=0, column=2, padx=5)
        self.entry_hum = customtkinter.CTkEntry(master=self.frame_parameter, placeholder_text="%", width=45)
        self.entry_hum.grid(row=0, column=3)
        self.entry_hum.bind("<Return>", self.run)

        self.label_cloud = customtkinter.CTkLabel(master=self.frame_parameter, text="Cloud cover (%)", anchor="w")
        self.label_cloud.grid(row=0, column=4, padx=5)
        self.entry_cloud = customtkinter.CTkEntry(master=self.frame_parameter, placeholder_text="%", width=45)
        self.entry_cloud.grid(row=0, column=5)
        self.entry_cloud.bind("<Return>", self.run)

        self.label_wind = customtkinter.CTkLabel(master=self.frame_parameter, text="Wind (km/h)", anchor="w")
        self.label_wind.grid(row=0, column=6, padx=5)
        self.entry_wind = customtkinter.CTkEntry(master=self.frame_parameter, placeholder_text="km/h", width=45)
        self.entry_wind.grid(row=0, column=7)
        self.entry_wind.bind("<Return>", self.run)

        self.button_try = customtkinter.CTkButton(self.frame_top, text="Run", command=self.run)
        self.button_try.pack(side='top', pady=20)

        # Set default values

        self.change_map("Google normal")
        self.map_widget.set_address("Riga")
        self.set_marker_event()
        customtkinter.set_appearance_mode("Dark")

    def search_event(self, event=None):
        """
        Handles the search event for the map.

        :param event: The event triggering the search (default is None).
        """
        self.map_widget.set_address(self.entry.get())
        self.set_marker_event()

    def set_marker_event(self):
        """
        Sets a marker on the map.
        """
        self.clear_marker_event()
        current_position = self.map_widget.get_position()
        self.marker = self.map_widget.set_marker(current_position[0], current_position[1])
        self.lat = current_position[0]
        self.long = current_position[1]

    def left_click_pos(self, coords):
        """
        Handles left-click events on the map.

        :param coords: Coordinates of the left-click position.
        """
        if self.temp_pos == coords:  # verify double click before set a marker
            self.set_marker_pos(coords)
        self.temp_pos = coords

    def set_marker_pos(self, coords):
        """
         Sets a marker at the specified coordinates.

         :param coords: Coordinates for placing the marker.
         """
        self.clear_marker_event()
        self.marker = self.map_widget.set_marker(coords[0], coords[1])
        self.lat = coords[0]
        self.long = coords[1]

    def clear_marker_event(self):
        """
        Clears marker on the map.
        """
        if self.marker is not None:
            self.marker.delete()

    def change_map(self, new_map: str):
        """
        Changes the map type based on user input.

        :param new_map: The new type of map selected by the user.
        """
        if new_map == "OpenStreetMap":
            self.map_widget.set_tile_server("https://a.tile.openstreetmap.org/{z}/{x}/{y}.png")
        elif new_map == "Google normal":
            self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}&s=Ga",
                                            max_zoom=22)
        elif new_map == "Google satellite":
            self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga",
                                            max_zoom=22)

    def on_closing(self, event=0):
        """
        Handles the closing event of the application.

        :param event: The event triggering the closing (default is 0).
        """
        self.destroy()

    def start(self):
        """
        Starts the application's main loop.
        """
        self.mainloop()

    def set_image(self, name):
        """
        Sets the image for the application.

        :param name: The name of the image file.
        """
        self.image = customtkinter.CTkImage(light_image=Image.open("img/" + name),
                                            dark_image=Image.open("img/" + name),
                                            size=(100, 100))
        self.image_button.configure(image=self.image, state='normal')

    def view_pie(self):
        """
        Displays a pie chart of the weather prediction results.
        """
        mp.pie(self.result.values(), labels=self.result.keys(), autopct='%1.1f%%', startangle=90)
        mp.title("Details of results")
        mp.show()

    def check_variable(self):
        """
        Checks if the input variables are valid.

        :return: True if variables are valid, False otherwise.
        """
        def is_number(a):
            try:
                float(a)
                return True
            except ValueError:
                return False

        # Check if all input variables are numbers
        if is_number(self.entry_temp.get()) and is_number(self.entry_hum.get()) and is_number(
                self.entry_wind.get()) and is_number(self.entry_cloud.get()):
            # Check if humidity is within the valid range (0 to 100)
            # Check if cloud cover is within the valid range (0 to 100)
            # Check if wind speed is greater than or equal to 0
            if 0 <= float(self.entry_hum.get()) <= 100 and 0 <= float(self.entry_cloud.get()) <= 100 and float(
                    self.entry_wind.get()) >= 0:
                return True  # All conditions are met, variables are valid
        return False  # At least one condition is not met, variables are not valid

    def generate_model(self):
        """
        Generate a weather prediction model based on selected parameters.
        """
        # Extract the month from the calendar date
        month = self.cal.get_date().split("/")[0]
        if len(month) == 1:
            month = "0" + month

        # Check if the month or coordinates have changed, and update the model
        if month != self.month or self.lat_mem != self.lat or self.long_mem != self.long:
            self.month = month
            self.lat_mem = self.lat
            self.long_mem = self.long
            s = time.time()
            self.model = get_model(self.month, self.lat, self.long)
            e = time.time()
            print("Model done in {}s".format(round(e - s)))

    def run(self, event=None):
        """
        Runs the weather prediction based on user input.

        :param event: The event triggering the weather prediction (default is None).
        """
        # Check if input variables are valid
        if self.check_variable():

            self.generate_model()  # Generate the model

            # If a model is available, proceed with weather prediction
            if self.model is not None:
                # Get the city name from coordinates
                city = convert_coordinates_to_city(self.lat, self.long)
                if city is None:
                    city = ""
                # Create a string with selected parameters
                parameter = "Parameters selected: " + city + " T: " \
                            + self.entry_temp.get() + "°C -  H: " \
                            + self.entry_hum.get() + "% - C: " \
                            + self.entry_cloud.get() + "% W: " \
                            + self.entry_wind.get() + "km/h"
                # Update the info label with the selected parameters
                self.label_info_string.set(parameter)

                # Get weather prediction results
                result = get_weather(self.model, float(self.entry_temp.get()), float(self.entry_hum.get()),
                                     float(self.entry_cloud.get()), float(self.entry_wind.get()))

                # Load details and icons from JSON files
                with open("weathercodes_detail.json") as F:
                    details = json.load(F)
                with open("weathercodes_icon.json") as F:
                    icons = json.load(F)

                # Remove results with value 0
                updated_result = {key: value for key, value in result.items() if value != 0}
                # Map result codes to detailed descriptions
                self.result = {details[str(key)]: value for key, value in updated_result.items()}

                # Find the maximum result code and update weather info label
                max_key = max(updated_result, key=lambda k: updated_result[k])
                self.label_info_weather_string.set(
                    details[str(max_key)] + " ({}%)".format(round(updated_result[max_key] * 100)))

                # Set the image based on the maximum result code
                self.set_image(icons[str(max_key)])
            else:
                self.label_info_string.set("No model yet run")
        else:
            self.label_info_string.set("Parameters invalid")

    def run_test(self):
        """
        Run a test on the generated model and update the test results in the GUI.
        """
        # Generate a model
        self.generate_model()

        # Perform a test on the model
        tested_model = test_model(self.model)
        for i in ["Exact", "In propositions", "Not corresponding"]:
            if i in tested_model:
                self.result_test[i] = tested_model[i]
            else:
                self.result_test[i] = 0

        # Calculate the number of correct predictions and total predictions
        good = self.result_test["Exact"] + self.result_test["In propositions"]
        total = sum(self.result_test.values())

        # Update the GUI labels with the test results
        self.label_test_model_string.set(f"Good predictions: {round(good * 100 / total)}%")
        self.label_test_model.configure(state='normal')

    def get_pie_test(self):
        """
        Display a pie chart of the test results.
        """
        mp.pie(self.result_test.values(), labels=self.result_test.keys(), autopct='%1.1f%%', startangle=90)
        mp.title("Details of results")
        mp.show()


if __name__ == "__main__":
    app = App()
    app.start()
