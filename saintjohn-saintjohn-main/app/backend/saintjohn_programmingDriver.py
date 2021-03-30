from flask import Flask, request
from flask_cors import CORS
from flask_marshmallow import Marshmallow
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from math import trunc
import os
import json
import random

app = Flask(__name__)
CORS(app)
ma = Marshmallow(app)

penatly_value_array = []
plant_production_rate_array = []
insentive_rate_array = []

class IncentiveRates():
    def __init__(self, name, rate):
        self.name = name
        self.rate = rate


class IncentiveRatesSchema(ma.Schema):
    class Meta:
        fields = ('name', 'rate')


insentive_rate_schema = IncentiveRatesSchema()
insentive_rates_schema = IncentiveRatesSchema(many=True)


class PenatlyValues():
    def __init__(self, name, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11):
        self.name = name
        self.z1 = z1
        self.z2 = z2
        self.z3 = z3
        self.z4 = z4
        self.z5 = z5
        self.z6 = z6
        self.z7 = z7
        self.z8 = z8
        self.z9 = z9
        self.z10 = z10
        self.z11 = z11
        self.z11 = z11


class PenatlyValuesSchema(ma.Schema):
    class Meta:
        fields = ('name', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11')


penatly_value_schema = PenatlyValuesSchema()
penatly_values_schema = PenatlyValuesSchema(many=True)


class PlantProductionRates():
    def __init__(self, name, thermal, nuclear, combustion_turbine, hydro, wind):
        self.name = name
        self.thermal = thermal
        self.nuclear = nuclear
        self.combustion_turbine = combustion_turbine
        self.hydro = hydro
        self.wind = wind


class PlantProductionRatesSchema(ma.Schema):
    class Meta:
        fields = ('name', 'thermal', 'nuclear', 'combustion_turbine', 'hydro', 'wind')


plant_production_rate_schema = PlantProductionRatesSchema()
plant_production_rates_schema = PlantProductionRatesSchema(many=True)


@app.route('/plant_production_rate')
def get_all_plant_production_rates():
    plant_production_rates = parse_plant_production_rate_data_from_file()


def parse_plant_production_rate_data_from_file():
    list_of_plant_production_rates = []
    data_file = open("../../Information/PlantProductionRates.csv", "r")
    lines_from_file = data_file.readlines()

    for i in range(1, len(lines_from_file)):
        plant_production_rate = get_plant_production_rate_data_from_line(lines_from_file[i], i)
        list_of_plant_production_rates.append(plant_production_rate)

    return list_of_plant_production_rates


def get_plant_production_rate_data_from_line(line, number):
    plant_production_rate_data = line.split(',')

    plant_production_rate_name = 'Z' + str(number)
    plant_production_rate_thermal = int(plant_production_rate_data[0])
    plant_production_rate_nuclear = int(plant_production_rate_data[1])
    plant_production_rate_combustion_turbine = int(plant_production_rate_data[2])
    plant_production_rate_hydro = int(plant_production_rate_data[3])
    plant_production_rate_wind = int(plant_production_rate_data[4])

    plant_production_rate = PlantProductionRates(plant_production_rate_name, plant_production_rate_thermal,
                                                 plant_production_rate_nuclear,
                                                 plant_production_rate_combustion_turbine, plant_production_rate_hydro,
                                                 plant_production_rate_wind)
    convert_plant_production_values_to_kwh(plant_production_rate)

    plant_production_rate_array.append(plant_production_rate)

    return plant_production_rate


def convert_plant_production_values_to_kwh(plant_production_rate):
    plant_production_rate.thermal = plant_production_rate.thermal * 60 * 60 * 24 * 30 * 0.2778  # MJ/s -> KWh
    plant_production_rate.nuclear = plant_production_rate.nuclear * 60 * 60 * 24 * 30 * 0.2778  # MJ/s -> KWh
    plant_production_rate.combustion_turbine = plant_production_rate.combustion_turbine * 60 * 60 * 24 * 30 * 0.2778  # MJ/s -> KWh
    plant_production_rate.hydro = plant_production_rate.hydro * 60 * 60 * 24 * 30 * 0.2778  # MJ/s -> KWh
    plant_production_rate.wind = plant_production_rate.wind * 60 * 60 * 24 * 30 * 0.2778  # MJ/s -> KWh


@app.route('/penatly_values')
def get_all_penatly_values():
    penatly_values = parse_penatly_value_data_from_file()


def parse_penatly_value_data_from_file():
    list_of_penatly_values = []
    data_file = open("../../Information/PenaltyValues.csv", "r")
    lines_from_file = data_file.readlines()

    for i in range(1, len(lines_from_file)):
        penatly_value = get_penatly_value_data_from_line(lines_from_file[i], i)
        list_of_penatly_values.append(penatly_value)

    return list_of_penatly_values


def get_penatly_value_data_from_line(line, number):
    penatly_value_data = line.split(',')

    penatly_value_name = 'Z' + str(number)

    penatly_value_z1 = float(penatly_value_data[0])
    penatly_value_z2 = float(penatly_value_data[1])
    penatly_value_z3 = float(penatly_value_data[2])
    penatly_value_z4 = float(penatly_value_data[3])
    penatly_value_z5 = float(penatly_value_data[4])
    penatly_value_z6 = float(penatly_value_data[5])
    penatly_value_z7 = float(penatly_value_data[6])
    penatly_value_z8 = float(penatly_value_data[7])
    penatly_value_z9 = float(penatly_value_data[8])
    penatly_value_z10 = float(penatly_value_data[9])
    penatly_value_z11 = float(penatly_value_data[10])

    penatly_value = PenatlyValues(penatly_value_name, penatly_value_z1, penatly_value_z2, penatly_value_z3,
                                    penatly_value_z4, penatly_value_z5, penatly_value_z6, penatly_value_z7,
                                    penatly_value_z8, penatly_value_z9, penatly_value_z10, penatly_value_z11)

    penatly_value_array.append(penatly_value)

    return penatly_value


@app.route('/insentive_rate')
def get_all_insentive_rates():
    insentive_rates = parse_insentive_rate_data_from_file()


def parse_insentive_rate_data_from_file():
    list_of_insentive_rates = []
    data_file = open("../../Information/IncentiveRates.csv", "r")
    lines_from_file = data_file.readlines()

    for i in range(1, len(lines_from_file)):
        insentive_rate = get_insentive_rate_data_from_line(lines_from_file[i], i)
        list_of_insentive_rates.append(insentive_rate)

    return list_of_insentive_rates


def get_insentive_rate_data_from_line(line, number):
    insentive_rate_data = line.split(',')
    if number == 0:
        insentive_rate_name = 'emission_tax'
    else:
        insentive_rate_name = 'non_emission_incentive'
    insentive_rate_rate = float(insentive_rate_data[0])
    insentive_rate = IncentiveRates(insentive_rate_name, insentive_rate_rate)

    insentive_rate_array.append(insentive_rate)

    return insentive_rate


# predicted power forecast for each zone in the province, for each month of the year
def output_file_level_1(filename, data):  # data -> [[1,2,..], [1, 2, ...]] and all units are GWh
    f = open(filename, "a")
    for i in range(0, 7):
        for j in range(0, 11):
            if j != 0:
                print(",", file=f, end="")
            print(data[j][i], file=f, end="")
        print("\n", file=f, end="")

    f.close()


# predicted power forecast for each zone in the province, for each month of the year
def output_file_level_2(filename, data):  # data -> [[1,2,..], [1, 2, ...]]
    f = open(filename, "a")
    print("Format -> Month Number, Cost, Total Consumed Power (GWh), Renewable Power Used (%)")
    for i in range(0, 11):
        print(str(i) + "," + data[i][0] + "," + data[i][1] + "," + data[i][2], file=f)
    f.close()


def get_forecasted_data(year):
    if year not in [2019, 2020, 2021, 2022]:
        return json.dump({'Error': f'No forecast available for {year}'})

    data_file = open(f'../../Forecast/NBTrend{year}_Forecast.csv', 'r')

    forecast_dict = {f'Zone {i + 1}': [] for i in range(7)}
    for line in data_file.readlines():
        values = [float(val) for val in line.split(',')]
        for i in range(len(values)):
            forecast_dict[f'Zone {i+1}'].append(values[i])

    return json.dumps({'year': year, 'forecast': forecast_dict})



##################################################################################
# Machine Learning Code
##################################################################################
scaler = MinMaxScaler(feature_range=(-1, 1))


def load_datasets():
    years = [2015, 2016, 2017, 2018]

    dataset = None
    for year in years:
        file_path = f'../../PastYearData/NBTrend{year}.csv'
        yearly_dataset = pd.read_csv(file_path, header=None)
        if dataset is None:
            dataset = yearly_dataset.copy()
        else:
            dataset = dataset.append(yearly_dataset, ignore_index=True)

    test_dataset_size = 0
    dataset = scaler.fit_transform(dataset)
    if test_dataset_size != 0:
        train_dataset = dataset[:-test_dataset_size].astype(float)
        test_dataset = dataset[-test_dataset_size:].astype(float)
    else:
        train_dataset = dataset
        test_dataset = None
    return train_dataset, test_dataset


class RecurrentNN(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=10, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.dense = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        output, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 7, -1))
        predictions = self.dense(output)
        return predictions[-1]


def create_model_input(dataset, train_window=4):
    dataset = torch.FloatTensor(dataset)
    inout_seq = create_inout_sequences(dataset, train_window)
    return inout_seq


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def train(train_data, model, epochs=2048, lr=0.001):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(model)

    train_inout_seq = create_model_input(train_data)

    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            output = model(seq)

            mse_loss = loss_function(output, labels)
            mse_loss.backward()
            optimizer.step()

        if i % 16 == 1:
            print(f'Epoch #{i:3}\tMSE Loss: {mse_loss.item():.5f}')

    print(f'Epoch #{i:3}\tMSE Loss: {mse_loss.item():.5f}')


def test(test_data, model):
    model.eval()
    loss_function = nn.MSELoss()

    test_inout_seq = create_model_input(test_data)
    actual_predictions = []
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        for seq, labels in test_inout_seq:
            output = model(seq)
            actual_predictions.append(scaler.inverse_transform(output.numpy().T))

            mse_loss = loss_function(output.T, labels)

        print(f'Test MSE Loss: {mse_loss.item():.5f}')
    return actual_predictions


def predict(input_seq):
    assert input_seq.shape == (4, 7)

    normalized_seq = scaler.transform(input_seq)
    rnn = torch.load('recurrent-neural-network.model')
    rnn.eval()
    with torch.no_grad():
        rnn.hidden_cell = (torch.zeros(1, 1, rnn.hidden_layer_size),
                           torch.zeros(1, 1, rnn.hidden_layer_size))
        output = rnn(torch.FloatTensor(normalized_seq))
    return scaler.inverse_transform(output.numpy().T)[0]


def forecast_through_2022(train_data, window_size):
    seq = scaler.inverse_transform(train_data[-window_size:])
    forecast = []
    for month in range(8 + 4 * 12):
        prediction = predict(seq)
        forecast.append([trunc(x*100)/100 for x in prediction])
        seq = np.append(seq, prediction.reshape(1, 7), axis=0)[1:]
    forecast_df = pd.DataFrame(forecast)
    if not os.path.exists(f'Forecast/WindowSize{window_size}'):
        os.makedirs(f'Forecast/WindowSize{window_size}')
    forecast_df.to_csv(f'Forecast/WindowSize{window_size}/NBTrend2019-2022_Forecast.csv', header=False, index=False)

def run_ml():
    for i in range(8, 13):
      window_size = i
      train_data, test_data = load_datasets()
      model = RecurrentNN()
      train(train_data, model, epochs=1024, lr=0.0005)
      torch.save(model, 'recurrent-neural-network.model')
      forecast_through_2022(train_data, window_size)

##################################################################################
# Genetic Algorithm Code
##################################################################################

def get_total_provincial_cost():
  power_distribution = get_power_distribution_between_zones()
  print(power_distribution)
  # return sum of cost of each province

def get_power_distribution_between_zones():
  population_size = 10
  number_of_generations = 10
  number_of_zones = 7

  print('> starting gen_0 generation...')
  gen_0 = generate_generation_zero(population_size, number_of_zones)
  print('< finished gen_0 generation')
  print('> starting evolution...')
  results = run_evolution(gen_0, number_of_generations, population_size)
  print('< finished evolution')

  return get_best_result(results)

def generate_generation_zero(population_size, number_of_zones):
  # gen_0: list of 10 possible solutions (gnomes)
  gen_0 = []
  for i in range(0, population_size):
    new_gnome = []
    for j in range(0, 11):
      new_percentages = get_random_percentages(number_of_zones)
      new_gnome.append(new_percentages)
    gen_0.append(new_gnome)
  
  return gen_0

def get_random_percentages(number_of_percentages):
  random_numbers = random.sample(range(0, 100), number_of_percentages)
  return calculate_percentages(random_numbers)
  

def calculate_percentages(list_of_numbers):
  sum_of_numbers = sum(list_of_numbers)
  percentages = list(map(lambda num: num/sum_of_numbers, list_of_numbers))
  rounded_percentages = list(map(lambda num: round(num, 2), percentages))
  return rounded_percentages

def calculate_fitness_for_each_gnome(population):
  fitness_values = []

  for solution in population:
    fitness_value = calculate_cost_of_solution(solution)
    fitness_values.append(fitness_value)
  return fitness_values

# NOTE: this should have the actual amount of power needed for each zone. This will be the projected amount

# real data: (getting from ML output)


# example data:
amount_required = [1000, 2000, 1000, 3000, 2000, 1500, 1000, 2000, 1000, 3000, 2000]

# real data:
cost_between_zones = [[], []]


# NOTE: this is going to cause a problem...
# it will try to calculate this for all 11 zones but we only want it doing it for the first 7, the ones in NB
# maybe make it so that if the amount needed is 0, do nothing?

def calculate_cost_of_solution(solution):
  year_data = json.loads(get_forecasted_data(2022))
  forecast = year_data['forecast']

  print(len(penatly_value_array))
  print(len(penatly_value_array[0]))

  print(len(solution))
  print(len(solution[0]))

  total_cost_of_solution = 0
  for row in range(0, len(solution)):
    for col in range(0, len(solution[row])):
      # 1. multiply percentage in each zone against the projected values for that zone (zone=current row in table) => amount taken from each zone
      percentage_from_zone = solution[row][col]
      # 2. multiply the amount from each zone by the cost to transfer from that zone => cost to transfer from each zone
      # amount_from_zone = percentage_from_zone * amount_required[row, col]
      amount_from_zone = percentage_from_zone * forecast[f'Zone {row + 1}'][col] # "it should work" - Nathan
      # 3. sum costs for each column => cost for that zone
      cost_of_amount = amount_from_zone * penatly_value_array[row][col]
      # 4. sum costs for each row (zone) => total provincial cost
      total_cost_of_solution += cost_of_amount

def run_evolution(previous_generation, number_of_generations, population_size):
  if number_of_generations == 0:
    return previous_generation
  
  next_generation = generate_next_generation(previous_generation, population_size)

  resulting_solutions = run_evolution(next_generation, number_of_generations-1, population_size)

  return resulting_solutions


def generate_next_generation(previous_generation, population_size):
  next_generation = []
  
  fitness_values = calculate_fitness_for_each_gnome(previous_generation)

  # save the best two for the next generation
  min_1, min_2 = get_two_best_from_generation(previous_generation, fitness_values)
  next_generation.append(min_1)
  next_generation.append(min_2)

  next_generation = run_crossover_on_gnomes(next_generation, previous_generation, fitness_values, population_size)

  # skipping, seems random enough
  # next_generation = run_mutation_on_new_gnomes(next_generation)
  
  return next_generation

  
def get_two_best_from_generation(generation, fitness_values):
  min_1 = generation[0]
  min_2 = generation[1]
  for i in range(2, len(generation)):
    if generation[i] < min_1:
      min_1, min_2 = generation[i], min_1
    elif generation[i] < min_2:
      min_2 = generation[i]
  return min_1, min_2

def run_crossover_on_gnomes(next_generation, previous_generation, fitness_values, population_size):
  sorted_fitness_values = fitness_values.sort()
  viable_fitness_values = sorted_fitness_values[:7] # only keep the best 7 to chose from
  
  while len(next_generation) < population_size:
    random_gnome1 = get_random_gnome(previous_generation, viable_fitness_values, fitness_values)
    random_gnome2 = get_random_gnome(previous_generation, viable_fitness_values, fitness_values)

    random_gnome1, random_gnome2 = swap_random_values(random_gnome1, random_gnome2)

    next_generation.append(random_gnome1)
    next_generation.append(random_gnome2)
  return next_generation

# pick gnome with "promising" value. Pick any random 1 of the top 7, bottom 3 are ignored
def get_random_gnome(previous_generation, viable_fitness_values, fitness_values):
  random_index = random.randint(0,6)
  gnome_index = fitness_values.index(viable_fitness_values[random_index])
  return previous_generation[gnome_index]

# run a "crossover" on them: swap one value per row
def swap_random_values(gnome1, gnome2):
  # swap 1 value per row
  for row in range(0, len(gnome1)):
    random_index = random.randint(0,6)

    gnome1[row][random_index], gnome2[row][random_index] = gnome2[row][random_index], gnome1[row][random_index]

    # re-calculate percentages
    gnome1[row] = calculate_percentages(gnome1[row])
    gnome2[row] = calculate_percentages(gnome2[row])

  return gnome1, gnome2

def run_mutation_on_new_gnomes(next_generation):
  pass

# returns a 7x7 result based on the output from the final generation
def get_best_result(results):
  # calculate costs
  fitness_values = calculate_fitness_for_each_gnome(results)

  # get min cost
  min_index = fitness_values.index(min(fitness_values))

  return results[min_index]

##################################################################################
# Final Calculations
##################################################################################

zone_emissions_array = [] # defined as [[total_emissions_for_z1, total_non_emissions_for_z1], [total_emissions_for_z2, total_non_emissions_for_z2], ..]

zone_renewable_array = [] # defined as [[total_renewable_for_z1, total_non_renewable_for_z1], [total_renewable_for_z2, total_non_renewable_for_z2], ..]

cost_zone = [] # 12 x 7 (12 every month, 7 is every zone) -> values from main given forumla

cost_prov = [] # all the zones added up of the array above

# contains all zones total emission values for the month (plant production)
def create_zone_emission_values(): 
  for i in range(0, len(plant_production_rate_array)):
    total_emissions = plant_production_rate_array[i].thermal + plant_production_rate_array[i].combustion_turbine
    total_non_emissions = plant_production_rate_array[i].wind + plant_production_rate_array[i].hydro + plant_production_rate_array[i].nuclear
    zone_emissions_array.append([total_emissions, total_non_emissions])
  
# contains all zones total renewable values for the month (plant production)
def create_zone_renewable_values(): 
  for i in range(0, len(plant_production_rate_array)):
    total_renewable = plant_production_rate_array[i].wind + plant_production_rate_array[i].hydro
    total_non_renewable = plant_production_rate_array[i].thermal + plant_production_rate_array[i].combustion_turbine + plant_production_rate_array[i].nuclear
    zone_renewable_array.append([total_renewable, total_non_renewable])

# # fills the values of the cost_zone array
# def create_cost_zone_values():
#   for month in range(0, 11):
#     for zone in range(0, 7):
#       cost_zone[month, zone] = penatly_value_array[zone] (dot) Danny[month, zone] ... 

# if __name__ == '__main__':
#     get_total_provincial_cost()
#     app.run(debug=True)

def compute_level_2():
  get_total_provincial_cost()

  year_data = json.loads(get_forecasted_data(2022))
  forecast = year_data['forecast']
  month_sums = []
  for i in range(0, 12):
    for zone in forecast.keys():
      month_sums[i] += forecast[zone][i]

  #data = [cost/month, month_sums, % renewables/month]
  data = []

  output_file_level_2("saintjohn", data)
  pass

def run_app():
  # parse in all the data
  get_all_insentive_rates()
  get_all_penatly_values()
  get_all_plant_production_rates()

  # compute level 1
  run_ml()

  compute_level_2()

run_app()

