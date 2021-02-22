from flask import Flask
from flask_cors import CORS
from flask_marshmallow import Marshmallow
import os

app = Flask(__name__)
CORS(app)
ma = Marshmallow(app)

class Car():
  def __init__(self, name, mileage, cost):
    self.name = name
    self.mileage = mileage
    self.cost = cost

class CarSchema(ma.Schema):
  class Meta:
    fields = ('name', 'cost', 'mileage')


@app.route('/')
def get_best_car():
  cars = parse_car_data_from_file()
  best_car = get_best_car_from_list(cars)
  print(cars)
  return ""
  pass


def parse_car_data_from_file():
  list_of_cars = []
  data_file = open("data.txt", "r")
  lines_from_file = data_file.readlines()

  # skip the first two lines
  for i in range(2, len(lines_from_file)):
      car = get_car_data_from_line(lines_from_file[i])
      list_of_cars.append(car)
  
  return list_of_cars

def get_car_data_from_line(line):
  car_data = line.split(',')
  car_name = car_data[0]
  car_mileage = car_data[1]
  car_cost = car_data[2]
  car = Car(car_name, car_mileage, car_cost)
  return car

def get_best_car_from_list(list_of_cars):
  pass


if __name__ == '__main__':
    app.run(debug = True) 