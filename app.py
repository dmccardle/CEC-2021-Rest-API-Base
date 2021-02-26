from flask import Flask, request
from flask_cors import CORS
from flask_marshmallow import Marshmallow
import os
import json

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

car_schema = CarSchema()
cars_schema = CarSchema(many=True)

@app.route('/', methods = ['GET', 'POST'])
def get_best_car():
  request_data = json.loads(request.data)
  min_price = int(request_data['minPrice'])

  cars = parse_car_data_from_file()
  best_car = pick_best_car_from_list(cars, min_price)
  return car_schema.jsonify(best_car)

@app.route('/all')
def get_all_cars():
   cars = parse_car_data_from_file()
   return cars_schema.jsonify(cars)

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
  car_mileage = int(car_data[1])
  car_cost = int(car_data[2])
  car = Car(car_name, car_mileage, car_cost)
  return car

def pick_best_car_from_list(list_of_cars, min_price):
  if not list_of_cars:
    return
  
  list_of_possible_cars = list(filter(lambda car: car.cost >= min_price, list_of_cars))
  print(list_of_possible_cars)

  best_car = min(list_of_possible_cars, key=lambda car: car.cost)
  return best_car

if __name__ == '__main__':
    app.run(debug = True) 