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

car_schema = CarSchema()
cars_schema = CarSchema(many=True)

@app.route('/')
def get_best_car():
  cars = parse_car_data_from_file()
  best_car = pick_best_car_from_list(cars)
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

def pick_best_car_from_list(list_of_cars):
  if not list_of_cars:
    return
  
  best_car = list_of_cars[0]
  for i in range(1, len(list_of_cars)):
    if list_of_cars[i].cost < best_car.cost:
      best_car = list_of_cars[i]
  return best_car

if __name__ == '__main__':
    app.run(debug = True) 