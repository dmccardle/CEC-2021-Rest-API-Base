from flask import Flask
from flask_cors import CORS
from flask_marshmallow import Marshmallow

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
    fields = ('name, cost, mileage')


@app.route('/')
def get_best_car():
  parse_car_data_from_file()
  pass


def parse_car_data_from_file():
  pass