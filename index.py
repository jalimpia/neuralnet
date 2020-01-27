from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def derivatives(x):
    return x * (1-x)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/home')
def home():
	return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
	training_input = np.array([[0,1,0],
							   [0,0,1],
							   [1,0,0],
							   [1,1,0],
							   [1,1,1]])
	training_output = np.array([[1,0,0,1,1]]).T
	np.random.seed(1)
	weights = 2 * np.random.rand(3,1) - 1
	bias = np.random.rand(1)
	for i in range(1000):
		output = sigmoid(np.dot(training_input,weights)+bias)
		error = training_output - output
		
		adjustment = error * derivatives(output)
		weights += np.dot(training_input.T,adjustment)
		for num in adjustment:
			bias = bias * num

	x1 = request.form['x1']
	x2 = request.form['x2']
	x3 = request.form['x3']
	
	new_data = np.array([int(x1),int(x2),int(x3)])
	result = sigmoid(np.dot(new_data,weights)+bias)
	return str(result)


if __name__ == "__main__":
	app.run(debug=True, host="0.0.0.0")

