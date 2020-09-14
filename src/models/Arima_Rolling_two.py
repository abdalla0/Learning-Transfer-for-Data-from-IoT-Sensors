from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

 
def parser(x):
	return datetime.strptime(x, '%M-%S')

dev_id  = 33 	
series = read_csv('knoy_pcb_1_X_320.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
X = series.values
size = int(len(X) * 0.67)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,2,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test,label='Actual Readings')
pyplot.plot(predictions,label='Predicted Readings', color='red')
legend = pyplot.legend(['Actual Readings', 'Predicted Readings'],loc='upper center', shadow=True, fontsize='x-large')
pyplot.xlabel('Test Samples', fontdict=None, labelpad=None)
pyplot.ylabel('Dielectric', fontdict=None, labelpad=None)
pyplot.show()
