from cgi import print_form
import pandas as pd
import yfinance as yf
import datetime

print("Inserire ticket")
ticket = input()
print("Inserire lunghezza previsione")
inpLen = input()
lenForecast = int(inpLen)

from datetime import date, timedelta
today = date.today()

d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=730)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

#data = yf.download('LTC-EUR', start=start_date, end=end_date, progress=False)
data = yf.download(ticket, start=start_date, end=end_date, progress=False)

data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
print(data.head())

data.shape

import plotly.graph_objects as go
figure = go.Figure(data=[go.Candlestick(x=data["Date"], open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"])])
figure.update_layout(title = "Price Analysis", xaxis_rangeslider_visible=False)
figure.show()


correlation = data.corr()
print(correlation["Close"].sort_values(ascending=False))


from autots import AutoTS
#model = AutoTS(forecast_length=30, frequency='infer', ensemble='simple')
model = AutoTS(forecast_length=lenForecast, frequency='infer', ensemble='simple')
model = model.fit(data, date_col='Date', value_col='Close', id_col=None)
prediction = model.predict()
forecast = prediction.forecast
print(forecast)


#https://thecleverprogrammer.com/2021/12/27/cryptocurrency-price-prediction-with-machine-learning/