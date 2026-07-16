import requests
key = "e3e661ace0b2420babebd6b1c371ddfe"
url = f"https://api.twelvedata.com/quote?symbol=EUR/USD&apikey={key}"
print(requests.get(url).json())