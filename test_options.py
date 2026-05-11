# save as test_options.py and run it
import requests, os
from datetime import datetime, timedelta
from config import Config

c = Config()

lte = (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d")
gte = datetime.now().strftime("%Y-%m-%d")

headers = {
    "APCA-API-KEY-ID": c.ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": c.ALPACA_SECRET_KEY,
}
r = requests.get(
    "https://paper-api.alpaca.markets/v2/options/contracts",
    headers=headers,
    params={
        "underlying_symbols": "NVDA",
        "expiration_date_gte": gte,
        "expiration_date_lte": lte,
        "type": "put",
        "limit": 5,
    },
)
print("Query range:", gte, "to", lte)
print("Status:", r.status_code)
for c in r.json().get("option_contracts", []):
    print(c["symbol"], "strike=", c["strike_price"], "exp=", c["expiration_date"])
