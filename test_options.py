# save as test_options.py and run it
import requests, os
from config import Config
c = Config()

headers = {
    "APCA-API-KEY-ID":     c.ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": c.ALPACA_SECRET_KEY,
}
r = requests.get(
    "https://paper-api.alpaca.markets/v2/options/contracts",
    headers=headers,
    params={
        "underlying_symbols": "NVDA",
        "expiration_date_gte": "2025-05-01",
        "expiration_date_lte": "2025-06-30",
        "type": "put",
        "limit": 5,
    }
)
for c in r.json().get("option_contracts", []):
    print(c["symbol"], "strike=", c["strike_price"], "exp=", c["expiration_date"])