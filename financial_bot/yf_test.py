import yfinance as yf, requests, socket, time
symbols = ["RELIANCE.NS","^NSEI","CDSL.NS"]
print("DNS check for yahoo hosts:")
for host in ["query1.finance.yahoo.com","finance.yahoo.com"]:
    try:
        print(host, socket.gethostbyname(host))
    except Exception as e:
        print("DNS error for", host, e)
for sym in symbols:
    try:
        print("===", sym)
        t = yf.Ticker(sym)
        h = t.history(period="2d")
        print("history rows:", None if h is None else len(h))
        if h is not None and not h.empty:
            print(h.tail(2).to_dict())
    except Exception as e:
        print("ERROR:", e)
    time.sleep(1.5)
