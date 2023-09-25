import requests
import pandas as pd
from datetime import datetime

def get_historical_price_vnd(symbol:str, start_time:int, end_time:int, max_days: int = 365) -> pd.DataFrame:
    """Get historical price from VND

    Args:
        symbol (str): stock name
        start_time (int): VND accept timestamp type int
        end_time (int): VND accept timestamp type int
        max_days (int): batching request based on max days
    Returns:
        pd.DataFrame: Price DataFrame
    """
    if symbol.startswith("VN"):
        symbol="VNINDEX"
    elif symbol.startswith("HN"):
        symbol="HNX"
    elif symbol.startswith("UP"):
        symbol="UPCOM"
    else:
        return pd.DataFrame()
    cookies=None
    total_day = (pd.to_datetime(end_time, unit="s") - pd.to_datetime(start_time, unit="s"))
    try:
        with requests.Session() as session:
            url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution=D&symbol={symbol}&from={start_time}&to={end_time}"
            payload = {}
            headers = {
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
                'Connection': 'keep-alive',
                'Origin': 'https://dchart.vndirect.com.vn',
                'Referer': 'https://dchart.vndirect.com.vn/',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-site',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
                'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"'
            }
            response = session.get(url, headers=headers, data=payload)
            df = pd.DataFrame(response.json())
            df.t = pd.to_datetime(df.t, unit='s')
            
        vnd_column_map = {
            "t": 'rowDate',
            "c":"last_closeRaw",
            "o": "last_openRaw",
            "h": "last_maxRaw",
            "l": "last_minRaw",
            "v": "volumeRaw",
        }
        df.columns = df.columns.map(vnd_column_map)
        df = df[list(vnd_column_map.values())]
        df["rowDate"] = pd.to_datetime(df["rowDate"], format="%Y/%m/%d")
        df = df.sort_values("rowDate")
        numeric_columns = sorted(set(vnd_column_map.values()) - {"rowDate"})
        df.loc[:, numeric_columns] = df.loc[:, numeric_columns].astype("Float32")
        df.sort_values("rowDate", inplace=True)
        return df
    except Exception as e:
        print(e)
    return pd.DataFrame()
    

def get_historical_price_cafef(symbol:str, start_time:str, end_time:str, max_days: int =365) -> pd.DataFrame:
    """Get historical price from VND

    Args:
        symbol (str): stock name
        start_time (str): VND accept timestamp type int e.g. 09/12/2023
        end_time (str): VND accept timestamp type int e.g. 09/18/2023
        max_days (str): batching request based on max days
    Returns:
        pd.DataFrame: Price DataFrame
    """
    try:
        cookies=None
        total_day = (pd.to_datetime(end_time, format="%m/%d/%Y")  - pd.to_datetime(start_time, format="%m/%d/%Y")).days
        with requests.Session() as session:
            url = f"https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx?Symbol={symbol}&StartDate={start_time}&EndDate={end_time}&PageIndex=1&PageSize={total_day}"
            payload = {}
            headers = {
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
                'Connection': 'keep-alive',
                'Cookie': 'ASP.NET_SessionId=2yibk41q4h2aotdmcvb2xvp2; favorite_stocks_state=1; dtdz=9fc6b680-ae5c-4389-b2df-f3241d14dfe4; _gid=GA1.2.1305167968.1695010007; _ga_860L8F5EZP=GS1.1.1695010004.5.1.1695010007.0.0.0; _ga=GA1.1.1531303246.1692260460; _ga_XLBBV02H03=GS1.1.1695007312.2.1.1695012086.0.0.0; _ga_D40MBMET7Z=GS1.1.1695007312.2.1.1695012086.0.0.0',
                'Referer': 'https://s.cafef.vn/lich-su-giao-dich-vnindex-1.chn',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-origin',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
                'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"'
            }
            response = session.get(url, headers=headers, data=payload, cookies=cookies)
            df = pd.DataFrame(response.json()['Data']['Data'])
            if session.cookies.get_dict():
                session=session.cookies.get_dict()
                
            cafef_column_map = {
            "Ngay": 'rowDate',
            "GiaDongCua": "last_closeRaw",
            "GiaMoCua": "last_openRaw",
            "GiaCaoNhat": "last_maxRaw",
            "GiaThapNhat": "last_minRaw",
            "KhoiLuongKhopLenh": "volumeRaw",
            }
            df.columns = df.columns.map(cafef_column_map)
            df = df[list(cafef_column_map.values())]
            df["rowDate"] = pd.to_datetime(df["rowDate"], format="%Y/%m/%d")
            numeric_columns = sorted(set(cafef_column_map.values()) - {"rowDate"})
            df.loc[:, numeric_columns] = df.loc[:, numeric_columns].astype("Float32")
            df = df.sort_values("rowDate").reset_index(drop=True)
        return df
    except Exception as e:
        print(e)
    return pd.DataFrame()

def get_token_fireant(session):
    token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6IkdYdExONzViZlZQakdvNERWdjV4QkRITHpnSSIsImtpZCI6IkdYdExONzViZlZQakdvNERWdjV4QkRITHpnSSJ9.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmZpcmVhbnQudm4iLCJhdWQiOiJodHRwczovL2FjY291bnRzLmZpcmVhbnQudm4vcmVzb3VyY2VzIiwiZXhwIjoxODg5NjIyNTMwLCJuYmYiOjE1ODk2MjI1MzAsImNsaWVudF9pZCI6ImZpcmVhbnQudHJhZGVzdGF0aW9uIiwic2NvcGUiOlsiYWNhZGVteS1yZWFkIiwiYWNhZGVteS13cml0ZSIsImFjY291bnRzLXJlYWQiLCJhY2NvdW50cy13cml0ZSIsImJsb2ctcmVhZCIsImNvbXBhbmllcy1yZWFkIiwiZmluYW5jZS1yZWFkIiwiaW5kaXZpZHVhbHMtcmVhZCIsImludmVzdG9wZWRpYS1yZWFkIiwib3JkZXJzLXJlYWQiLCJvcmRlcnMtd3JpdGUiLCJwb3N0cy1yZWFkIiwicG9zdHMtd3JpdGUiLCJzZWFyY2giLCJzeW1ib2xzLXJlYWQiLCJ1c2VyLWRhdGEtcmVhZCIsInVzZXItZGF0YS13cml0ZSIsInVzZXJzLXJlYWQiXSwianRpIjoiMjYxYTZhYWQ2MTQ5Njk1ZmJiYzcwODM5MjM0Njc1NWQifQ.dA5-HVzWv-BRfEiAd24uNBiBxASO-PAyWeWESovZm_hj4aXMAZA1-bWNZeXt88dqogo18AwpDQ-h6gefLPdZSFrG5umC1dVWaeYvUnGm62g4XS29fj6p01dhKNNqrsu5KrhnhdnKYVv9VdmbmqDfWR8wDgglk5cJFqalzq6dJWJInFQEPmUs9BW_Zs8tQDn-i5r4tYq2U8vCdqptXoM7YgPllXaPVDeccC9QNu2Xlp9WUvoROzoQXg25lFub1IYkTrM66gJ6t9fJRZToewCt495WNEOQFa_rwLCZ1QwzvL0iYkONHS_jZ0BOhBCdW9dWSawD6iF1SIQaFROvMDH1rg"
    return token
def get_historical_price_fireant(symbol:str, start_time:str, end_time:str, max_days: int =365) -> pd.DataFrame:
    """Get historical price from FIREANT

    Args:
        symbol (str): stock name
        start_time (str): FIREANT accept timestamp type int e.g. 09-12-2023
        end_time (str): FIREANT accept timestamp type int e.g. 09-18-2023
        max_days (str): batching request based on max days
    Returns:
        pd.DataFrame: Price DataFrame
    """
    try:
        cookies=None
        total_day = (pd.to_datetime(end_time, format="%m-%d-%Y")  - pd.to_datetime(start_time, format="%m-%d-%Y")).days
        with requests.Session() as session:
            token = get_token_fireant(session)
            url = f"https://restv2.fireant.vn/symbols/{symbol}/historical-quotes?startDate={start_time}&endDate={end_time}&offset=0&limit={total_day}"
            print(token)
            payload = {}
            headers = {
                'authority': 'restv2.fireant.vn',
                'accept': 'application/json, text/plain, */*',
                'accept-language': 'en-US,en;q=0.9,vi;q=0.8',
                'authorization': f'Bearer {token}',
                'origin': 'https://fireant.vn',
                'referer': 'https://fireant.vn/',
                'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-site',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
            }
            response = session.get(url, headers=headers, data=payload, cookies=cookies)
            df = pd.DataFrame(response.json())
            if session.cookies.get_dict():
                cookies=session.cookies.get_dict()
                
            fireant_column_map = {
                "date": 'rowDate',
                "priceClose": "last_closeRaw",
                "priceOpen": "last_openRaw",
                "priceHigh": "last_maxRaw",
                "priceLow": "last_minRaw",
                "dealVolume": "volumeRaw",
            }
            df.columns = df.columns.map(fireant_column_map)
            df = df[list(fireant_column_map.values())]
            df["rowDate"] = pd.to_datetime(df["rowDate"], format="%Y/%m/%d")
            numeric_columns = sorted(set(fireant_column_map.values()) - {"rowDate"})
            df.loc[:, numeric_columns] = df.loc[:, numeric_columns].astype("Float32")
            df = df.sort_values("rowDate").reset_index(drop=True)
        return df
    except Exception as e:
        print(e)
    return pd.DataFrame()

def get_vn_holidays() -> pd.DataFrame:
    try:
        df_holiday = []
        time_now = datetime.now().date()
        i = 0
        while True:
            try:
                year = time_now.year + i
                holidays = pd.read_html(f"https://www.officeholidays.com/countries/vietnam/{year}")[0]
                holidays["Date"] = pd.to_datetime(holidays["Date"] +' ' +str(year), format="%b %d %Y")
                df_holiday.append(holidays)
                i += 1
            except Exception as e:
                break
        df_holiday = pd.concat(df_holiday).sort_values("Date").reset_index(drop=True)
        return df_holiday
    except Exception as e:
        print(e)
    return pd.DataFrame()