import json 
from openai import OpenAI
import pandas as pd
from matplotlib import pyplot as plt
import streamlit as st
import yfinance as yf
from tenacity import retry, wait_random_exponential, stop_after_attempt

GPT_MODEL = "gpt-4-turbo-preview"
client = OpenAI()

def get_stock_price(ticker, start_date, end_date):
    print(f'Getting Data for Ticker = {ticker} for period {start_date} - {end_date}')
    df = yf.Ticker(ticker).history(start=start_date, end=end_date)
    df = df.reset_index().set_index('Date')[['Close']]
    df = df.rename(columns={'Close': ticker})
    #print(df.columns)
    return df

def calculate_sma(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.rolling(window=window).mean().iloc[-1])

def calculate_ema(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])


def create_portfolio(tickers, benchmark, start_date, end_date):
    portfolio = pd.DataFrame()
    factor = 1 / len(tickers)
    for ticker in tickers:
        temp_df = yf.Ticker(ticker).history(start=start_date, end=end_date)
        temp_df = temp_df.reset_index().set_index('Date')[['Close']]
        temp_df = temp_df.rename(columns={'Close': ticker})
        if portfolio.empty:
            portfolio = temp_df
            portfolio['port'] = temp_df[ticker] * factor
        else:
            portfolio = portfolio.merge(temp_df, left_index=True, right_index=True)
            portfolio['port'] += temp_df[ticker] * factor
        
        
    
    temp_benchmark = yf.Ticker(benchmark).history(start=start_date, end=end_date)
    temp_benchmark = temp_benchmark.reset_index().set_index('Date')[['Close']]
    temp_benchmark = temp_benchmark.rename(columns={'Close': benchmark})
    portfolio = portfolio.merge(temp_benchmark, left_index=True, right_index=True)
    portfolio = portfolio[['port', benchmark]]
    return portfolio

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e



tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the close price of a stock given a ticker and start date and end date",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The ticker of the stock. eg AAPL. Note: FB is now META",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "start date in YYYY-MM-DD format",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "start date in YYYY-MM-DD format",
                    },
                },
                "required": ["ticker", "start_date", 'end_date'],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_portfolio",
            "description": "Creates a portfolio by choosing a list of stock tickers and then compare it against a benchmark for a period with start date and end date",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                         "items": {
                        "type": "string"
                        },
                        "description": "List of tickers to be used portfolio construction. eg [AAPL, NVDA]. Note: FB is now META",
                    },
                    "benchmark": {
                        "type": "string",
                        "description": "The ticker of the benchmark. eg AAPL. Note: FB is now META",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "start date in YYYY-MM-DD format",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "start date in YYYY-MM-DD format",
                    },
                },
                "required": ["tickers", "benchmark", "start_date", 'end_date'],
            },
        }
    },
]


f_map = {
    'get_stock_price': get_stock_price,
    'create_portfolio': create_portfolio  
}
messages = []
if 'messages' not in st.session_state:
    
    st.session_state['messages'] = messages
    st.session_state['messages'].append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})

st.title('Alpha Copilot')

user_input = st.text_area('Your Input')
if user_input:
    try:
        question = str(user_input)
        messages.append({'role': 'user', 'content': question})
 
        chat_response = chat_completion_request(
            messages, tools=tools
        )

        assistant_message = chat_response.choices[0].message
        messages.append(assistant_message)
        f_name = f_map[assistant_message.tool_calls[0].function.name]
        args = json.loads(assistant_message.tool_calls[0].function.arguments)
        
        df =f_name(**args)
        st.dataframe(df) 
        st.line_chart(df)

    except Exception as e:
        raise e
        st.text('Try Again...')