from st_aggrid import AgGrid
import pandas as pd

df = pd.read_csv('Health Care Chatbot -FAQ.csv', encoding='ISO-8859-1')
AgGrid(df)
