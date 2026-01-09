
import sqlite3
import pandas as pd

conn = sqlite3.connect('metrics.db')
df = pd.read_sql_query("SELECT risk_level, COUNT(*) as count FROM inference_logs GROUP BY risk_level", conn)
print("Risk Levels in DB:")
print(df)
conn.close()
