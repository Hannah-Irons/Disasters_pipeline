import sqlite3 as sql

con = sql.connect(r'./DisasterResponse.db')
mycur = con.cursor() 
mycur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
available_table=(mycur.fetchall())
print(available_table)