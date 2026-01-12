import sqlite3

conn = sqlite3.connect('data/sqlite/agentsync.db')
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in database:")
for tbl in tables:
    print(tbl[0])

conn.close()



conn = sqlite3.connect('data/sqlite/agentsync.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM agents;")
rows = cursor.fetchall()
print("Rows in agents table:")
for row in rows:
    print(row)

conn.close()
