import sqlite3

connection = sqlite3.connect('sales.db')
cursor = connection.cursor()

# command = """CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT NOT NULL)"""

# cursor.execute(command)

cursor.execute("INSERT INTO users (username, password) VALUES ('admin','admin')")
cursor.execute("INSERT INTO users (username, password) VALUES ('analyst','password')")
cursor.execute("INSERT INTO users (username, password) VALUES ('peony','pass123')")

connection.commit()