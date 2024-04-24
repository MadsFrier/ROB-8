# loading in modules
import sqlite3
import cv2

# creating file path
dbfile = '../test.db'
# Create a SQL connection to our SQLite database
con = sqlite3.connect(dbfile)

# creating cursor
cur = con.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")

rows = cur.fetchall()

for row in rows:
    print(row)

# Be sure to close the connection
con.close()