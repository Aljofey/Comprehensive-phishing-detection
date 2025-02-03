from urllib.parse import urlparse
from bs4 import BeautifulSoup
import urllib.request, urllib.error
import psycopg2
from requests import get
import time
import re
import chardet
import os
import socket
import psycopg2

conn = psycopg2.connect("host=localhost dbname=postgres user=postgres password=asd")
cur = conn.cursor()
        
folder_path1 ="C:\\Users\\ps\\Documents\\datasets\\D333_bad"
items1= os.listdir(folder_path1)

for item in items1:
    try:
        #item_path1 = os.path.join(folder_path1, item)
                match = re.match(r'(\d+)', item)
        
                numeric_part = match.group(1)

                cur.execute("select url from temp2 where id = %s", (numeric_part,))
                rows = cur.fetchall()
                
                if rows:
                                    with open('C:\\Users\\ps\\Documents\\datasets\\D333_bad\\'+str(numeric_part)+'u.text', 'w', encoding='utf-8') as f4:
                                        f4.write(str(rows[0][0]))
                                        #print(rows[0][0])
    except Exception as e:
            print(f"An error occurred: {str(e)}")


                
