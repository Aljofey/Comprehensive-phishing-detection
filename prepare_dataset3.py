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


##
with open('C:\\Users\\ps\\Documents\\datasets\\antiphishing\\sourceData_old\\sourceData\\index\\alexa_links.txt', 'r') as file:
    urls = file.read().splitlines()

import requests
timeout = 10

count=445
for url in urls:
    try:
            socket.setdefaulttimeout(timeout)
        #response = requests.get("http://"+url)
            response = urllib.request.urlopen(url)
##        if response.status_code == 200:
            html_content = BeautifulSoup(response, 'lxml')
            #html_contents.append(response.text)
            with open('C:\\Users\\ps\\Documents\\datasets\\D333_good\\'+str(count)+'h.text', 'w', encoding='utf-8') as f1:
                    f1.write(str(html_content))
            with open('C:\\Users\\ps\\Documents\\datasets\\D333_good\\'+str(count)+'u.text', 'w', encoding='utf-8') as f4:
                    f4.write(url)
            count+=1
        
    except Exception as e:
        print(f"Failed to retrieve HTML from {url}: {str(e)}")
        


##folder_path1 ="C:\\Users\\ps\\Documents\\datasets\\D33_good"
##items1= os.listdir(folder_path1)
##
##for item in items1:
##    try:
##        #item_path1 = os.path.join(folder_path1, item)
##                match = re.match(r'(\d+)', item)
##        
##                numeric_part = match.group(1)
##
##                with open('C:\\Users\\ps\\Documents\\datasets\\antiphishing\\all\\sourceData\\index\\legi_phish_idx.txt', 'r', encoding='latin-1') as file:
##                        urls = file.read().splitlines()
##                        for url in urls:
##                                # Extract the part after 'legitimate/' and before '/'
##                                number_part = re.search(r'L(\d+)', url)
##                                number_value = number_part.group(1)
##                                # Extract the part before '/!!!'
##                                before_exclamation = re.search(r'(.+?)!!!', url)
##                                before_exclamation_value = before_exclamation.group(1)
##
##                                if numeric_part== number_value:
##                                    with open('C:\\Users\\ps\\Documents\\datasets\\D33_URLs_good\\'+str(numeric_part)+'u.text', 'w', encoding='utf-8') as f4:
##                                        f4.write(before_exclamation_value)
##                                    break;
##    except Exception as e:
##        print(f"An error occurred: {str(e)}")
                                    

                                
                            
                              

            
            
            
            













#counter = 8000
###fisrt
##for i in range(8000, 16417):
##    try:
##            ##            counter_str = f"{counter:05d}"
##                html=' '
##                       
##                with open('C:\\Users\\ps\\Documents\\datasets\\antiphishing\\all\\sourceData\\legitimate\\L'+str(i)+'\\sourceHtml.txt', 'rb') as file:
##                           html = file.read().decode('utf-8', errors='ignore')
##
##                with open('C:\\Users\\ps\\Documents\\datasets\\D33_good\\'+str(i)+'h.text', 'w', encoding='utf-8') as f4:
##                            f4.write(html)
##                
##                #counter += 1
##    except Exception as e:
##          print(f"An error occurred: {str(e)}")



        
