import requests
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import urllib.request, urllib.error
import psycopg2
from requests import get
import time
import re
import chardet
import os
import csv



def contains_html_content(content):
##    with open(file_path, 'r', encoding='utf-8') as file:
##        content = file.read()
    
    # Check for common HTML tags
    if any(tag in content for tag in ['<html>', '<head>', '<body>', '<p>', '<a>']):
        return True
    
    # Count angle brackets
    angle_bracket_count = content.count('<') + content.count('>')
    if angle_bracket_count > 10:  # Adjust the threshold as needed
        return True
    
    # Check for HTML-related keywords
    if any(keyword in content for keyword in ['DOCTYPE', 'DOCTYPE HTML', 'html']):
        return True
    
    # Analyze content structure (simple heuristic)
    if content.count('<') > content.count('\n') * 0.1:  # Adjust the threshold as needed
        return True
    
    return False

#first  
##folder_path1 ="C:\\Users\\ps\\D1"
##items1= os.listdir(folder_path1)
##for item in items1:
##    try:
##        item_path1 = os.path.join(folder_path1, item)
##        with open(item_path1, 'rb') as file1:
##            html_content = file1.read().decode('utf-8', errors='ignore')
##        if contains_html_content(html_content):
##            with open('C:\\Users\\ps\\Documents\\datasets\\D1_legit\\'+item+'html.text', 'w', encoding='utf-8') as f4:
##                    f4.write(html_content)
##        else:
##            with open('C:\\Users\\ps\\Documents\\datasets\\D1_legit_text\\'+item+'html.text', 'w', encoding='utf-8') as f5:
##                    f5.write(html_content)
##
##    except Exception as e:
##             print(f"An error occurred: {str(e)}")

###second
folder_path1 ="C:\\Users\\ps\\Documents\\datasets\\D1_legit"
items1= os.listdir(folder_path1)

for item in items1:
    try:
        item_path1 = os.path.join(folder_path1, item)
        with open(item_path1, 'rb') as file1:
            html_content = file1.read().decode('utf-8', errors='ignore')
            #print(item)

            match = re.match(r'(\d+)', item)
        if match:
            numeric_part = match.group(1)
                #print(numeric_part)
                
        file_path2 = "C:\\Users\\ps\\D1\\ID__URLs_labell.csv"  # Update with your file path
        with open(file_path2, "r",encoding="utf-8",errors='ignore') as file:
            reader = csv.reader(file, delimiter=",")  # Set the delimiter to "\t" for tab-separated values
            s=0
            for row in reader:
                   
                
##                            print(row[0])
##                            print(numeric_part)
##                            print(row[2])
##                            print('\n')
                            s+=1
                  
                            if str(row[0])==str(numeric_part) and str(row[2])=='1':
                                #print("my duck")
                                with open('C:\\Users\\ps\\Documents\\datasets\\D1_bad\\'+str(row[0])+'h.text', 'w', encoding='utf-8') as f5:
                                    f5.write(html_content)
                                with open('C:\\Users\\ps\\Documents\\datasets\\D1_bad\\'+str(row[0])+'u.text', 'w', encoding='utf-8') as f6:
                                   f6.write(str(row[1]))
                                break
                            if str(row[0])==str(numeric_part) and str(row[2])=='-1':
                                #print("my ass")
                                with open('C:\\Users\\ps\\Documents\\datasets\\D1_good\\'+str(row[0])+'h.text', 'w', encoding='utf-8') as f8:
                                   f8.write(html_content)
                                with open('C:\\Users\\ps\\Documents\\datasets\\D1_good\\'+str(row[0])+'u.text', 'w', encoding='utf-8') as f9:
                                   f9.write(str(row[1]))
                                break
            print(s)



    except Exception as e:
             print(f"An error occurred: {str(e)}")


#third
##
##folder_path1 = "C:\\Users\\ps\\Documents\\datasets\\D1_legit"
##items1 = os.listdir(folder_path1)
##
##for item in items1:
##    try:
##        item_path1 = os.path.join(folder_path1, item)
##        with open(item_path1, 'rb') as file1:
##            html_content = file1.read().decode('latin-1', errors='ignore')
##            # print(item)
##
##            match = re.match(r'(\d+)', item)
##            if match:
##                numeric_part = match.group(1)
##                # print(numeric_part)
##
##        file_path2 = "C:\\Users\\ps\\D1\\ID__URLs_labell.csv"  # Update with your file path
##        with open(file_path2, "r", encoding='utf-8') as file:
##            reader = csv.reader(file, delimiter=",")
##            for row in reader:
##                if row and len(row) > 2:
##                    if str(row[0]) == numeric_part and str(row[2]) == '1':
##                        with open('C:\\Users\\ps\\Documents\\datasets\\D1_bad\\' + str(row[0]) + 'h.text', 'w', encoding='utf-8') as f5:
##                            f5.write(html_content)
##                        with open('C:\\Users\\ps\\Documents\\datasets\\D1_bad\\' + str(row[0]) + 'u.text', 'w', encoding='utf-8') as f6:
##                            f6.write(row[1])
##                        break
##                    if str(row[0]) == str(numeric_part) and str(row[2]) == '-1':
##                        with open('C:\\Users\\ps\\Documents\\datasets\\D1_good\\' + str(row[0]) + 'h.text', 'w', encoding='utf-8') as f5:
##                            f5.write(html_content)
##                        with open('C:\\Users\\ps\\Documents\\datasets\\D1_good\\' + str(row[0]) + 'u.text', 'w', encoding='utf-8') as f6:
##                            f6.write(row[1])
##                        break
##
##    except Exception as e:
##        print(f"An error occurred: {str(e)}")



            
            
