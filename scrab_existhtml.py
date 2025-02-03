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




url=""
j=0
typ=""
id=""

counter = 1
action_url=' '
iframe_url=' '
empty_links1=0
empty_links2=0
error_links=0
text=[]
labels=[]


def getnoicytext(html):
    #print("success") 
    #print(array)
    all_div = html.find_all("div")
    for i in all_div:
        if i.get("class")!=None:
            array.append(i.get("class"))
        if i.get("id")!=None:
            array.append(i.get("id"))
    all_meta = html.find_all("meta")
    for i in all_meta:
        if i.get("name")!=None:
            array.append(i.get("name"))
        if i.get("property")!=None:
            array.append(i.get("property"))
    all_body = html.find("body")
    if all_body!=None:
        array.append(all_body.get("class"))
    all_aside = html.find_all("aside")
    for i in all_aside:
        if i.get("class")!=None:
            array.append(i.get("class"))
    all_a = html.find_all("a")
    for i in all_a:
        if i.get("class")!=None:
            array.append(i.get("class"))
    all_img = html.find_all("img")
    for i in all_img:
        if i.get("alt")!=None:
            array.append(i.get("alt"))
    f = html.find("form")
    if f!=None and f.get("class")!=None:
        array.append(f.get("class"))
    all_inputs = html.find_all("input")
    for i in all_inputs:
        if i.get("name")!=None:
            array.append(i.get("name"))
        if i.get("class")!=None:
            array.append(i.get("class"))
        if i.get("placeholder")!=None:
            array.append(i.get("placeholder"))
    all_p = html.find_all("p")
    for i in all_p:
        if i.get("class")!=None:
            array.append(i.get("class"))
    all_option = html.find_all("option")
    for i in all_option:
        if i.get("value")!=None:
            array.append(i.get("value"))
    all_button = html.find_all("button")
    for i in all_button:
        if i.get("class")!=None:
            array.append(i.get("class"))
    all_h1 = html.find_all("h1")
    for i in all_h1:
        if i.get("class")!=None:
            array.append(i.get("class"))
    all_h2 = html.find_all("h2")
    for i in all_h2:
        if i.get("class")!=None:
            array.append(i.get("class"))
    all_h3 = html.find_all("h3")
    for i in all_h3:
        if i.get("class")!=None:
            array.append(i.get("class"))
    all_li = html.find_all("li")
    for i in all_li:
        if i.get("class")!=None:
            array.append(i.get("class"))
        if i.get("id")!=None:
            array.append(i.get("id"))
    all_blockquote = html.find_all("blockquote")
    for i in all_blockquote:
        if i.get("class")!=None:
            array.append(i.get("class"))
    all_span= html.find_all("span")
    for i in all_span:
        if i.get("class")!=None:
            array.append(i.get("class"))
    all_td= html.find_all("td")
    for i in all_td:
        if i.get("class")!=None:
            array.append(i.get("class"))
    all_footer= html.find("footer")
    if all_footer!=None:
         array.append(all_footer.get("class"))
    all_td= html.find_all("td")
    for i in all_td:
        if i.get("class")!=None:
            array.append(i.get("class"))
    all_nav= html.find_all("nav")
    for i in all_nav:
        if i.get("class")!=None:
            array.append(i.get("class"))
        if i.get("id")!=None:
            array.append(i.get("id"))
    all_figure= html.find_all("figure")
    for i in all_figure :
        if i.get("class")!=None:
            array.append(i.get("class"))
    all_section= html.find_all("section")
    for i in all_section :
        if i.get("class")!=None:
            array.append(i.get("class"))
        if i.get("id")!=None:
            array.append(i.get("id"))
    all_time= html.find("time")
    if all_time!=None and all_time.get("class")!=None:
        array.append(all_time.get("class"))
    all_video= html.find("video")
    if all_video!=None and all_video.get("class")!=None:
        array.append(all_video.get("class"))
    all_ins= html.find_all("ins")
    for i in all_ins:
        if i.get("class")!=None:
            array.append(i.get("class"))
    all_article = html.find_all("article")
    for i in all_article:
        if i.get("class")!=None:
            array.append(i.get("class"))
    all_ul = html.find_all("ul")
    for i in all_ul:
        if i.get("class")!=None:
            array.append(i.get("class"))
    all_label= html.find_all("label")
    for i in all_label:
        if i.get("class")!=None:
            array.append(i.get("class"))
    all_svg= html.find_all("svg")
    for i in all_svg:
        if i.get("class")!=None:
            array.append(i.get("class"))
    all_select= html.find_all("select")
    for i in all_select:
        if i.get("class")!=None:
            array.append(i.get("class"))
        if i.get("id")!=None:
            array.append(i.get("id"))
    all_summary= html.find_all("summary")
    for i in all_summary:
        if i.get("class")!=None:
            array.append(i.get("class"))
    all_template= html.find_all("template")
    for i in all_template:
        if i.get("class")!=None:
            array.append(i.get("class"))
        if i.get("id")!=None:
            array.append(i.get("id"))
    all_details= html.find_all("details")
    for i in all_details:
        if i.get("class")!=None:
            array.append(i.get("class"))
            
    return array

def Extract_metadata(soup):
    metadata = soup.find_all('meta')  # Extract all meta tags
    title = soup.title
    tag = []
    new = ""

    if title is not None:
        title = title.string  # Extract the title of the webpage

    meta = soup.find_all('meta')

    if meta is not None:
        for t in meta:
            if t.get('name') != "":
                tag.append(t.get('name'))
            if t.get('value') != "":
                tag.append(t.get('value'))
            if t.get('content') != "":
                tag.append(t.get('content'))

    new = str(title) + str(tag)
    return new




def text_cleaner(text):
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
    return text.lower()

def removestopwords(text):
    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(text)

    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    filtered_sentence = []
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()


    for w in word_tokens:
        if w not in stop_words:
            stem=ps.stem(w)
            filtered_sentence.append(lemmatizer.lemmatize(stem))
            #filtered_sentence.append(w)
    return filtered_sentence

def is_valid(url):
    """
    Checks whether `url` is a valid URL.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def is_empty(url):
    """
    Checks whether `url` is a valid URL.
    """
    if url=='':
        return True
    elif url=='#':
        return True
    elif url=='javascript:void(0);':
        return True
    elif url[0]=='#':
        return True
    else:
        return False

def anchor_URLs(soup):
    # Your code to extract URLs from the soup and store them in a list
    urls = []

    # Example: Find all anchor tags and extract href attributes
    for anchor in soup.find_all('a'):
        href = anchor.get('href')
        if href:
            urls.append(href)

    return urls

def get_JavaScriptfiles(soup):
    # Your code to extract URLs from the soup and store them in a list
    urls = []

    # Example: Find all anchor tags and extract href attributes
    for anchor in soup.find_all('script'):
        href = anchor.get('src')
        if href:
            urls.append(href)

    return urls

def get_action_URLs(soup):
    # Your code to extract URLs from the soup and store them in a list
    urls = []

    # Example: Find all anchor tags and extract href attributes
    for anchor in soup.find_all('form'):
        href = anchor.get('action')
        if href:
            urls.append(href)

    return urls

def imgfiles(soup):
    # Your code to extract URLs from the soup and store them in a list
    urls = []

    # Example: Find all anchor tags and extract href attributes
    for anchor in soup.find_all('img'):
        href = anchor.get('src')
        if href:
            urls.append(href)

    return urls

def CSSfiles(soup):
    # Your code to extract URLs from the soup and store them in a list
    urls = []

    # Example: Find all anchor tags and extract href attributes
    for anchor in soup.find_all('link'):
        href = anchor.get('href')
        if href:
            urls.append(href)

    return urls

def get_iframe(soup):
    # Your code to extract URLs from the soup and store them in a list
    urls = []

    # Example: Find all anchor tags and extract href attributes
    for anchor in soup.find_all('iframe'):
        href = anchor.get('src')
        if href:
            urls.append(href)

    return urls

import os
##text=[]
##labels=[]
##
##
##folder_path1 ="C:\\Users\\ps\\Documents\\datasets\\DatasetwithHTML\\html_data"
##items1= os.listdir(folder_path1)
##for item in items1:
##    item_path1 = os.path.join(folder_path1, item)
##    if "u" not in str(item) and "h" not in str(item):
##        with open(item_path1, 'rb') as file1:
##            text.append( file1.read().decode('utf-8', errors='ignore'))
##            labels.append('0')
##
##folder_path2 ="C:\\Users\\ps\\Documents\\datasets\\DatasetwithHTML\\phishingdata"
##items2= os.listdir(folder_path2)
##for item in items2:
##    item_path2 = os.path.join(folder_path2, item)
##    if "u" not in str(item) and "h" not in str(item):
##        with open(item_path2, 'rb') as file2:
##            text.append( file2.read().decode('utf-8', errors='ignore'))
##            labels.append('1')        
##
##print(len(text))
##print(len(labels))


### Specify the path to the folder you want to read
##folder_path = 'C:\\Users\\ps\\Documents\\datasets\\DatasetwithHTML\\html_data'
##
### List all items (files and subdirectories) in the folder
##items = os.listdir(folder_path)
##
### Iterate through the items
##for item in items:
##    item_path = os.path.join(folder_path, item)  # Get the full path of the item
##    if os.path.isfile(item_path):
##        print(f"File: {item_path}")
##        with open(item_path, 'rb') as file:
##               html_content = file.read().decode('utf-8', errors='ignore')
##               print(html_content)
##    elif os.path.isdir(item_path):
##        print(f"Directory: {item_path}")
##    else:
##        print(f"Unknown: {item_path}")





##rows=int(15)
##for row in range(rows):
##

#for i in range(1, 1001):

folder_path1 ="C:\\Users\\ps\\Documents\\datasets\\D333_bad"
items1= os.listdir(folder_path1) 
for item in items1:
    u=0
    h=0
    array=[]
    array3=[]
    array2=[]
    hperlinks=[]
    cleantext=" "
    str1=""
    noicy=""
    meta=""
    html_content=""
    url=""
    match = re.match(r'(\d+)', str(item))
    number = int(match.group(1))
    #print(item)
    try:
##                    url=""
##                    html=""
        item_path1 = os.path.join(folder_path1, item)
##        if "u"in str(item) :
##                        u=1
##                        with open(item_path1, 'rb') as file1:
##                            url= file1.read().decode('utf-8', errors='ignore')
        if "h"in str(item) :
                        h=1
                        with open(item_path1, 'rb') as file2:
                            html_content= file2.read().decode('utf-8', errors='ignore')

##            if u==1:
##                    url=remove_http_www(url)
##                    url="http://"+url


        
           

##            # Convert the counter to a string with leading zeros
##            counter_str = f"{counter:05d}"
##                # Increment the counter
##            counter += 1    
###dataset d3
##            folder_path ="C:\\Users\\ps\\Documents\\datasets\\DatasetwithHTML\\phishing"+str(row+1)+"\\P"+counter_str+"\\RAW-HTML"
##            items= os.listdir(folder_path)
##            for item in items:
##                item_path = os.path.join(folder_path, item)
##                with open(item_path, 'rb') as file:
##                   html_content = file.read().decode('utf-8', errors='ignore')
##
##            with open('C:\\Users\\ps\\Documents\\datasets\\DatasetwithHTML\\phishing'+str(row+1)+'\\P'+counter_str+'\\URL\\URL.txt', 'rb') as file:
##               url = file.read().decode('utf-8', errors='ignore')


##        if u==1:
##             with open('C:\\Users\\ps\\Documents\\datasets\\D1_good_textual features\\'+str(number)+'u.text', 'w', encoding='utf-8') as f5:
##                f5.write(str(url))
        if h==1:
            soup = BeautifulSoup(html_content, "lxml")
            cleantext = BeautifulSoup(html_content, "lxml").get_text()
            bad_chars={';','0','1','2','3','4','5',
                        '6','7','8','9','\n',':','!',"*",
                        '[',']','{','(',')',",",';','.','!','?',
                        ':',"'",'"\"','/',"\\",'|','_','@','#',
                        '$','%','^','&','*','~','`','+','"','=',
                        '<','>','(',')','[',']','{','}'}
            cleantext=str(cleantext)
            for i in bad_chars: 
                    cleantext = cleantext.replace(i, ' ')
            cleantext = re.sub(r'\s+', ' ', cleantext)
            meta=Extract_metadata(soup)
            meta=str(meta)
            meta=meta.replace("\n","")
            meta=meta.replace("[", "").replace("]", "").replace("'", "")
            cleantext+=meta

            #lxml
            
            hperlinks+=anchor_URLs(soup)
            hperlinks+=get_JavaScriptfiles(soup)
            hperlinks+=get_action_URLs(soup)
            hperlinks+=imgfiles(soup)
            hperlinks+=CSSfiles(soup)
            hperlinks+=get_iframe(soup)
            hy=str(hperlinks)
            hy.replace("\n", "")
            hy=hy.replace("[", "").replace("]", "").replace("'", "")
           #print(hy)
            #array.append(cleantext)
          
           
           
        ##    for i in bad_chars: 
        ##        str1 = str1.replace(i, ' ')
        ##    str1 = str1.replace('-', ' ')
        ##    str1 = str1.replace(' n', ' ')
             #str1 = str1.replace(' n', ' ')
           
           
            #str1=text_cleaner(str1)
           
##            str1 = str1.replace('-', ' ')
##            str1 = str1.replace(' n', ' ')
##            str1 = str1.replace(' t', ' ')
           
            #str1+=hy
            noicy=getnoicytext(soup)
            noicy=str(noicy)
            noicy = noicy.replace("\n","")
            noicy=noicy.replace("[", "").replace("]", "").replace("'", "")

            str1=cleantext+hy+noicy
           # print(text_features)
            
##            text.append(str1)
##            labels.append("1")
            
            with open('C:\\Users\\ps\\Documents\\datasets\\D333_bad_text\\'+str(number)+'.text', 'w', encoding='utf-8') as f1:
                if str1: 
                    f1.write(str1)
                else:
                     f1.write("This is cleantext")
##            with open('C:\\Users\\ps\\Documents\\datasets\\D3333_good_text\\'+str(number)+'h.text', 'w', encoding='utf-8') as f4:
##                f4.write(html_content)
                        

##            with open('C:\\Users\\ps\\Documents\\datasets\\DatasetwithHTML\\phishing_text_feattures\\'+'P'+counter_str+'hy.text', 'w', encoding='utf-8') as f2:
##                if hy: 
##                    f2.write(str(hy))
##                else:
##                     f2.write("This is hyperlinks text")
##                
##
##            with open('C:\\Users\\ps\\Documents\\datasets\\DatasetwithHTML\\phishing_text_feattures\\'+'P'+counter_str+'noicy.text', 'w', encoding='utf-8') as f3:
##                if noicy:
##                    f3.write(str(noicy))
##                else:
##                    f3.write('This is noicy text')

            


           
                
            
    except Exception as e:
##            ee=str(e)
##            if "URL" in ee:
##                with open('C:\\Users\\ps\\Documents\\datasets\\DatasetwithHTML\\html_data\\'+'L'+counter_str+'u.text', 'w', encoding='utf-8') as f2:
##                    f2.write(f"An error occurred: {str(e)}")
            
            print(f"An error occurred: {str(e)}")

##print(len(text))
##print(len(labels))









