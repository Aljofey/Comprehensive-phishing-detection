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
from tld import get_tld
import tldextract
from collections import Counter
import decimal
from hyphenate import hyphenate_word
import whois
homo_dictionary = {}
import socket
from ipwhois import IPWhois
import distutils.spawn
import mechanize
import pandas as pd
import os




qwerty = {'1': '2q', '2': '3wq1', '3': '4ew2', '4': '5re3', '5': '6tr4', '6': '7yt5', '7': '8uy6', '8': '9iu7',
          '9': '0oi8', '0': 'po9', 'q': '12wa', 'w': '3esaq2', 'e': '4rdsw3', 'r': '5tfde4', 't': '6ygfr5',
          'y': '7uhgt6', 'u': '8ijhy7', 'i': '9okju8', 'o': '0plki9', 'p': 'lo0', 'a': 'qwsz', 's': 'edxzaw',
          'd': 'rfcxse', 'f': 'tgvcdr', 'g': 'yhbvft', 'h': 'ujnbgy', 'j': 'ikmnhu', 'k': 'olmji', 'l': 'kop',
          'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'}

qwertz = {'1': '2q', '2': '3wq1', '3': '4ew2', '4': '5re3', '5': '6tr4', '6': '7zt5', '7': '8uz6', '8': '9iu7',
          '9': '0oi8', '0': 'po9', 'q': '12wa', 'w': '3esaq2', 'e': '4rdsw3', 'r': '5tfde4', 't': '6zgfr5',
          'z': '7uhgt6', 'u': '8ijhz7', 'i': '9okju8', 'o': '0plki9', 'p': 'lo0', 'a': 'qwsy', 's': 'edxyaw',
          'd': 'rfcxse', 'f': 'tgvcdr', 'g': 'zhbvft', 'h': 'ujnbgz', 'j': 'ikmnhu', 'k': 'olmji', 'l': 'kop',
          'y': 'asx', 'x': 'ysdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'}

azerty = {'1': '2a', '2': '3za1', '3': '4ez2', '4': '5re3', '5': '6tr4', '6': '7yt5', '7': '8uy6', '8': '9iu7',
          '9': '0oi8', '0': 'po9', 'a': '2zq1', 'z': '3esqa2', 'e': '4rdsz3', 'r': '5tfde4', 't': '6ygfr5',
          'y': '7uhgt6', 'u': '8ijhy7', 'i': '9okju8', 'o': '0plki9', 'p': 'lo0m', 'q': 'zswa', 's': 'edxwqz',
          'd': 'rfcxse', 'f': 'tgvcdr', 'g': 'yhbvft', 'h': 'ujnbgy', 'j': 'iknhu', 'k': 'olji', 'l': 'kopm',
          'm': 'lp', 'w': 'sxq', 'x': 'wsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhj'}

keyboards = [qwerty, qwertz, azerty]


#check Deep Feature replacement and called in Containment()

def replacement(domain):
    result = []

    for i in range(0, len(domain)):
        for keys in keyboards:
            if domain[i] in keys:
                for c in keys[domain[i]]:
                    result.append(domain[:i] + c + domain[i+1:])

    return list(set(result))


#check Deep Feature subdomain and called in Containment()

def subdomain(domain):
    result = []

    for i in range(1, len(domain)-1):
        if domain[i] not in ['-', '.'] and domain[i-1] not in ['-', '.']:
            result.append(domain[:i] + '.' + domain[i:])

    return result


#check Deep Feature transpose and called in Containment()

def transposition(domain):
    result = []

    for i in range(0, len(domain)-1):
        if domain[i+1] != domain[i]:
            result.append(domain[:i] + domain[i+1] + domain[i] + domain[i+2:])

    return result


#check Deep Feature addition and called in Containment()

def addition(domain):
    result = []

    for i in range(97, 123):
        result.append(domain + chr(i))

    return result


def repetition(domain):
    result = []

    for i in range(0, len(domain)):
        if domain[i].isalnum():
            result.append(domain[:i] + domain[i] + domain[i] + domain[i+1:])

    return list(set(result))

def omission(domain):
    result = []

    for i in range(0, len(domain)):
        result.append(domain[:i] + domain[i+1:])

    return list(set(result))

def insertion(domain):
    result = []

    for i in range(1, len(domain)-1):
        for keys in keyboards:
            if domain[i] in keys:
                for c in keys[domain[i]]:
                    result.append(domain[:i] + c + domain[i] + domain[i+1:])
                    result.append(domain[:i] + domain[i] + c + domain[i+1:])

    return list(set(result))


def bitsquatting(domain):
    result = []
    masks = [1, 2, 4, 8, 16, 32, 64, 128]

    for i in range(0, len(domain)):
        c = domain[i]
        for j in range(0, len(masks)):
            b = chr(ord(c) ^ masks[j])
            o = ord(b)
            if (o >= 48 and o <= 57) or (o >= 97 and o <= 122) or o == 45:
                result.append(domain[:i] + b + domain[i+1:])

    return result

def vowel_swap(domain):
    vowels = 'aeiou'
    result = []

    for i in range(0, len(domain)):
        for vowel in vowels:
            if domain[i] in vowels:
                result.append(domain[:i] + vowel + domain[i+1:])

    return list(set(result))

def switch_all_letters(url):
    domains = []
    # url = get_tld(url, as_object=True, fix_protocol=True)

    domain = url
    a = []
    j = 0
    glyphs = homo_dictionary
    result1 = set()
    for ws in range(1, len(domain)):
        for i in range(0, (len(domain) - ws) + 1):
            win = domain[i:i + ws]
            j = 0
            while j < ws:
                c = win[j]
                if c in glyphs:
                    win_copy = win
                    for g in glyphs[c]:
                        win = win.replace(c, g)
                        result1.add(domain[:i] + win + domain[i + ws:])
                        win = win_copy
                j += 1

    result2 = set()
    for domain in result1:
        for ws in range(1, len(domain)):
            for i in range(0, (len(domain) - ws) + 1):
                win = domain[i:i + ws]
                j = 0
                while j < ws:
                    c = win[j]
                    if c in glyphs:
                        win_copy = win
                        for g in glyphs[c]:
                            win = win.replace(c, g)
                            result2.add(domain[:i] + win + domain[i + ws:])
                            win = win_copy
                    j += 1
    return list(result1 | result2)



def is_registered(domain_name):
    try:
        w = whois(domain_name)
        print(w)
    except Exception:
        return False
    else:
        return bool(w.domain_name)

def URLLength(str):
    length = len(str)
    #print ("The length of the URL is: ", length)
    return length

# Function to calculate ratio of digits to alphabets Feature09
def DigitAlphabetRatio(str):

    digit = 0
    numeric = set("0123456789")

    for num in str:
        if num in numeric:
            digit = digit + 1

    alphabet = 0
    engletter = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    flag = "Undefined"
    for num in str:
        if num in engletter:
            alphabet = alphabet + 1

    if alphabet != 0:
        ratio = digit/alphabet
        return ratio

    else:
        return flag
    
# Function to calculate ratio of special characters to alphabets Feature10 
def SpecialcharAlphabetRatio(str):
    schar = 0
    specialchar = set("!#$%&'()*+,-./:;<=>?@[\]^_`{|}~\"")

    for num in str:
        if num in specialchar:
            schar = schar + 1

    alphabet = 0
    engletter = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    flag = "Undefined"
    for num in str:
        if num in engletter:
            alphabet = alphabet + 1

    if alphabet != 0:
        ratio = schar / alphabet
        return ratio

    else:
        return flag

# Function to calculate ratio of uppercase letters to lowercase letters Feature11
def UppercaseLowercaseRatio(str):
    ucase = 0
    uppercase = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    for num in str:
        if num in uppercase:
            ucase = ucase + 1

    lcase = 0
    lowercase = set("abcdefghijklmnopqrstuvwxyz")
    flag = "Undefined"

    for num in str:
        if num in lowercase:
            lcase = lcase + 1

    if lcase != 0:
        ratio = ucase / lcase
        return ratio

    else:
        return flag

# Function to calculate ratio of Domain length to URL length Feature12
def DomainURLRatio(str):
    urllength = len(str)

    parsed_url = urllib.parse.urlparse(str)
    domain = parsed_url.netloc
    domainlength = len(domain)
    flag = "Undefined"

    if urllength != 0:
        ratio = domainlength / urllength
        return ratio

    else:
        return 0

# Function to count numeric characters Feature13
def NumericCharCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of numeric characters
    numeric = set("0123456789")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If numeric character is present
        # in set numeric
        if num in numeric:
            count = count + 1

    return count

# Function to count english letters Feature14
def EnglishLetterCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of english letters
    engletter = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If english letter is present
        # in set engletter
        if num in engletter:
            count = count + 1

    return count

# Function to count Special Characters Feature15
def SpecialCharCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of special characters
    specialchar = set("!#$%&'()*+,-./:;<=>?@[\]^_`{|}~\"")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If special character is present
        # in set specialchar
        if num in specialchar:
            count = count + 1

    return count

def DotCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Dot
    dot = set(".")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If dot character is present
        # in set dot
        if num in dot:
            count = count + 1
    return count

def SemiColCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Semi-colon
    semicolon = set(";")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If semi-colon character is present
        # in set semicolon
        if num in semicolon:
            count = count + 1

    return count


# Function to count Underscore Feature18
def UnderscoreCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Underscore
    underscore = set("_")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If underscore character is present
        # in set underscore
        if num in underscore:
            count = count + 1

    return count

def QuesMarkCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Question Mark
    quesmark = set("?")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If Question Mark character is present
        # in set QuesMark
        if num in quesmark:
            count = count + 1

    return count


def HashCharCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Hash Character
    hashchar = set("#")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If Hash Character is present
        # in set hashchar
        if num in hashchar:
            count = count + 1

    return count

def EqualCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Equals to Character
    equalchar = set("=")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If Equals to Character character is present
        # in set equalchar
        if num in equalchar:
            count = count + 1

    return count

def PercentCharCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Percentage Character
    percentchar = set("%")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If Percentage Character is present
        # in set percentchar
        if num in percentchar:
            count = count + 1

    return count

def AmpersandCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Ampersand Character
    ampersandchar = set("&")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If Ampersand Character is present
        # in set ampersandchar
        if num in ampersandchar:
            count = count + 1

    return count

def DashCharCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Dash Character
    dashchar = set("-")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If Dash Character is present
        # in set dashchar
        if num in dashchar:
            count = count + 1

    return count

def DelimiterCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Delimiter Characters
    delim = set("(){}[]<>'\"")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If Delimiter Character is present
        # in set delimiter
        if num in delim:
            count = count + 1

    str1 = str.lower()
    # In string, what is the count that <? occurs
    a = str1.count("<?")
    if a != 0:
        count = count-a

    str2 = str.lower()
    # In string, what is the count that ?> occurs
    b = str2.count("?>")
    if b != 0:
        count = count-b

    str3 = str.lower()
    # In string, what is the count that <% occurs
    c = str3.count("<%")
    if c != 0:
        count = count - c

    str4 = str.lower()
    # In string, what is the count that %> occurs
    d = str4.count("%>")
    if d != 0:
        count = count - d

    str5 = str.lower()
    # In string, what is the count that /* occurs
    e = str5.count("/*")

    str6 = str.lower()
    # In string, what is the count that */ occurs
    f = str6.count("*/")

    return count+a+b+c+d+e+f

def AtCharCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of At Character
    atchar = set("@")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If At Character is present
        # in set atchar
        if num in atchar:
            count = count + 1

    return count

# Function to count Tilde Character Feature27
def TildeCharCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Tilde Character
    tildechar = set("~")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If Tilde Character character is present
        # in set tildechar
        if num in tildechar:
            count = count + 1

    return count


def DoubleSlashCount(str):
    str = str.lower()
    # In string, what is the count that // occurs
    count = str.count("//")
    return count

def CheckIPAsHostName(stri):
    parsed_url = urllib.parse.urlparse(stri)
    #print(parsed_url)
    h = parsed_url.netloc
    if type(h) == str:
        flag = 1
    else:
        flag = 0
    return flag

def HostNameLength(str):
    parsed_url = urllib.parse.urlparse(str)
    #print(parsed_url.netloc)
    return len(parsed_url.netloc)

def PathLength(str):
    parsed_url = urllib.parse.urlparse(str)
    #print(parsed_url.path)
    return len(parsed_url.path)

def QueryLength(str):
    parsed_url = urllib.parse.urlparse(str)
    #print(parsed_url.query)
    return len(parsed_url.query)

def HttpsInHostName(str):
    parsed_url = urllib.parse.urlparse(str)
    hostname = parsed_url.netloc
    #print(hostname)
    hostname = hostname.lower()
    # In string, what is the count that // occurs
    count = 0
    count = hostname.count("https")
    if count == 0:
        #print("Not present")
        return 0
    else:
        if count != 0:
            #print("Present")
            return 1

def DomainURLRatio(str):
    urllength = len(str)

    parsed_url = urllib.parse.urlparse(str)
    domain = parsed_url.netloc
    domainlength = len(domain)
    flag = "Undefined"

    if urllength != 0:
        ratio = domainlength / urllength
        return ratio

    else:
        return flag

def TLD(str):
    res = get_tld(str, as_object=True)
    a = res.tld
    return a

def IsHashed(str):

    def is_hex(str):

        hex_digits = set(string.hexdigits)

        return all(c in hex_digits for c in str)

    ishash = False
    if len(str) ==16 or len(str) == 32 or len(str) == 64 and str.isdigit == True:
        ishash = True
        #print("Hashed")

    if (len(str) == 32 or len(str) == 64 or len(str) == 128) and is_hex(str) == True:
        ishash = True
        return True
    else:
        #print("Not Hashed")
        return False

def TLDInSubdomain(str):
    res = get_tld(str, fix_protocol=True)
    subdom1 = (tldextract.extract(str))
    subdom2 = (subdom1.subdomain)
    if res in subdom2:
        #print("Yes")
        return 1
    else:
        #print("No")
        return 0


def TLDInPath(str):
    parsed_url = urllib.parse.urlparse(str)
    h = parsed_url.path
    #print(h)
    res = get_tld(str, fix_protocol=True)
    if res in h:
        #print("Yes")
        return 1
    else:
        #print("No")
        return 0

def HttpsInUrl(str):
    res = "https"
    if res in str:
        #print("Yes")
        return 1
    else:
        #print("No")
        return 0


def DistDigitAlphabet(str):
    r_avg = 0
    letters = sum(c.isalpha() for c in str)
    #print(letters)
    numbers = sum(c.isdigit() for c in str)
    #print(numbers)
    number_ratio = numbers / len(str)
    alphabet_ratio = letters / len(str)
    #print(alphabet_ratio)
    #print(number_ratio)

    if alphabet_ratio != 0:
        r_avg = r_avg + (number_ratio / alphabet_ratio)
    elif alphabet_ratio == 0:
        r_avg = r_avg + 1

    #print(r_avg)
    #x = number_ratio / alphabet_ratio
    #print(x)

    if alphabet_ratio != 0:
        r_distance = r_avg - (number_ratio / alphabet_ratio)
    elif alphabet_ratio == 0:
        r_distance = r_avg - 1

    return r_distance

def IsDomainEnglishWord(str):
    parsedurl = tldextract.extract(str)
    dom = parsedurl.domain
    #print(dom)

    res = dom.isalpha()
    return 1 if res else 0

def IsDomainPronounceable(str):
    dictionary = enchant.Dict("en_US")
    parsedurl = tldextract.extract(str)
    dom = parsedurl.domain
    #print(dom)

    #syn = wordnet.synsets(dom)[0]
    #res = syn.pos()
    #print(res)

    res2 = dictionary.check(dom)
    res3 = dom.isalpha()

    check = 2
    if res3 == True and res2 == True:
        #if res == "n" or res == "v" or res == "a" or res == "r":
            check = 1
    else:
        check = 0

    if check == 1:
        #print("Pronounceable")
        return True
    else:
        #print("Not pronounceable")
        return False


#Function to calculate Unigram probability of the URL Feature45
def Unigram(str):
    #print("Hello World")

    concat_total_url = ''
    val_without_tld = (str.rsplit('.', 1))[0]
    #print(val_without_tld)

    concat_total_url = concat_total_url + val_without_tld
    #print(concat_total_url)


    # for calculate distribuation alphabet for Unigram calculation
    len_concat_total_url = len(concat_total_url)
    res = Counter(concat_total_url[idx: idx + 1] for idx in range(len_concat_total_url - 1))
    dict_res = dict(res)
    for c in dict_res:
        if len(c) == 1:
            dict_res[c] = dict_res[c] / len_concat_total_url

    # calculate Unigram probability
    concat_url = val_without_tld
    p_uni_gram = 1
    concat_url = val_without_tld
    # print(dict_res)

    # print(type(dict_res))
    res = 1
    for val in dict_res.values():
        res = res * val

    p_uni_gram = res / len(dict_res)
    return p_uni_gram


#Function to calculate Bigram probability of the URL Feature46
def Bigram(str):
    concat_total_url = ''
    val_without_tld = (str.rsplit('.', 1))[0]
    # print(val_without_tld)

    concat_total_url = concat_total_url + val_without_tld
    # print(concat_total_url)


    # for calculate distribuation alphabet for Bigram calculation
    len_concat_total_url = len(concat_total_url)
    res1 = Counter(concat_total_url[idx: idx + 2] for idx in range(len_concat_total_url - 1))
    dict_res1 = dict(res1)
    for c1 in dict_res1:
        if len(c1) == 2:
            dict_res1[c1] = dict_res1[c1] / len_concat_total_url


    # calculate Bigram probability
    concat_url = val_without_tld
    len_concat_total_url_bigram = len(concat_url)
    res_bigram = Counter(concat_url[idx1: idx1 + 2] for idx1 in range(len_concat_total_url_bigram - 1))
    p_bi_gram = 1
    for u1 in res_bigram:
        if len(u1) == 2:
            p_bi_gram = p_bi_gram * dict_res1[u1]
    p_bi_gram = p_bi_gram * (len(concat_url) / len_concat_total_url * 100)
    decimal.getcontext().prec = 25  # Change 25 to the precision you want.
    p_bi_gram = decimal.Decimal(p_bi_gram) / decimal.Decimal(10)

    return p_bi_gram


#Function to calculate Trigram probability of the URL Feature47
def Trigram(str):
    concat_total_url = ''
    val_without_tld = (str.rsplit('.', 1))[0]
    # print(val_without_tld)

    concat_total_url = concat_total_url + val_without_tld
    # print(concat_total_url)
    # for calculate distribuation alphabet for Trigram calculation
    len_concat_total_url = len(concat_total_url)
    res2 = Counter(concat_total_url[idx: idx + 3] for idx in range(len_concat_total_url - 1))
    dict_res2 = dict(res2)
    for c2 in dict_res2:
        if len(c2) == 3:
            dict_res2[c2] = dict_res2[c2] / len_concat_total_url
    # calculate Trigram probability
    concat_url = val_without_tld
    len_concat_total_url_trigram = len(concat_url)
    res_trigram = Counter(concat_url[idx2: idx2 + 3] for idx2 in range(len_concat_total_url_trigram - 1))
    p_tri_gram = 1
    for u2 in res_trigram:
        if len(u2) == 3:
            p_tri_gram = p_tri_gram * dict_res2[u2]
    p_tri_gram = p_tri_gram * (len(concat_url) / len_concat_total_url * 100)
    decimal.getcontext().prec = 25  # Change 25 to the precision you want.
    p_tri_gram = decimal.Decimal(p_tri_gram) / decimal.Decimal(10)

    return p_tri_gram



def IPAddress(str):
    parsed_url = urllib.parse.urlparse(str)
    dom = parsed_url.netloc
    #print(str)
    #print(dom)
    #print("Hello World!")
    stri = dom
    #print(stri)
    #input data like www.pythonguides.com
    try:
        IP_addres = socket.gethostbyname(stri)

    #obj = IPWhois(IP_addres)
    #res = obj.lookup()
    #print(res)
    #asn_num = res['asn']
    #print(res['asn_country_code'])
    #asn_country = res["nets"][0]['country']
    #print(res["nets"][0]['address'])
    #print(res["nets"][0]['name'])
    #asn_cidr = res["nets"][0]['cidr']
    #print(res["nets"][0]['state'])
    #asn_postal_code = res["nets"][0]['postal_code']
    #created_date = res["nets"][0]['created']
    #updated_date = res["nets"][0]['updated']

        if IP_addres == None:
            return 0
        else:
            return 1
    except Exception as e:
        return 0
  

##def CheckEXE(str):
##    res = distutils.spawn.find_executable(str)
##    if res == None:
##        return 0
##    else:
##        return 1


def GoogleSearchFeature(selfwp):
    parsed_url = urllib.parse.urlparse(selfwp)
    hostname = parsed_url.netloc

    #parsedurl1 = tldextract.extract(selfwp)
    #dom = parsedurl1.domain
    #print(dom)

    count = 0
    num = 0
    ld = 0
    try:
        from googlesearch import search
    except ImportError:
        print("No module named 'google' found")

    # to search
    query = hostname
    print(hostname)

    for j in search(query, tld="com", num=10, stop=10, pause=3):
        # print(j)
        j = j.lower()
        if hostname in j:
            #count = j.count("geeks")
            count = count + 1
        if num == 0:
            #print("yes zero counted")
            ld = (enchant.utils.levenshtein(j, hostname))

        num = num + 1
    #print(count)
    #print(ld)
    arr = [ld, count]
    return arr


#these are new features 
def count_subdomains(url):
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Get the netloc (domain) part of the URL
    domain = parsed_url.netloc
    
    # Split the domain into subdomains
    subdomains = domain.split('.')
    
    # Exclude www as it's a common subdomain
    if subdomains[0].lower() == 'www':
        subdomains.pop(0)
    
    # Return the count of subdomains
    return len(subdomains)



def measure_subdirectory_depth(url):
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Get the path part of the URL
    path = parsed_url.path
    
    # Split the path into segments
    path_segments = path.split('/')
    
    # Remove any empty segments
    path_segments = [segment for segment in path_segments if segment]
    
    # Return the count of path segments (subdirectory depth)
    return len(path_segments)


import re

def detect_url_encoding(url):
    # Define a regular expression pattern to match URL-encoded characters
    url_encoded_pattern = r'%[0-9a-fA-F]{2}'
    
    # Find all URL-encoded character sequences in the URL
    url_encoded_matches = re.findall(url_encoded_pattern, url)
    
    # Check if any URL-encoded characters were found
    if url_encoded_matches:
        return 1
    else:
        return 0


def keyword_analysis(url):
    # Define a list of known phishing-related keywords or phrases
    phishing_keywords = ["login", "password", "account", "bank", "paypal", "secure", "verify", "update", "signin", "payment", 'bank', 'Bank', 'banking', 'architect', 'chemist', 'pharma', 'account',
                         'credit', 'transfer', 'allow','assure', 'government', 'organisation', 'fund', 'secure', 'confirm', 'Secure', 'Confirm', 'webscr',
              'login', 'Login', 'Log in', 'Log In', 'ebayisapi', 'sign in', 'Sign in', 'Sign In', 'sign up', 'Sign up',
              'Sign Up','trust', 'authority', 'offer', 'accept', 'Accept', 'admit', 'allow', 'cookies', 'Cookies',
              'safe', 'browse', 'fix', 'get', 'cash', 'credit', 'buy', 'purchase', 'coin', 'money', 'obtain', 'help',
              'connect', 'drug']
    
    # Compile a regular expression pattern to match any of the keywords
    keyword_pattern = re.compile(fr'\b({"|".join(map(re.escape, phishing_keywords))})\b', re.IGNORECASE)
    
    # Search for the keywords in the URL
    keyword_matches = keyword_pattern.search(url)
    
    # Check if any phishing-related keywords were found
    if keyword_matches:
        return 1
    else:
        return 0

def is_legitimate_tld(url, whitelist):
    # Parse the URL to get the TLD
    parts = url.split('.')
    tld = parts[-1].lower()

    tld=tld.upper()
    #print(tld)
    
    # Check if the TLD is in the whitelist
    if tld in whitelist:
        return 0
    else:
        return 1



whitelist = []

with open('C:\\Users\\ps\\Documents\\datasets\\whitlist1.txt', 'r',encoding='utf-8') as file:
      for line in file:
          whitelist.append(line)


whitelist=str(whitelist)      



def analyze_hyphen_distribution(url):
    # Count the number of hyphens in the URL
    hyphen_count = url.count('-')
    
    # Check if the hyphen count is above a certain threshold (adjust as needed)
    threshold = 3  # Adjust this threshold as needed
    if hyphen_count >= threshold:
        return 1
    else:
        return 0


from urllib.parse import urlparse, parse_qs

def has_unique_identifiers(url):
    # Parse the URL to extract query parameters
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    # Check if there are any query parameters
    if query_params:
        return 1
    else:
        return 0


def count_parameters(url):
    param_count=0
    # Parse the URL to extract query parameters
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    # Count the number of query parameters
    if query_params:
        param_count = len(query_params)
    
    return param_count

def contains_javascript(url):
    # Define a regular expression pattern to match JavaScript code or event handlers
    js_pattern = r'(javascript:|on\w+\s*=)'
    
    # Search for JavaScript patterns in the URL
    if re.search(js_pattern, url, re.IGNORECASE):
        return 1
    else:
        return 0


def contains_brand_keywords(url, brand_keywords):
    # Convert the URL to lowercase for case-insensitive matching
    url_lower = url.lower()
    
    # Check if any brand keywords are present in the URL
    for keyword in brand_keywords:
        if keyword.lower() in url_lower:
            return 0
    
    return 1

brand_keywords = brand_list = ["google.com", "facebook.com", "microsoft.com", "apple.com", "amazon.com", "twitter.com", "linkedin.com",
    "youtube.com", "wikipedia.org", "adobe.com", "ibm.com", "cnn.com", "bbc.com", "yahoo.com", "instagram.com", "netflix.com", "wordpress.com",
    "github.com", "stackoverflow.com"]


def contains_leetspeak(url):
    # Define a dictionary of leet-speak substitutions
    leet_dict = {
        'a': ['4', '@'],
        'b': ['8', '6'],
        'c': ['(', '{', '[', '<'],
        'e': ['3'],
        'g': ['9'],
        'h': ['#'],
        'i': ['1', '!', '|'],
        'l': ['1', '|'],
        'o': ['0'],
        's': ['5', '$'],
        't': ['7', '+'],
        'z': ['2'],
    }

    # Convert the URL to lowercase for case-insensitive matching
    url_lower = url.lower()

    # Check if any leet-speak substitutions are present in the URL
    for char, substitutions in leet_dict.items():
        for sub in substitutions:
            if sub in url_lower:
                return 1

    return 0


from urllib.parse import urlparse

def measure_path_length(url):
    try:
        # Parse the URL to extract its components
        parsed_url = urlparse(url)

        # Get the path component from the parsed URL
        path = parsed_url.path

        # Measure the length of the path
        path_length = len(path)

        return path_length
    except Exception as e:
        print(f"Error parsing URL: {e}")
        return 0


def remove_http_www(url):
    # Remove "http://www." if it exists
    url = re.sub(r'^http://www\.', '', url, flags=re.IGNORECASE)

    # Remove "http://" if it exists
    url = re.sub(r'^http://', '', url, flags=re.IGNORECASE)

    return url




def extract_http_method(url):
    try:
        # Parse the URL to extract the scheme and path
        parsed_url = urlparse(url)

        # Extract the HTTP method from the URL path (assumes it's the first part of the path)
        http_method = parsed_url.path.split('/')[1].upper()

        # List of common HTTP methods (customize as needed)
        common_http_methods = ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]

        # Check if the extracted HTTP method is unusual
        if http_method not in common_http_methods:
            return 1
        else:
            return 0
    except Exception as e:
        print(f"Error extracting HTTP method: {e}")
        return 0


import requests

def unwrap_shortened_url(shortened_url):
    try:
        # Send a HEAD request to the shortened URL to get the final destination
        response = requests.head(shortened_url, allow_redirects=True)

        # Extract the final URL after following redirects
        final_url = response.url

        return final_url
    except Exception as e:
        print(f"Error unwrapping URL: {e}")
        return None



import urllib.parse

def analyze_query_parameters(url, suspicious_keywords):
    try:
        # Parse the URL to extract query parameters
        parsed_url = urllib.parse.urlparse(url)
        query_parameters = urllib.parse.parse_qs(parsed_url.query)

        # Initialize a flag to track suspicious parameters
        has_suspicious_parameters = False

        # Iterate through the query parameters and their values
        for parameter, values in query_parameters.items():
            for value in values:
                # Check if the value contains any suspicious keywords
                for keyword in suspicious_keywords:
                    if keyword in value:
                        has_suspicious_parameters = True
                        print(f"Suspicious parameter: '{parameter}', Value: '{value}'")

        return has_suspicious_parameters
    except Exception as e:
        print(f"Error analyzing query parameters: {e}")
        return None


suspicious_keywords = ["admin", "password", "exploit"]





from pathlib import Path

def resource_type_detection(url, suspicious_extensions):
    try:
        # Parse the URL to extract the path component
        url_path = urllib.parse.urlparse(url).path

        # Extract the file extension from the URL path
        file_extension = Path(url_path).suffix

        # Remove the dot (.) from the file extension if present
        file_extension = file_extension.lstrip(".")

        # Check if the extracted file extension is in the list of suspicious extensions
        if file_extension.lower() in suspicious_extensions:
            return 0  # File extension is suspicious
        else:
            return 1  # File extension is not suspicious

    except Exception as e:
        print(f"Error performing resource type detection: {e}")
        return 0

suspicious_extensions = [
    "exe", "bat", "cmd", "vbs", "js", "jar", "ps1", "psm1", "wsf",  # Executable and script files
    "dll", "sys", "ocx", "drv", "scr",  # System and driver files
    "cpl", "app", "msp", "msi", "com",  # Installer and control panel files
    "sh", "bash", "zsh", "shar", "bsh",  # Shell script files
    "ade", "adp", "mdb", "accdb",  # Microsoft Access database files
    "vba", "vbe", "wsc", "ws", "wsf", "wsh",  # Windows Script Host files
    "hta", "chm", "hlp",  # HTML Help files
    "hta", "msc", "reg",  # Windows-specific script and configuration files
    "pif", "lnk", "scr",  # Shortcut and screen saver files
    "msp", "mst",  # Windows Installer patch files
    "torrent", "bittorrent", "magnet",  # Torrent and magnet link files
    "rar", "zip", "7z", "tar", "gz", "bz2", "xz",  # Archive and compressed files
    "iso", "img", "nrg", "bin", "cue", "dmg",  # Disk image files
    "swf", "fla",  # Flash files
    "woff", "ttf", "otf", "eot", "font", "fon",  # Font files
    "avi", "mpg", "mpeg", "mov", "mp4", "mkv",  # Video files
    "mp3", "wav", "flac", "ogg", "wma", "aac",  # Audio files
    "jpg", "jpeg", "png", "gif", "bmp", "svg", "ico",  # Image files
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "rtf",  # Document files
    "csv", "xml", "json",  # Data and configuration files
    "bak", "backup", "old", "temp", "tmp",  # Backup and temporary files
    "lock", "lck", "locky",  # Ransomware-related file extensions
    "phishing", "malware", "spyware", "trojan", "virus", "keylogger",  # Keywords related to malicious content
]



#html features
#Function to count number of images in the webpage Feature71
def ImgCount(html):
    soup = BeautifulSoup(html, "lxml")

    a = len(soup.find_all('img'))
    #b = (get_img_cnt(str))
    return a
    #print(b)

def TotalLinks(html):
    #working with url having https or full address
    soup = BeautifulSoup(html, "lxml")
    count = 0
    #urls = []
    for link in soup.find_all('a'):
        if link.get('href'):
            count = count + 1
    return count







# Function to check if the title tag is empty or not in HTML source code
def TitleCheck(html):
    soup = BeautifulSoup(html, "lxml")
    title = soup.title
    if title is not None:
        title = title.string  # Extract the title of the webpage
        return 1
    else:
        return 0


from bs4 import BeautifulSoup
import requests

# Function to check if a URL contains a "mailto" link
def CheckMailto(html):
    try:
            soup = BeautifulSoup(html, "lxml")   
            # Find all anchor (<a>) tags in the HTML
            anchor_tags = soup.find_all('a')
            
            # Check each anchor tag for a "mailto" link
            for tag in anchor_tags:
                href = tag.get('href', '')
                if href.startswith('mailto:'):
                    return True  # Found a "mailto" link
                    
        # If no "mailto" link was found or there was an error, return False
            return False
    
    except Exception as e:
        print(f"Error: {e}")
        return False






# Function to check if iframe or frame is used in HTML source code
def CheckIframeOrFrame(html):
    try:
        
            soup = BeautifulSoup(html, 'lxml')

            # Find all iframe and frame tags in the HTML
            iframe_tags = soup.find_all(['iframe', 'frame'])

            # Check if any iframe or frame tags were found
            if iframe_tags:
                return 0  # Found iframe or frame tags
            else:
                return 1

    except Exception as e:
        print(f"Error: {e}")
        return 1




# Function to check if HTML source code contains a JavaScript popup command
def CheckPopupCommands(html_source):
    try:
        # Regular expression pattern to match JavaScript popup commands
        popup_pattern = r'(window\.)?(alert|confirm|prompt)\s*\('

        # Search for the pattern in the HTML source code
        matches = re.findall(popup_pattern, html_source)

        # If matches are found, return True (popup command detected)
        if matches:
            return 1

        # If no matches are found, return False (no popup command detected)
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 0


#Function to find total number of query parameters in the URL Feature73

def NumParameters(self):
    params = self.split('&')
    a = len(params) - 1
    return a

#Function to find total number of fragments in the URL Feature74

def NumFragments(self):
    fragments = self.split('#')
    a = len(fragments) - 1
    return a



# Function to count the number of <body> tags in HTML source code
def CountBodyTags(html_source):
    try:
        # Create a BeautifulSoup object to parse the HTML source
        soup = BeautifulSoup(html_source, 'html.parser')

        # Find all <body> tags in the HTML
        body_tags = soup.find_all('body')

        # Count the number of <body> tags
        num_body_tags = len(body_tags)

        return num_body_tags

    except Exception as e:
        print(f"Error: {e}")
        return 0  # Return 0 in case of an error or no <body> tags found

from bs4 import BeautifulSoup

# Function to count the number of <meta> tags in HTML source code
def CountMetaTags(html_source):
    try:
        # Create a BeautifulSoup object to parse the HTML source
        soup = BeautifulSoup(html_source, 'html.parser')

        # Find all <meta> tags in the HTML
        meta_tags = soup.find_all('meta')

        # Count the number of <meta> tags
        num_meta_tags = len(meta_tags)

        return num_meta_tags

    except Exception as e:
        print(f"Error: {e}")
        return 0  # Return 0 in case of an error or no <meta> tags found


# Function to count the number of <div> tags in HTML source code
def CountDivTags(html_source):
    try:
        # Create a BeautifulSoup object to parse the HTML source
        soup = BeautifulSoup(html_source, 'html.parser')

        # Find all <div> tags in the HTML
        div_tags = soup.find_all('div')

        # Count the number of <div> tags
        num_div_tags = len(div_tags)

        return num_div_tags

    except Exception as e:
        print(f"Error: {e}")
        return 0  # Return 0 in case of an error or no <div> tags found




from bs4 import BeautifulSoup, Tag, NavigableString

# Function to measure the depth of nodes in the DOM tree
def MeasureNodeDepth(html_source):
    try:
        # Create a BeautifulSoup object to parse the HTML source
        soup = BeautifulSoup(html_source, 'html.parser')

        # Define a recursive function to calculate node depths
        def calculate_depth(node, depth):
            if isinstance(node, Tag):
                node_depth = depth
                for child in node.children:
                    child_depth = calculate_depth(child, depth + 1)
                    node_depth = max(node_depth, child_depth)
                return node_depth
            else:
                # Handle NavigableString (text) nodes
                return depth

        # Calculate depth for the entire DOM tree starting from the root
        root = soup.find('html')
        dom_depth = calculate_depth(root, 0)

        return dom_depth

    except Exception as e:
        print(f"Error: {e}")
        return 0  # Return 0 in case of an error





# Function to count the number of siblings for each node
def CountSiblings(html_source):
    try:
        # Create a BeautifulSoup object to parse the HTML source
        soup = BeautifulSoup(html_source, 'html.parser')

        # Define a recursive function to count siblings for each node
        def count_siblings(node):
            if node is None:
                return 0
            else:
                return 1 + count_siblings(node.find_next_sibling())

        # Find the root of the DOM tree (typically <html>)
        root = soup.find('html')

        # Create a dictionary to store the number of siblings for each element
        sibling_counts = {}

        # Iterate through all elements in the DOM tree
        for element in root.descendants:
            if element.name:
                sibling_counts[element.name] = count_siblings(element.find_previous_sibling())

        return sibling_counts

    except Exception as e:
        return {}





# Function to detect semantic HTML elements in HTML source code
def DetectSemanticElements(html_source):
    try:
        # Create a BeautifulSoup object to parse the HTML source
        soup = BeautifulSoup(html_source, 'html.parser')

        # Initialize counters for semantic elements
        heading_count = 0
        list_count = 0

        # Find all heading elements (h1 to h6)
        heading_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        heading_count = len(heading_elements)

        # Find all list elements (ul and ol)
        list_elements = soup.find_all(['ul', 'ol'])
        list_count = len(list_elements)

        return {
            'headings': heading_count,
            'lists': list_count
        }

    except Exception as e:
        print(f"Error: {e}")
        return {}


from urllib.parse import urlparse, urlunparse

def remove_www_from_url(url):
    # Parse the URL
    parsed_url = urlparse(url)

    # Check if the netloc (domain) starts with "www."
    if parsed_url.netloc.startswith("www."):
        # Remove "www." from the netloc
        netloc_without_www = parsed_url.netloc[len("www."):]
    else:
        # If "www." is not present, use the original netloc
        netloc_without_www = parsed_url.netloc

    # Reconstruct the URL without "www."
    updated_url = urlunparse((parsed_url.scheme, netloc_without_www, parsed_url.path, parsed_url.params, parsed_url.query, parsed_url.fragment))

    return updated_url



# Function to count internal and external links in an HTML document
def CountInternalAndExternalLinks(html_content, base_url):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        internal_link_count = 0
        external_link_count = 0
        link_ratio=0

        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            parsed_href = urlparse(full_url)
            parsed_base=urlparse(base_url)

            parsed_href = str(parsed_href.netloc)
            parsed_base=str(parsed_base.netloc)

            parsed_href=parsed_href.replace("www.", "")
            parsed_base=parsed_base.replace("www.", "")

##            print("parsed_href:", parsed_href)
##            print("urlparse(base_url):", parsed_base)
##
##            print("\n")
            if parsed_href:
                    if parsed_href == parsed_base:
                        internal_link_count += 1
                      
                    else:
                        external_link_count += 1
        if internal_link_count  > external_link_count:
            link_ratio =0
        else:
            link_ratio=1

        #return internal_link_count, external_link_count
        return {
            'internal_link_count': internal_link_count,
            'external_link_count': external_link_count,
            'link_ratio':link_ratio
                    }

    except Exception as e:
        print(f"Error: {e}")
        return 0, 0



def count_embedded_content(html_content):
    embedded_count = {
        'images': 0,
        'iframes': 0
    }

    soup = BeautifulSoup(html_content, 'html.parser')

    # Count images
    embedded_count['images'] = len(soup.find_all('img'))

    # Count iframes
    embedded_count['iframes'] = len(soup.find_all('iframe'))

    return embedded_count






def count_text_length(html_content, unit='characters'):
    # Initialize a variable to store the total text length
    total_length = 0

    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract plain text from the entire HTML content
    plain_text = soup.get_text()

    # Calculate text length
    if unit == 'characters':
        length = len(plain_text)
    elif unit == 'words':
        # Split text into words and count
        words = re.findall(r'\b\w+\b', plain_text)
        length = len(words)
    else:
        raise ValueError("Invalid unit. Use 'characters' or 'words'.")

    total_length += length

    return total_length







def calculate_keyword_density(html_content, keywords):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract text from HTML
    text = soup.get_text()

    # Tokenize the text into words
    words = re.findall(r'\w+', text.lower())

    # Initialize a dictionary to store keyword frequencies
    keyword_frequencies = {keyword: 0 for keyword in keywords}

    # Calculate the frequency of each keyword
    for word in words:
        for keyword in keywords:
            if keyword in word:
                keyword_frequencies[keyword] += 1

    # Calculate the total word count
    total_words = len(words)

    # Ensure that total_words is at least 1 to avoid division by zero
    total_words = max(total_words, 1)

    # Calculate keyword densities
    keyword_densities = {keyword: freq / total_words for keyword, freq in keyword_frequencies.items()}

    return keyword_densities



# List of target keywords related to phishing
target_keywords = ["login", "password", "account", "security", "verify", "update", "credit card", "bank", "paypal", "social security", "verify email", "suspicious activity", "win free", "urgent", "irs", "tax", "support", "reset password", "notification", "your account", "confirm"]


# List of target keywords related to phishing
target_keywords = ["login", "password", "account", "security", "verify", "update", "credit card", "bank", "paypal", "social security", "verify email", "suspicious activity", "win free", "urgent", "irs", "tax", "support", "reset password", "notification", "your account", "confirm"]






def count_link_types(html_content):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all anchor (<a>) elements
    links = soup.find_all('a', href=True)

    broken_links = 0
    empty_links = 0
    

    for link in links:
        href = link.get('href')

        # Check if the link is empty or consists of only whitespace
        if not href.strip():
            empty_links += 1

        # Check if the link is a hash link or contains 'javascript:void(0);'
        if href.startswith('#') or 'javascript:void(0);' in href:
            empty_links += 1

        # Check if the link has no scheme or netloc (considered broken)
        parsed_url = urlparse(href)
        if not parsed_url.scheme and not parsed_url.netloc:
            broken_links += 1

    return broken_links, empty_links



def count_open_graph_tags(html_content):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all meta tags with 'property' attribute starting with 'og:'
    meta_tags = soup.find_all('meta', attrs={'property': lambda x: x and x.startswith('og:')})

    return len(meta_tags)




def count_and_extract_form_elements(html_content):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all form elements in the HTML
    form_elements = soup.find_all('form')
    
    return len(form_elements)



from bs4 import BeautifulSoup
import urllib.parse

def analyze_form_action_urls(html_content, base_url):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all form elements in the HTML
    form_elements = soup.find_all('form')
    
    suspicious_destinations = []
    
    for form in form_elements:
        action_url = form.get('action', '')
        
        # Check if the action URL is empty
        if not action_url:
            continue
        
        # Parse and normalize the action URL
        action_url = urllib.parse.urljoin(base_url, action_url)
        
        # Check if the action URL is external (not part of the base URL)
        if not action_url.startswith(base_url):
            suspicious_destinations.append(action_url)
    
    return len(suspicious_destinations)





def count_hidden_form_fields(html_content):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all input elements with the type="hidden" attribute and count them
    count = len(soup.find_all('input', {'type': 'hidden'}))
    
    return count




# Function to extract inline JavaScript from a web page
def extract_inline_scripts(html):
    try:
      
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Find and extract inline JavaScript within <script> tags
        script_tags = soup.find_all('script')

        inline_scripts = []
        for script_tag in script_tags:
            # Check if the script is inline (no src attribute)
            if not script_tag.has_attr('src'):
                inline_scripts.append(script_tag.get_text())

        return inline_scripts

    except Exception as e:
        return []




# Function to extract the favicon URL from HTML source code
def extract_favicon_url(html_content, base_url):
    soup = BeautifulSoup(html_content, 'html.parser')
    favicon_tag = soup.find('link', rel='icon')  # Check for the standard favicon link tag
    if favicon_tag:
        favicon_url = favicon_tag.get('href')
        favicon_url = urljoin(base_url, favicon_url)  # Make the URL absolute if it's relative
        return favicon_url
    else:
        return None

# Function to verify the favicon URL matches the main domain
def verify_favicon_domain_match(favicon_url, main_domain):
    if favicon_url:
        favicon_domain = favicon_url.split('//')[1].split('/')[0]
        return favicon_domain == main_domain
    return False





def analyze_html_structure(html_content):
    # Parse the HTML content using Beautiful Soup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Count the total number of HTML tags
    total_tags = len(soup.find_all())

    # Identify commonly expected HTML tags for a basic structure
    expected_tags = ["html", "head", "title", "meta", "body", "h1", "p", "a"]

    # Count the number of times each expected tag appears
    tag_counts = {tag: len(soup.find_all(tag)) for tag in expected_tags}

    return total_tags, tag_counts



df = pd.DataFrame(columns=['id','url','URLLength1','DigitAlphabetRatio1','SpecialcharAlphabetRatio1','UppercaseLowercaseRatio1','DomainURLRatio1',
              'NumericCharCount1', 'EnglishLetterCount1','SpecialCharCount1','DotCount1','SemiColCount1',
              'UnderscoreCount1','QuesMarkCount1','HashCharCount1','EqualCount1','PercentCharCount1','AmpersandCount1'
              ,'DashCharCount1','DelimiterCount1','AtCharCount1','TildeCharCount1','DoubleSlashCount1','HostNameLength1','QueryLength1','HttpsInHostName1'
              ,'TLDInSubdomain1','TLDInPath1','HttpsInUrl1','IsDomainEnglishWord1','Unigram1','Bigram1','Trigram1','count_subdomains1',
              'measure_subdirectory_depth1','detect_url_encoding1', 'keyword_analysis1','is_legitimate_tld1','analyze_hyphen_distribution1',
              'has_unique_identifiers1','count_parameters1','contains_javascript1','contains_brand_keywords1','contains_leetspeak1','measure_path_length1'
              ,'resource_type_detection1', 'raw_word_count', 'avg_length', 'shortest_length','longest_length',
                'std_deviation', 'adjacent_word_count', 'avg_length', 'separated_count', 'count_random', 'lengths_Domain_Length',
                'lengths_Subdomain_Length','lengths_Path_Length',  'results_www_in_domain', 'results_com_in_domain',
                'results_www_in_subdomain', 'has_consecutive_char_repeats1','classes'])





def extract_raw_word_count(url):
    # Remove common URL prefixes and split the URL into words based on special characters
    words = re.split(r'[^\w-]+', url)

    # Count the number of words
    word_count = len(words)

    return word_count




def average_word_length_from_text(text, special_characters=" .,?!"):
    # Split the text into words using special characters as delimiters
    words = re.split("[" + re.escape(special_characters) + "]+", text)
    
    if len(words) == 0:
        return 0  # Avoid division by zero for empty lists

    total_length = sum(len(word) for word in words)
    return total_length / len(words)



def shortest_word_length_from_text(text, special_characters=" .,?!"):
    # Split the text into words using special characters as delimiters
    words = re.split("[" + re.escape(special_characters) + "]+", text)

    if len(words) == 0:
        return 0  # Return 0 for an empty list

    # Find the shortest word in terms of length
    shortest_word = min(words, key=len)

    return len(shortest_word)


def longest_word_length_from_text(text, special_characters=" .,?!"):
    # Split the text into words using special characters as delimiters
    words = re.split("[" + re.escape(special_characters) + "]+", text)

    if len(words) == 0:
        return 0  # Return 0 for an empty list

    # Find the longest word in terms of length
    longest_word = max(words, key=len)

    return len(longest_word)


import statistics
def word_length_standard_deviation(text, special_characters=" .,?!"):
    # Split the text into words using special characters as delimiters
    words = re.split("[" + re.escape(special_characters) + "]+", text)

    if len(words) < 2:
        return 0  # Return 0 for less than two words (no deviation)

    # Calculate word lengths
    word_lengths = [len(word) for word in words]

    # Calculate the standard deviation
    std_deviation = statistics.stdev(word_lengths)

    return std_deviation


def count_adjacent_words(text):
    # Define a regular expression to match words
    word_pattern = r'\w+'
    
    # Use the re.findall function to extract all words from the text
    words = re.findall(word_pattern, text)
    
    # Initialize a count variable
    count = 0
    
    # Iterate through the words and count adjacent words
    for i in range(len(words) - 1):
        if words[i] != '' and words[i+1] != '':
            count += 1

    return count


def calculate_adjacent_word_stats(text):
    # Define a regular expression to match words
    word_pattern = r'\w+'
    
    # Use the re.findall function to extract all words from the text
    words = re.findall(word_pattern, text)
    
    # Initialize variables for adjacent word statistics
    total_length = 0
    separated_word_count = 0
    
    # Iterate through the words and calculate statistics
    for i in range(len(words) - 1):
        if words[i] != '' and words[i+1] != '':
            total_length += len(words[i]) + len(words[i+1])
            separated_word_count += 2

    # Calculate the average adjacent word length
    average_length = total_length / (separated_word_count or 1)

    return average_length, separated_word_count



def random_word_count(url_text, min_word_length=3):
    # Split the URL text into words using non-alphanumeric characters as separators
    words = re.split(r'[^a-zA-Z0-9]+', url_text)

    # Initialize a count for random words
    random_word_count = 0

    # Iterate through the words and count those that meet the criteria
    for word in words:
        # Check if the word is longer than the specified minimum length
        if len(word) >= min_word_length:
            # Check if the word consists of random characters
            if re.match(r'^[A-Za-z]*$', word):
                random_word_count += 1

    return random_word_count


def calculate_url_lengths(url):
    # Parse the URL to extract its components
    parsed_url = urlparse(url)

    # Calculate the length of the domain
    domain_length = len(parsed_url.netloc)

    # Calculate the length of the subdomain
    subdomain_length = len(parsed_url.hostname.split('.')[0])

    # Calculate the length of the path
    path_length = len(parsed_url.path)

    return {
        "Domain Length": domain_length,
        "Subdomain Length": subdomain_length,
        "Path Length": path_length
    }


def contains_www_and_com(url):
    # Parse the URL to extract the domain and subdomain
    parsed_url = urlparse(url)
    
    # Regular expression patterns for "www" and "com"
    www_pattern = re.compile(r'www', re.IGNORECASE)
    com_pattern = re.compile(r'com', re.IGNORECASE)
    
    # Check if "www" and "com" are present in the domain or subdomain
    domain = parsed_url.netloc
    subdomain = parsed_url.hostname.split('.')[0]
    
    www_in_domain = bool(www_pattern.search(domain))
    com_in_domain = bool(com_pattern.search(domain))
    
    www_in_subdomain = bool(www_pattern.search(subdomain))
    com_in_subdomain = bool(com_pattern.search(subdomain))

    if www_in_domain:
        www_in_domai=1
    else:
        www_in_domai=0

    if com_in_domain:
        com_in_domai=1
    else:
        com_in_domai=0

    if www_in_subdomain:
        www_in_subdomai=1
    else:
        www_in_subdomai=0
    
    return {
        "www_in_domain": www_in_domai,
        "com_in_domain": com_in_domai,
        "www_in_subdomain": www_in_subdomai,
    }


def has_consecutive_char_repeats(url, max_repeat=3):
    # Define a regex pattern to find consecutive repeated characters.
    pattern = r'(\w)\1{' + str(max_repeat - 1) + r',}'
    
    # Search for consecutive repeated characters in the URL.
    match = re.search(pattern, url)
    
    if match:
        return 1
    else:
        return 0








#-----------------------------------------------------------------------------------------------------
##counter = 1
##for i in range(1, 15001):
#try:
       
            # Convert the counter to a string with leading zeros    
##        counter_str = f"{counter:05d}"
##                # Increment the counter
##        counter += 1


        
###html_data  phishingdata
##        item_path ="C:\\Users\\ps\\Documents\\datasets\\DatasetwithHTML\\html_data\\L"+counter_str+"u.text"
##        with open(item_path, 'rb') as file:
##            url = file.read().decode('utf-8', errors='ignore')
##
##        item_path2 ="C:\\Users\\ps\\Documents\\datasets\\DatasetwithHTML\\html_data\\L"+counter_str+"h.text"
##        with open(item_path2, 'rb') as file:
##            html = file.read().decode('utf-8', errors='ignore')



labels, texts=[],[]

file_path ='C:\\Users\\ps\\Documents\\datasets\\the third paper\\archive\\phishing_site_urls.csv' # Update to the actual CSV file path
        # Read the CSV file
df1 = pd.read_csv(file_path)
        # Check the structure of the CSV
print(f"Number of rows: {df.shape[0]}")
print(df.head())  # Check the first few rows to ensure proper formatting

classes=0
url=""
numeric_part=0
if 'URL' in df1.columns and 'Label' in df1.columns:
           # Process the CSV rows
    for index, row in df1.iterrows():
                texts.append(row['URL'])
                url=row['URL']
           # Convert 'bad' -> 1, 'good' -> 0
                label = row['Label'].strip().lower()
                if label == 'bad':
                  labels.append(1)
                  classes=1
                elif label == 'good':
                  labels.append(0)
                  classes=0
                else:
                  print(f"Invalid label at row {index}: {label}")
##else:
##            print("Error: CSV file does not contain 'URL' or 'Label' columns.")

##        # Print the size of the lists
##print(f"Number of texts: {len(texts)}")
##print(f"Number of labels: {len(labels)}")

##    for i in texts:
##        print (i)

##import random
##
### Ensure the lengths of texts and labels match
##assert len(texts) == len(labels), "Texts and labels must have the same length."
##
### Combine texts and labels for random sampling
##combined = list(zip(texts, labels))
##num_samples=100
##
### Randomly sample 10,000 pairs
##sampled = random.sample(combined, num_samples)
##
### Unzip the sampled data back into separate lists
##texts_subset, labels_subset = zip(*sampled)

### Convert back to lists (optional)
##texts_subset = list(texts_subset)
##labels_subset = list(labels_subset)
##
##print("texts_subset",texts_subset.shape)
##print("labels_subset",labels_subset.shape)
##
##for url, classes in zip(texts_subset, labels_subset):
                try:
                    #print (url,"  ",classes)
            
                    url=remove_http_www(url)
                    url="http://"+url
                    numeric_part+=1
                    URLLength1=URLLength(url)
                    DigitAlphabetRatio1=DigitAlphabetRatio(url)
                    SpecialcharAlphabetRatio1=SpecialcharAlphabetRatio(url)
                    UppercaseLowercaseRatio1=UppercaseLowercaseRatio(url)
                    DomainURLRatio1=DomainURLRatio(url)
                    NumericCharCount1=NumericCharCount(url)
                    EnglishLetterCount1=EnglishLetterCount(url)
                    SpecialCharCount1=SpecialCharCount(url)
                    DotCount1=DotCount(url)
                    SemiColCount1=SemiColCount(url)
                    UnderscoreCount1=UnderscoreCount(url)
                    QuesMarkCount1=QuesMarkCount(url)
                    HashCharCount1=HashCharCount(url)
                    EqualCount1=EqualCount(url)
                    PercentCharCount1=PercentCharCount(url)
                    AmpersandCount1=AmpersandCount(url)
                    DashCharCount1=DashCharCount(url)
                    DelimiterCount1=DelimiterCount(url)
                    AtCharCount1=AtCharCount(url)
                    TildeCharCount1=TildeCharCount(url)
                    DoubleSlashCount1=DoubleSlashCount(url)
                    HostNameLength1=HostNameLength(url)
                    PathLength1=PathLength(url)
                    QueryLength1=QueryLength(url)
                    HttpsInHostName1=HttpsInHostName(url)
                    TLDInSubdomain1=TLDInSubdomain(url)
                    TLDInPath1=TLDInPath(url)
                    HttpsInUrl1=HttpsInUrl(url)
                    #DistDigitAlphabet1=DistDigitAlphabet(url)
                    IsDomainEnglishWord1=IsDomainEnglishWord(url)
                    Unigram1=Unigram(url)
                    Bigram1=Bigram(url)
                    Trigram1=Trigram(url)
            ##        IPAddress1=IPAddress(url)
                    count_subdomains1=count_subdomains(url)
                    measure_subdirectory_depth1=measure_subdirectory_depth(url)
                    detect_url_encoding1=detect_url_encoding(url)
                    keyword_analysis1=keyword_analysis(url)
                    is_legitimate_tld1=is_legitimate_tld(url,whitelist)
                    analyze_hyphen_distribution1=analyze_hyphen_distribution(url)
                    has_unique_identifiers1=has_unique_identifiers(url)
                    count_parameters1=count_parameters(url)
                    contains_javascript1=contains_javascript(url)
                    contains_brand_keywords1=contains_brand_keywords(url,brand_keywords)
                    contains_leetspeak1=contains_leetspeak(url)
                    measure_path_length1=measure_path_length(url)
                    resource_type_detection1=resource_type_detection(url,suspicious_extensions)
                    
                    raw_word_count = extract_raw_word_count(url)
                    avg_length = average_word_length_from_text(url)
                    shortest_length = shortest_word_length_from_text(url)
                    longest_length = longest_word_length_from_text(url)
                    std_deviation = word_length_standard_deviation(url)
                    # Example usage:
                    adjacent_word_count = count_adjacent_words(url)

                    avg_length, separated_count = calculate_adjacent_word_stats(url)
                    count_random = random_word_count(url)
                    lengths = calculate_url_lengths(url)
                    results = contains_www_and_com(url)
                    has_consecutive_char_repeats1 = has_consecutive_char_repeats(url)

                    
                    df = df.append({'id':str(numeric_part),'url': str(url), 'URLLength1': URLLength1,'DigitAlphabetRatio1': DigitAlphabetRatio1,'SpecialcharAlphabetRatio1':SpecialcharAlphabetRatio1,
                            'UppercaseLowercaseRatio1':UppercaseLowercaseRatio1,'DomainURLRatio1':DomainURLRatio1, 'NumericCharCount1':NumericCharCount1, 'EnglishLetterCount1':EnglishLetterCount1,
                            'SpecialCharCount1':SpecialCharCount1,'DotCount1':DotCount1,'SemiColCount1':SemiColCount1,'UnderscoreCount1':UnderscoreCount1,'QuesMarkCount1':QuesMarkCount1,
                            'HashCharCount1':HashCharCount1,'EqualCount1':EqualCount1,'PercentCharCount1':PercentCharCount1,
                            'AmpersandCount1':AmpersandCount1,'DashCharCount1':DashCharCount1,'DelimiterCount1':DelimiterCount1,
                            'AtCharCount1':AtCharCount1,'TildeCharCount1':TildeCharCount1,'DoubleSlashCount1':DoubleSlashCount1,
                            'HostNameLength1':HostNameLength1,'QueryLength1':QueryLength1,'HttpsInHostName1':HttpsInHostName1
                          ,'TLDInSubdomain1':TLDInSubdomain1,'TLDInPath1':TLDInPath1,'HttpsInUrl1':HttpsInUrl1,
                            'IsDomainEnglishWord1':IsDomainEnglishWord1,'Unigram1':Unigram1,'Bigram1':Bigram1,
                            'Trigram1':Trigram1,'count_subdomains1':count_subdomains1,
                          'measure_subdirectory_depth1':measure_subdirectory_depth1,'detect_url_encoding1':detect_url_encoding1,
                            'keyword_analysis1':keyword_analysis1,'is_legitimate_tld1':is_legitimate_tld1,'analyze_hyphen_distribution1':analyze_hyphen_distribution1,
                          'has_unique_identifiers1':has_unique_identifiers1,'count_parameters1':count_parameters1,
                            'contains_javascript1':contains_javascript1,'contains_brand_keywords1':contains_brand_keywords1,
                            'contains_leetspeak1':contains_leetspeak1,'measure_path_length1':measure_path_length1
                          ,'resource_type_detection1':resource_type_detection1, 'raw_word_count':raw_word_count , 'avg_length':avg_length ,'shortest_length': shortest_length,'longest_length':longest_length,'std_deviation':
                                    std_deviation,'adjacent_word_count':adjacent_word_count, 'avg_length': avg_length, 'separated_count':separated_count,
                                    'count_random':count_random, 'lengths_Domain_Length': lengths['Domain Length'], 'lengths_Subdomain_Length':lengths['Subdomain Length'],
                                    'lengths_Path_Length': lengths['Path Length'], 'results_www_in_domain':results['www_in_domain'], 'results_com_in_domain': results['com_in_domain'],
                                    'results_www_in_subdomain': results['www_in_subdomain'],'has_consecutive_char_repeats1':has_consecutive_char_repeats1,'classes': str(classes)}, ignore_index=True)
##                    
                
##
##                    print(numeric_part, URLLength1,"|",DigitAlphabetRatio1,"|",SpecialcharAlphabetRatio1,"|",UppercaseLowercaseRatio1,"|",DomainURLRatio1,"|"
##                          ,NumericCharCount1,"|", EnglishLetterCount1,"|",SpecialCharCount1,"|",DotCount1,"|",SemiColCount1,"|",
##                          UnderscoreCount1,"|",QuesMarkCount1,"|",HashCharCount1,"|",EqualCount1,"|",PercentCharCount1,"|",AmpersandCount1,
##                          "|",DashCharCount1,"|",DelimiterCount1,"|",AtCharCount1,"|",TildeCharCount1,"|",DoubleSlashCount1,"|",HostNameLength1,"|",QueryLength1,"|",HttpsInHostName1
##                          ,"|",TLDInSubdomain1,"|",TLDInPath1,"|",HttpsInUrl1,"|",IsDomainEnglishWord1,"|",Unigram1,"|",Bigram1,"|",Trigram1,"|",count_subdomains1,
##                          "|",measure_subdirectory_depth1,"|",detect_url_encoding1,"|", keyword_analysis1,"|",is_legitimate_tld1,"|",analyze_hyphen_distribution1,"|",
##                          has_unique_identifiers1,"|",count_parameters1,"|",contains_javascript1,"|",contains_brand_keywords1,"|",contains_leetspeak1,"|",measure_path_length1
##                          ,"|",resource_type_detection1,"|", raw_word_count,"|", avg_length,"|",shortest_length,"|",longest_length,"|",std_deviation,"|",adjacent_word_count,"|",avg_length
##                          ,"|", separated_count,"|",count_random,"|",lengths['Domain Length'],"|",lengths['Subdomain Length'],"|", lengths['Path Length']
##                          , results['www_in_domain'], "|", results['com_in_domain'], "|", results['www_in_subdomain'], "|",has_consecutive_char_repeats1, classes)
##
##                    #print('\n')
        
        

                except Exception as e: 
                        print(f"An error occurred: {str(e)}") 

df.to_csv( r'C:\Users\ps\Documents\datasets\output.csv', index=False)



                   
