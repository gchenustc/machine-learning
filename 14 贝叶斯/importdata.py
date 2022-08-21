import os
import numpy as np
from email.parser import Parser
from bs4 import BeautifulSoup
import re
from html import unescape
import pickle
import enchant
wordsTable = enchant.Dict("en_US")

def getTextPaths(filepath, current_dir, sep='/', datasets_dir='data/trec06p'):
    text = open(filepath, "r", encoding="gb2312", errors='ignore').read()

    info_list = re.split(r"\n",text)
    info_list_ = [x.split() for x in info_list if len(x.split()) == 2]

    labels = [x[0] for x in info_list_]
    textpaths = [current_dir + sep + x[1].replace('..', datasets_dir) for x in info_list_]

    return labels,textpaths


def getText(filepath):
    mail_ori = open(filepath, 'r', encoding='gb2312', errors='ignore').read() # 读取内容
    email = Parser().parsestr(mail_ori) # 获得邮件类
    type_ = email.get_content_maintype() # 获得邮件类型
    data = None
    if type_ == "multipart":
        for part in email.get_payload():
            if type(part) == str: # bug
                return None
            part_charset = part.get_content_charset()
            part_type = part.get_content_type()
            if part_type in ["text/plain", 'text/html', "text"]:
                data=part.get_payload(decode=True)
                try:
                    data=data.decode(part_charset,errors="replace")
                except:
                    data=data.decode('gb2312',errors="replace")

    elif type_ in ["text/plain", 'text/html', 'text']:
        part_charset = email.get_content_charset()
        data=email.get_payload(decode=True)
        try:
            data=data.decode(part_charset,errors="replace")
        except:
            data=data.decode('gb2312',errors="replace")

    if data: 
        return data

    else: # 如果非文本格式，返回 None
        return None   


def textParse(inputing_string):
    if inputing_string is None:
        return None
    data_proc = html_to_plain_text(inputing_string)  # html2str
    data_proc = re.sub(r'\$[\d]*',' dollardollar ',data_proc) # 替换$\d*为" dollardollar "
    data_proc = re.sub(r'\w',lambda x:x.group().lower(),data_proc) # 把字母换成小写
    words_list = re.findall(r'[A-Za-z]+', data_proc)  # 提取单词
    words_list = [word for word in words_list if wordsTable.check(word) and len(word)>2] # 检测是否为有效单词
    return words_list
        

def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', ' ', html, flags=re.M | re.S | re.I)
    text = re.sub(r'<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', ' ', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)


def pickleDump(filename, *datas):
    with open(filename, 'wb') as file:
        for data in datas:
            pickle.dump(data, file)


def importDataset():
    sep = '/'
    current_dir = os.path.dirname(__file__)
    datasets_save_dir = current_dir + sep + "datasets"
    filepath = f"{current_dir}/data/trec06p/full/index"

    if not os.path.exists(datasets_save_dir):
        labels, textpaths = getTextPaths(filepath, current_dir, sep)

        labels = np.array(labels[:2000], dtype=object)
        doclist = np.array([textParse(getText(path)) for path in textpaths[:2000]],dtype=object)
        labels = labels[doclist!=None]
        doclist = doclist[doclist!=None]

        datasets = {'data': doclist, 'labels': labels}
        # 泡菜打包
        pickleDump(datasets_save_dir,datasets)