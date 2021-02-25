import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from time import sleep

header = ['Volume_CE','OI_CE','changeOI_CE','IV_CE','netChange_CE','LTP_CE','strikePrice','LTP_PE','netChange_PE','IV_PE','changeOI_PE','OI_PE','Volume_PE']
count = 0
timestamp = ''
temp = {}

def get_Data(expirydate="9APR2020"):
    global count
    while(True):
        urlheader = {
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36",
          "authority": "www.nseindia.com",
          "scheme":"https"
        }
        url="https://www.nseindia.com/live_market/dynaContent/live_watch/option_chain/optionKeys.jsp?"
        params="segmentLink=17&instrument=OPTIDX&symbol=NIFTY&date="
        url_encoded=url + params + expirydate
       
        req = requests.get(url_encoded, headers=urlheader)
        soup = BeautifulSoup(req.content, "lxml")
        table = soup.find('table', id="octable")
        
        rows = table.findAll('tr')
        header_text = []
        headers = rows[1]  
        # removing columns from call and put dataframe which are not needed such as Bid Qty, Bid Price, Ask Price etc..    remove_indices_for_put = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,18,22]
        remove_indices_for_call = [0,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22]
        remove_indices_for_put = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,22]
             
            
            
        # add the header text to array
        for th in headers.findAll('th'):
            header_text.append(th.text)
            
        header_text = [i for j, i in enumerate(header_text) if j not in remove_indices_for_call]
        header_text.reverse()
        df_call = pd.DataFrame(columns=header_text)
        df_put = pd.DataFrame(columns=header_text)
        
        #row_text_array = []
        for row in rows[2:-1]:
            row_text = []
            row_text_call = []
            row_text_put = []
            # loop through the elements
            for row_element in row.findAll(['th', 'td']):
                # append the array with the elements inner text
                row_text.append(row_element.text.replace('\n', '').strip())
        
            # append the text array to the row text array
            row_text_put = [i for j, i in enumerate(row_text) if j not in remove_indices_for_put]
            for i in range(len(row_text_put)):
                if (row_text_put[i]=='-'):
                    row_text_put[i]=0
                else:
                    row_text_put[i]=row_text_put[i].replace(',', '')
                    row_text_put[i]=float(row_text_put[i])
            row_text_call = [i for j, i in enumerate(row_text) if j not in remove_indices_for_call]
            row_call_temp=[None] * len(row_text_call)
            for i in range(len(row_text_call)):       
                if (row_text_call[i]=='-'):
                    row_call_temp[len(row_text_call)-i-1]=0
                else:
                    row_text_call[i]=row_text_call[i].replace(',', '')
                    row_text_call[i]=float(row_text_call[i])
                    row_call_temp[len(row_text_call)-i-1]=row_text_call[i]
            df_call = df_call.append(pd.Series(dict(zip(df_call.columns, row_call_temp))), ignore_index=True)   
            df_put = df_put.append(pd.Series(dict(zip(df_put.columns, row_text_put))), ignore_index=True)
            
        count += 1
        save_Data(df_call,df_put,expirydate)
        print('recording',datetime.now())
        sleep(30)
        
    return df_call,df_put

def save_Data(df_call, df_put,expirydate):
    df_call.columns = [['strikePrice','netChange_CE','LTP_CE','IV_CE','Volume_CE','changeOI_CE','OI_CE']]
    df_put.columns =[['strikePrice','netChange_PE','LTP_PE','IV_PE','Volume_PE','changeOI_PE','OI_PE']]
    df = pd.concat([df_call,df_put.drop('strikePrice', axis=1)], axis=1)
    df = df[header]
    global temp
    
    if(temp != df.to_dict()):
        if(count == 1):
            df.to_csv(expirydate+'.csv', mode='a', header=True)
            #dat.to_csv('ratios.csv', mode='a', header = True)
        else:
            df.to_csv(expirydate+'.csv', mode='a', header=False)
            #dat.to_csv('ratios.csv', mode='a', header = False)
        
    temp = df.to_dict() 
    
get_Data('16APR2020')

    