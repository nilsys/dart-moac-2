import json
import datetime

import pandas as pd
import time as _time
import requests
import numpy as np
import boto3
import datetime
from dateutil import tz
from scipy.stats import linregress
import io

from talib import SMA,ATR

from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
import os
# from finta import TA
# import math

#s3://dart-moac/currentPortfolio.csv
#Todo check index list from a csv and exclude if not in nindex

#Global variables
folderPath = "/home/swabha/Tech/MoAC_Source_10M"
s3bucket = "dart-moac"
riskFactor = .0015
# investmentFund = 200000.00

# from yahoo_finance_api import YahooFinance as yf
pd.set_option('display.max_columns', None)
pd.options.display.max_rows = 2000

class YahooFinance:
    def __init__(self, ticker, result_range='1mo', start=None, end=None, interval='15m', dropna=True):
        """
        Return the stock data of the specified range and interval
        Note - Either Use result_range parameter or use start and end
        Note - Intraday data cannot extend last 60 days
        :param ticker:  Trading Symbol of the stock should correspond with yahoo website
        :param result_range: Valid Ranges "1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"
        :param start: Starting Date
        :param end: End Date
        :param interval:Valid Intervals - Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
        :return:
        """
        if result_range is None:
            start = int(_time.mktime(_time.strptime(start, '%d-%m-%Y')))
            end = int(_time.mktime(_time.strptime(end, '%d-%m-%Y')))
            # defining a params dict for the parameters to be sent to the API
            params = {'period1': start, 'period2': end, 'interval': interval}

        else:
            params = {'range': result_range, 'interval': interval}

        # sending get request and saving the response as response object
        url = "https://query1.finance.yahoo.com/v8/finance/chart/{}".format(ticker)
        r = requests.get(url=url, params=params)
        data = r.json()
        # Getting data from json
        error = data['chart']['error']
        if error:
            raise ValueError(error['description'])
        self._result = self._parsing_json(data)
        if dropna:
            self._result.dropna(inplace=True)

    @property
    def result(self):
        return self._result

    def _parsing_json(self, data):
        timestamps = data['chart']['result'][0]['timestamp']
        # Formatting date from epoch to local time
        timestamps = [_time.strftime('%a, %d %b %Y %H:%M:%S', _time.gmtime(x)) for x in timestamps]
        volumes = data['chart']['result'][0]['indicators']['quote'][0]['volume']
        opens = data['chart']['result'][0]['indicators']['quote'][0]['open']
        opens = self._round_of_list(opens)
        closes = data['chart']['result'][0]['indicators']['quote'][0]['close']
        closes = self._round_of_list(closes)
        lows = data['chart']['result'][0]['indicators']['quote'][0]['low']
        lows = self._round_of_list(lows)
        highs = data['chart']['result'][0]['indicators']['quote'][0]['high']
        highs = self._round_of_list(highs)
        df_dict = {'Open': opens, 'High': highs, 'Low': lows, 'Close': closes, 'Volume': volumes}
        df = pd.DataFrame(df_dict, index=timestamps)
        df.index = pd.to_datetime(df.index)
        return df

    def _round_of_list(self, xlist):
        temp_list = []
        for x in xlist:
            if isinstance(x, float):
                temp_list.append(round(x, 2))
            else:
                temp_list.append(pd.np.nan)
        return temp_list

    def to_csv(self, file_name):
        self.result.to_csv(file_name)


def fetch_data(symbol, period, interval):
    pd.options.display.max_rows = 2000
    fin_prod_data = YahooFinance(symbol, result_range=period, interval=interval).result
    return fin_prod_data

def calcMomentum(returns):
    x = np.arange(1,len(returns) +1)
    slope, _, rvalue, _, _ = linregress(x, returns)
    radjustedslope = ((1 + slope) ** 252) * (rvalue ** 2)
    return [slope, rvalue, radjustedslope] # annualize slope and multiply by R^2

def takeBackupS3(runReport):
    s3 = boto3.resource('s3')
    copy_source = {
        'Bucket': s3bucket,
        'Key': '{0}/currentPortfolio.csv'.format(runReport)
    }
    s3.meta.client.copy(copy_source, s3bucket, '{0}/previousPortfolio.csv'.format(runReport))


    s3 = boto3.resource('s3')
    copy_source = {
        'Bucket': s3bucket,
        'Key': '{0}/currentRanking.csv'.format(runReport)
    }
    s3.meta.client.copy(copy_source, s3bucket, '{0}/previousRanking.csv'.format(runReport))

def calcIndex200():
    period='6mo'
    indexContent = fetch_data('^CRSLDX', '1y', '1d')
    indexcloses = indexContent['Close'].values
    indexsma = SMA(indexcloses, timeperiod=200)
    indexsma = np.round(indexsma,2)
    indexsma = indexsma[-1]
    indexLastClose = indexcloses[-1]

    if indexLastClose < indexsma:
        isIndexAbove200 = 0

    else:
        isIndexAbove200 = 1

    return isIndexAbove200

def getPastPortfolio(runReport):
    # print(folderPath)

    # Read ignore data
    # This is current at the start - as of last week
    currePortfolioFile = '{0}/currentPortfolio.csv'.format(runReport)
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=s3bucket, Key=currePortfolioFile)
    # pastPortFolioString = response['Body'].read().decode('utf-8')
    pastPortFolio = pd.read_csv(io.BytesIO(response['Body'].read()))
    # pastPortFolio = pastPortFolio.split("\n")


    # pastPortFolio = pd.read_csv('{0}/currentPortfolio.csv'.format(folderPath))
    pastPortFolio.reset_index(inplace=True)
    pastPortFolio = pastPortFolio.drop(columns=['index'])
    return pastPortFolio

# def getPortfolioValue(pastPortFolio,currentRanking):
#     portfolioValue = 0
#     for portfolioStock in pastPortFolio['Stock']:
#         portfolioStockClose = currentRanking[currentRanking['Stock'] == portfolioStock]['Close'].values[0]
#         portfolioStockCount = pastPortFolio[pastPortFolio['Stock'] == portfolioStock]['numberofStocks'].values[0]
#         portfolioValue = portfolioValue + (portfolioStockClose * portfolioStockCount)
#     return portfolioValue
def getCurrentRanking(isIndexAbove200):

    period='6mo'

    index_list_file = 'index_list.txt'
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=s3bucket, Key=index_list_file)
    indexList = response['Body'].read().decode('utf-8')
    indexList = indexList.replace(" ","")
    indexList = indexList.split("\n")
    indexList = list(filter(None, indexList))
    currentRankingDf = pd.DataFrame(columns=['Stock','Close','Slope','RValue','RAdjustedSlope','100SMA','isAbove100SMA','ATR','maxGap','numberofStocks','cost','targetPercentage','isIndexAbove200'])

    # if indexLastClose < indexsma:
    #     isIndexAbove200 = 0
    #     return isIndexAbove200, indexContent, currentRankingDf
    # else:
    #     isIndexAbove200 = 1

    for symbol in indexList:
        content = fetch_data(symbol, period, '1d')
        content = content.tz_localize(tz='Etc/UTC')
        content = content.tz_convert(tz='Asia/Kolkata')

        # USe 6mo data to calculate 100 SMA
        closes = content['Close'].values
        sma = SMA(closes, timeperiod=100)
        sma = np.round(sma, 2)
        se = pd.Series(sma)
        content["100SMA"] = se.values

        content = content.tail(90)
        # Overwrite closes to the 90 limit to use in slope
        closes = content['Close'].values

        # content = content.head(90/)
        content.reset_index(inplace=True)
        content.rename(columns={'index': 'startTime'}, inplace=True)

        # hma = myHMACalc(content, 21)
        # content['21HMA'] = hma.values
        # ohlc = content["close"].resample("1h").ohlc()
        # print("Dummy")
        content['Symbol'] = symbol

        returns = np.log(closes)
        content['logClose'] = returns

        content['prevClose'] = np.nan
        for indexval, row in content.iterrows():
            content['prevClose'] = content['Close'] - content['Close'].diff(periods=1)
            content['gapPercent'] = abs((content['Open'] - content['prevClose']) * 100 / content['prevClose'])
        # content.to_csv('/home/swabha/Tech/SOTM_Start/{0}.csv'.format(symbol.split(".")[0]))

        momentum = calcMomentum(returns)
        lastClose = closes[-1]
        last100sma = content["100SMA"].values[-1]

        slope = momentum[0]
        rvalue = momentum[1]
        radjustedslope = momentum[2]
        lastATR = ATR(content['High'].values, content['Low'].values, content['Close'].values, timeperiod=20)
        lastATR = lastATR[-1]
        maxGap = content['gapPercent'].max()

        if lastClose >= last100sma:
            isAbove100SMA = 1
        else:
            isAbove100SMA = 0

        # use a dummyvalue of number of stocks = 0 for all
        # repopulate after portfolio value
        numberofStocks = 0
        cost = 0.0
        targetPercentage = 0.0

        # numberofStocks = round(investmentFund * riskFactor / lastATR, 0)
        # if numberofStocks == 0:
        #     numberofStocks = 1
        # cost = numberofStocks * lastClose
        # targetPercentage = cost * 100 / investmentFund

        tempDf = pd.DataFrame([[symbol, lastClose, slope, rvalue, radjustedslope, last100sma, isAbove100SMA, lastATR,
                                maxGap, numberofStocks, cost, targetPercentage, isIndexAbove200]],
                              columns=['Stock', 'Close', 'Slope', 'RValue', 'RAdjustedSlope', '100SMA', 'isAbove100SMA',
                                       'ATR', 'maxGap', 'numberofStocks', 'cost', 'targetPercentage',
                                       'isIndexAbove200'])
        currentRankingDf = currentRankingDf.append(tempDf, ignore_index=True)
        print(symbol)

    return indexList, currentRankingDf



def getCurrentPortfolio(pastPortFolio,currentRanking):
    currentPortfolio = pd.DataFrame(columns=pastPortFolio.columns)
    currentPortfolio = currentPortfolio.append(pastPortFolio)
    # update the close value of currentportfolio to current Close and make cost and percentage 0
    for currentPortfolioStock in currentPortfolio['Stock'].values:
        currentPortfolioStockClose = currentRanking[currentRanking['Stock'] == currentPortfolioStock]['Close'].values[0]
        currentPortfolio.loc[currentPortfolio['Stock'] == currentPortfolioStock, 'Close'] = currentPortfolioStockClose

        # currentPortfolio[currentPortfolio[currentPortfolio['Stock'] == currentPortfolioStock.index]]['Close'] = \
        # currentRanking[currentRanking['Stock'] == currentPortfolioStock]['Close']
    currentPortfolio['cost'] = 0.0
    currentPortfolio['targetPercentage'] = 0.0
    portfolioValue = (currentPortfolio['Close'] * currentPortfolio['numberofStocks']).sum()
    return currentPortfolio, portfolioValue

# This is for calculating number of stocks based on the portfolio value
def calcNumOfStock(portfolioValue,currentRanking):
    currentRanking['numberofStocks'] = round(portfolioValue * riskFactor / currentRanking['ATR'] , 0)
    currentRanking['cost'] = currentRanking['numberofStocks'] * currentRanking['Close']
    currentRanking['targetPercentage'] = currentRanking['cost'] * 100 / portfolioValue
    return currentRanking


def getExitStocks(pastPortFolio,currentRanking):
    exitCost = 0
    exitStocks = pd.DataFrame(columns=["Stock","exitCause","stockCount","exitValue"])
    for portfolioStock in pastPortFolio['Stock']:
        isAbove100SMA = currentRanking[currentRanking['Stock'] == portfolioStock]['isAbove100SMA'].values[0]
        maxGap = currentRanking[currentRanking['Stock'] == portfolioStock]['maxGap'].values[0]
        stockCurrentRank = (currentRanking[currentRanking['Stock'] == portfolioStock]).index[0]
        exitStockClose = currentRanking[currentRanking['Stock'] == portfolioStock]['Close'].values[0]
        portfolioStockCount = pastPortFolio[pastPortFolio['Stock'] == portfolioStock]['numberofStocks'].values[0]
        if maxGap > 15 :

            exitCost = exitCost + (exitStockClose * portfolioStockCount)
            exitStocks.loc[len(exitStocks)] = [portfolioStock, "MAXGAP", portfolioStockCount,exitCost ]
        elif isAbove100SMA == 0:

            exitCost = exitCost + (exitStockClose * portfolioStockCount)
            exitStocks.loc[len(exitStocks)] = [portfolioStock, "100SMA", portfolioStockCount,exitCost]

        elif stockCurrentRank > 99:

            exitCost = exitCost + (exitStockClose * portfolioStockCount)
            exitStocks.loc[len(exitStocks)] = [portfolioStock, "RANK", portfolioStockCount,exitCost]

    return exitStocks, exitCost

def portfolioRebalance(portfolioValue, currentInvEligible, currentPortfolio, exitCost):
    newBuy = currentInvEligible
    newBuy = newBuy[~newBuy['Stock'].isin(currentPortfolio['Stock'])]
    newBuy.reset_index(inplace=True)
    newBuy = newBuy.drop(columns=['index'])
    newBuy['cumCost'] = 0
    for indexval, row in newBuy.iterrows():
        if indexval == 0:
            newBuy.loc[indexval,'cumCost'] = newBuy.loc[indexval,'cost']
        elif newBuy.loc[(indexval -1),'cumCost'] <= exitCost:
            newCumCost = row['cost'] + newBuy.loc[(indexval-1),'cumCost']
            if (newCumCost - exitCost) < 1.5 * portfolioValue / 100:
                newBuy.loc[indexval,'cumCost'] = newCumCost
            else:
                break
        else:
            break
    newBuy = newBuy[newBuy['cumCost'] != 0 ]

    newNet = newBuy.iloc[-1,:]['cumCost'] - exitCost
    newBuyPrintOutout = newBuy
    newBuyPrintOutout = newBuyPrintOutout.drop(columns=["Slope","RValue","RAdjustedSlope","100SMA","isAbove100SMA","ATR","maxGap","isIndexAbove200","targetPercentage"])
    return newBuy, newNet,newBuyPrintOutout



def positionRebalance(newNet, currentPortfolio, currentInvEligible):
    positionRebalanceTrades = pd.DataFrame(columns=['Stock','portfolioNumberofStocks','currentNumOfStocks','Close'])
    positionRebalanceNet = 0

    for portfolioStock in currentPortfolio['Stock']:
        portfolioNumberofStocks = currentPortfolio[currentPortfolio['Stock'] == portfolioStock]['numberofStocks'].values[0]
        currentNumOfStocks = currentInvEligible[currentInvEligible['Stock'] == portfolioStock]['numberofStocks'].values[0]
        portfolioStockClose = currentInvEligible[currentInvEligible['Stock'] == portfolioStock]['Close'].values[0]
        if abs(100 * (currentNumOfStocks - portfolioNumberofStocks) / portfolioNumberofStocks) > 10.0:
            positionRebalanceTrades.loc[len(positionRebalanceTrades)] = [portfolioStock,portfolioNumberofStocks,currentNumOfStocks,portfolioStockClose]
            positionRebalanceNet = positionRebalanceNet + ((currentNumOfStocks - portfolioNumberofStocks) * portfolioStockClose)

    actualNetToReallocateRatio = 0
    # netAdjustment = newNet + positionRebalanceNet
    #
    # #Below block adjust all postion reblance proportionately by cost, so readjustment is minimal
    # if positionRebalanceNet != 0:
    #     totalPositionReblancePostionCost = (positionRebalanceTrades['currentNumOfStocks'] * positionRebalanceTrades['Close']).sum()
    #     actualNetToReallocateRatio = (totalPositionReblancePostionCost - netAdjustment)  / totalPositionReblancePostionCost
    #     positionRebalanceTrades['currentNumOfStocks'] = round((actualNetToReallocateRatio * positionRebalanceTrades['currentNumOfStocks']),0)
    #     positionRebalanceNet = ((positionRebalanceTrades['currentNumOfStocks'] - positionRebalanceTrades['portfolioNumberofStocks']) * positionRebalanceTrades['Close']).sum()



    return positionRebalanceTrades, positionRebalanceNet, actualNetToReallocateRatio

def createCsv(currentRanking,currentPortfolio,runReport):

    india_tz= tz.gettz('Asia/Kolkata')
    dtString = datetime.datetime.now(tz=india_tz).strftime("%d-%m-%Y")
    portfolioFileName = 'currentPortfolio'
    rankingFileName = 'currentRanking'
    # print(outputDf)
    tmpPortfolioPath = '/tmp/{0}.csv'.format(portfolioFileName)
    tmpRankingPath = '/tmp/{0}.csv'.format(rankingFileName)
    tmpPortfolioPathDt = '/tmp/{0}_{1}.csv'.format(portfolioFileName,dtString)
    tmpRankingPathDt = '/tmp/{0}_{1}.csv'.format(rankingFileName,dtString)


    currentRanking.to_csv(tmpRankingPath,index=False)
    currentRanking.to_csv(tmpRankingPathDt,index=False)
    currentPortfolio.to_csv(tmpPortfolioPath,index=False)
    currentPortfolio.to_csv(tmpPortfolioPathDt,index=False)

    s3_client = boto3.client('s3')
    s3_client.upload_file(tmpPortfolioPath, s3bucket, '{0}/{1}.csv'.format(runReport,portfolioFileName))
    s3_client.upload_file(tmpRankingPath, s3bucket, '{0}/{1}.csv'.format(runReport,rankingFileName))
    s3_client.upload_file(tmpPortfolioPathDt, s3bucket, '{0}/{1}_{2}.csv'.format(runReport,portfolioFileName,dtString))
    s3_client.upload_file(tmpRankingPathDt, s3bucket, '{0}/{1}_{2}.csv'.format(runReport,rankingFileName,dtString))

    return tmpPortfolioPath, tmpRankingPath


def getMailBody(portfolioValue,exitStocks,exitCost,newBuyPrintOutout,newNet,positionRebalanceTrades,positionRebalanceNet,currentPortfolio,actualNetToReallocateRatio):
    BODY_TEXT = "portfolio value as on {0} : ".format(portfolioValue)
    BODY_TEXT = BODY_TEXT + "\n\n" + "======================="
    BODY_TEXT = BODY_TEXT + "\n\n" + "Excluded list of stocks is"
    BODY_TEXT = BODY_TEXT + "\n\n" + exitStocks.to_string()
    BODY_TEXT = BODY_TEXT + "\n\n" + "Total Exit value to be added back is {0}".format(exitCost)
    BODY_TEXT = BODY_TEXT + "\n\n" + "======================="
    BODY_TEXT = BODY_TEXT + "\n\n" + "New added stocks list is"
    BODY_TEXT = BODY_TEXT + "\n\n" + newBuyPrintOutout.to_string()
    BODY_TEXT = BODY_TEXT + "\n\n" + "additional value over and above the exit value added"
    BODY_TEXT = BODY_TEXT + "\n\n" + str(newNet)
    BODY_TEXT = BODY_TEXT + "\n\n" + "======================="
    BODY_TEXT = BODY_TEXT + "\n\n" + "Position AdjustmentRatio is {0}".format(actualNetToReallocateRatio)
    BODY_TEXT = BODY_TEXT + "\n\n" + "position Readjustment is"
    BODY_TEXT = BODY_TEXT + "\n\n" + positionRebalanceTrades.to_string()
    BODY_TEXT = BODY_TEXT + "\n\n" + "additional net for position readjustment"
    BODY_TEXT = BODY_TEXT + "\n\n" + str(positionRebalanceNet)
    BODY_TEXT = BODY_TEXT + "\n\n" + "======================="
    BODY_TEXT = BODY_TEXT + "\n\n" + "New portfolio"
    BODY_TEXT = BODY_TEXT + "\n\n" + currentPortfolio.to_string(col_space=18, justify='left')


    return BODY_TEXT

# def sendAWSMailAttachment(recipeint, subject, body_text, attachPath1, attachPath1):
def sendAWSMailAttachment(mailtype, subject, body_text,
                          attachPath1,attachPath2,cx_customer_list=["dartconsultants.hyd@gmail.com"]):
    # Replace sender@example.com with your "From" address.
    # This address must be verified with Amazon SES.
    DestinationDict={}
    RECIPIENT=[]

    if mailtype == 'ADMIN':
        RECIPIENT = ["dartconsultants.hyd@gmail.com"]
        DestinationDict = {'ToAddresses': RECIPIENT}

    elif mailtype == 'CUSTOMER':
        RECIPIENT = ["dartconsultants.hyd@gmail.com"]
        RECIPIENTBCC = cx_customer_list
        DestinationDict = {'ToAddresses': RECIPIENT, 'BccAddresses': RECIPIENTBCC}
    # Replace sender@example.com with your "From" address.
    # This address must be verified with Amazon SES.
    SENDER = "dartconsultants.hyd@gmail.com"

    print("ttach path are {0} {1}".format(attachPath1,attachPath2))

    print(body_text)

    # Replace recipient@example.com with a "To" address. If your account
    # is still in the sandbox, this address must be verified.


    # Specify a configuration set. If you do not want to use a configuration
    # set, comment the following variable, and the
    # ConfigurationSetName=CONFIGURATION_SET argument below.
    # CONFIGURATION_SET = "ConfigSet"

    # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
    AWS_REGION = "us-east-1"

    # The subject line for the email.
    SUBJECT = subject

    # The full path to the file that will be attached to the email.
    ATTACHMENT1 = attachPath1
    ATTACHMENT2 = attachPath2

    # The email body for recipients with non-HTML email clients.
    BODY_TEXT = body_text

    # # The HTML body of the email.
    # BODY_HTML = """\
    # <html>
    # <head></head>
    # <body>
    # <h1>Hello!</h1>
    # <p>Please see the attached file for a list of customers to contact.</p>
    # </body>
    # </html>
    # """

    # The character encoding for the email.
    CHARSET = "utf-8"

    # Create a new SES resource and specify a region.
    client = boto3.client('ses', region_name=AWS_REGION)

    # Create a multipart/mixed parent container.
    msg = MIMEMultipart('mixed')
    # Add subject, from and to lines.
    msg['Subject'] = SUBJECT
    msg['From'] = SENDER
    msg['To'] = RECIPIENT

    # Create a multipart/alternative child container.
    msg_body = MIMEMultipart('alternative')

    # Encode the text and HTML content and set the character encoding. This step is
    # necessary if you're sending a message with characters outside the ASCII range.
    textpart = MIMEText(BODY_TEXT.encode(CHARSET), 'plain', CHARSET)
    # htmlpart = MIMEText(BODY_HTML.encode(CHARSET), 'html', CHARSET)

    # Add the text and HTML parts to the child container.
    msg_body.attach(textpart)
    # msg_body.attach(htmlpart)

    # Define the attachment part and encode it using MIMEApplication.
    att1 = MIMEApplication(open(ATTACHMENT1, 'rb').read())
    att2 = MIMEApplication(open(ATTACHMENT2, 'rb').read())

    # Add a header to tell the email client to treat this part as an attachment,
    # and to give the attachment a name.
    att1.add_header('Content-Disposition', 'attachment', filename=os.path.basename(ATTACHMENT1))
    att2.add_header('Content-Disposition', 'attachment', filename=os.path.basename(ATTACHMENT2))

    # Attach the multipart/alternative child container to the multipart/mixed
    # parent container.
    msg.attach(msg_body)

    # Add the attachment to the parent container.
    msg.attach(att1)
    msg.attach(att2)
    # print(msg)
    try:
        # Provide the contents of the email.
        response = client.send_raw_email(
            Source=SENDER,
            Destinations=DestinationDict,
            RawMessage={
                'Data': msg.as_string(),
            }
            # ConfigurationSetName=CONFIGURATION_SET
        )
    # Display an error if something goes wrong.
    except Exception as e:
        print(e)
    else:
        print("Email sent! Message ID:"),
        print(cx_customer_list)
        print(response['MessageId'])

def handler(event, context):
    # This will use the already data generated which has every weekly list
    # The nse 500 emerges from 200 by 21 Jul - so this will be from Jul to Oct - 21 - 4 months
    # This will pave way for the full back test
    # Source data on Dart drive - SOTM_BackTest
    # Will go from week 20 when
    #TODO For furutre loooking need to use the index list Plus portfolio holding as the universe for currentRanking
    cx_mail_list = event['cx_mail_list']

    runReport = 'weekly'
    takeBackupS3(runReport)

    performanceDF = pd.DataFrame(columns=['Date', 'PortfolioValue', 'InvestmentValue', 'AdjustmentNet'])

    # Below for testing
    # #Todo Delete this for prod
    # isIndexAbove200 = 1
    # # for week in range(21, 34, 1):
    isIndexAbove200 = calcIndex200()


    # currentRanking = pd.read_csv('{0}/week{1}_master.csv'.format(folderPath,week))
    outputDf = pd.DataFrame(columns=['Stock','Close','Slope','RValue','RAdjustedSlope','100SMA','isAbove100SMA','ATR','maxGap','numberofStocks','cost','targetPercentage','isIndexAbove200'])

    if isIndexAbove200 == 1:

        pastPortFolio = getPastPortfolio(runReport)
        indexList, currentRanking = getCurrentRanking(isIndexAbove200)
        currentRanking = currentRanking.sort_values(by='RAdjustedSlope', ascending=False)
        currentRanking.reset_index(inplace=True)
        currentRanking = currentRanking.drop(columns=['index','numberofStocks','cost','targetPercentage'])


        week = int(pastPortFolio['week'].unique())
        investmentValue = pastPortFolio['investmentValue'].unique()[0]
        # portfolioValue = getPortfolioValue(pastPortFolio,currentRanking)
        currentPortfolio, portfolioValue =  getCurrentPortfolio(pastPortFolio,currentRanking)




        #below is for calculating the number of stocks  static porfolio value
        currentRanking = calcNumOfStock(portfolioValue,currentRanking)


        # Below needs to be developed. After the rest of the code to be split between daily and weekly
        # if run == 'daily':
        #     previousRanking = getPreviousRanking()
        #     biggestgainers, biggestlosers = getDailyGainersLosers(currentRanking)

        # Portfolio rebalance based on exit
        exitStocks, exitCost =  getExitStocks(pastPortFolio,currentRanking)

        currentInvEligible = currentRanking[currentRanking['isAbove100SMA'] == 1]
        currentInvEligible = currentInvEligible[currentInvEligible['maxGap'] < 15]

        newBuy = pd.DataFrame(columns = currentInvEligible.columns)
        newBuyPrintOutout = newBuy
        newNet = 0
        if exitCost > 0:
            newBuy, newNet,newBuyPrintOutout = portfolioRebalance(portfolioValue,currentInvEligible, currentPortfolio, exitCost)
            currentPortfolio = currentPortfolio[~currentPortfolio['Stock'].isin(exitStocks['Stock'])]
            # currentPortfolio = currentPortfolio.append(newBuy.loc[:, currentPortfolio.columns])
            currentPortfolio = currentPortfolio.append(newBuy.reindex(columns = currentPortfolio.columns))



        # Every alternate week postion rebalance.
        positionRebalanceTrades = pd.DataFrame(columns=['Stock', 'portfolioNumberofStocks', 'currentNumOfStocks'])
        positionRebalanceNet = 0
        actualNetToReallocateRatio = 0
        if week % 2 == 0:
            positionRebalanceTrades, positionRebalanceNet, actualNetToReallocateRatio = positionRebalance(newNet, currentPortfolio, currentInvEligible)
            if len(positionRebalanceTrades) > 0:
                for rebalanceStock in positionRebalanceTrades['Stock'].values:
                    currentPortfolio.loc[currentPortfolio['Stock'] == rebalanceStock,'numberofStocks'] = positionRebalanceTrades.loc[positionRebalanceTrades['Stock'] == rebalanceStock,'currentNumOfStocks'].values[0]



        #Update cost and Target percentage of Current Portfolio
        currentPortfolio['cost'] = currentPortfolio['numberofStocks'] * currentPortfolio['Close']
        currentPortfolio['targetPercentage'] = (currentPortfolio['cost'] * 100 / portfolioValue)

        #write current portfolio
        # currentPortfolio.to_csv('{0}/currentPortfolio.csv'.format(folderPath), index=False)



        # print("debug")
        netAdjustment = newNet + positionRebalanceNet

        # #populate data frame for charting the performance

        # performanceDF.loc[len(performanceDF)] = [currentRanking.head(1)['startTime'].values[0][0:10],portfolioValue,investmentValue,netAdjustment]
        investmentValue = investmentValue + netAdjustment
        week = week + 1

        currentPortfolio['week'] = week
        currentPortfolio['investmentValue'] = investmentValue

        tmpPortfolioPath, tmpRankingPath = createCsv(currentRanking,currentPortfolio,runReport)


        # outputDate = currentRanking.head(1)['startTime'].values[0][0:10]
        BODY_TEXT = getMailBody(portfolioValue,exitStocks,exitCost,newBuyPrintOutout,newNet,positionRebalanceTrades,positionRebalanceNet,currentPortfolio,actualNetToReallocateRatio)
        SUBJECT = "{0} MoAC report".format(runReport)

        # sendAWSMailAttachment('dartconsultants.hyd@gmail.com', SUBJECT, BODY_TEXT, tmpPortfolioPath, tmpRankingPath)
        sendAWSMailAttachment('CUSTOMER', SUBJECT, BODY_TEXT, tmpPortfolioPath, tmpRankingPath, cx_mail_list)
        # sendAWSMailAttachment('CUSTOMER', SUBJECT, BODY_TEXT, tmpPath, cx_mail_list)
        #TOdO overwrite current portfolio and create a date based current porfolio. Also may be print past portfolio as is
        #Todo print current rank and have past rank
        #TODO top absolut movers
        #TODO mail

        # print(performanceDF)
    else:
        print("Index below 200 DMA - Move to cash")

#
# if __name__== "__main__":
#     main()


    data = {
        'output': 'Hello World',
        'timestamp': datetime.datetime.utcnow().isoformat()
    }
    return {'statusCode': 200,
            'body': json.dumps(data),
            'headers': {'Content-Type': 'application/json'}}
