import random
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 OPR/85.0.4341.72",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 OPR/85.0.4341.72",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Vivaldi/5.3.2679.55",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Vivaldi/5.3.2679.55",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Brave/1.40.107",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Brave/1.40.107",
]

random.choice(USER_AGENTS)

class Results:
    @staticmethod
    def scrape(race_id_list):
        """
        レース結果データをスクレイピングする関数
        Parameters:
        ----------
        race_id_list : list
            レースIDのリスト
        Returns:
        ----------
        race_results_df : pandas.DataFrame
            全レース結果データをまとめてDataFrame型にしたもの
        
        船橋競馬場の場合
        20xx（年：xxのところは任意）+43（船橋競馬）
        ・ここまでは共通
        04（開催月：任意）+04（開催日：任意）+01（レース番号：01から12まで）
        ・これを指定しなければならないので非常に注意
        
        例）
        ・船橋競馬場開催
        ・2019年1月15日の第一レースの場合
        201943011501
        race_idは上記のようになる
        ------------------------------
        2019年
        ・1月
        15/16/17/18
        ・2月
        18/19/20/21/22
        ・3月
        11/12/13/14/15
        ・4月
        15/16/17/18/19
        ・5月
        06/07/08/09/10
        ・6月
        17/18/19/20/21
        ・7月
        14/15/16/17
        ・8月
        07/08/09/10
        29/30/31
        ・9月
        02
        23/24/25/26/27
        ・10月
        28/29/30/31
        ・11月
        01
        ・12月
        09/10/11/12/13

        ------------------------------
        2020年
        ・1月
        07/08/09/10/11
        ・2月
        10/11/12/13/14
        ・3月
        10/11/12/13/14
        30/31
        ・4月
        01/02/03
        ・5月
        04/05/06/07/08
        ・6月
        15/16/17/18/19
        ・7月
        18/19/20/21/22
        ・8月
        03/04/05/06
        ・9月
        28/29/30
        ・10月
        01/02
        26/27/28/29/30
        ・11月
        30
        ・12月
        01/02/03/04

        """
        #race_idをkeyにしてDataFrame型を格納
        race_results = {}
        for race_id in tqdm(race_id_list):
            #任意の秒間隔でリクエストを送信
            RandTime = random.randint(3, 8)
            #print(f"待機時間：{RandTime} seconds")
            time.sleep(RandTime)
            try:
                url = "https://db.netkeiba.com/race/" + race_id
                headers = {'User-Agent': random.choice(USER_AGENTS)}
                html = requests.get(url, headers=headers)
                html.encoding = "EUC-JP"
                # メインとなるテーブルデータを取得
                df = pd.read_html(html.text)[0]
                # 列名に半角スペースがあれば除去する
                df = df.rename(columns=lambda x: x.replace(' ', ''))
                # 天候、レースの種類、コースの長さ、馬場の状態、日付をスクレイピング
                soup = BeautifulSoup(html.text, "html.parser")
                texts = (
                    soup.find("div", attrs={"class": "data_intro"}).find_all("p")[0].text
                    + soup.find("div", attrs={"class": "data_intro"}).find_all("p")[1].text
                )
                info = re.findall(r'\w+', texts)
                for text in info:
                    if text in ["芝", "ダート"]:
                        df["race_type"] = [text] * len(df)
                    if "障" in text:
                        df["race_type"] = ["障害"] * len(df)
                    if "m" in text:
                        df["course_len"] = [int(re.findall(r"\d+", text)[-1])] * len(df)
                    if text in ["良", "稍重", "重", "不良"]:
                        df["ground_state"] = [text] * len(df)
                    if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                        df["weather"] = [text] * len(df)
                    if "年" in text:
                        df["date"] = [text] * len(df)
                #馬ID、騎手IDをスクレイピング
                horse_id_list = []
                horse_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
                    "a", attrs={"href": re.compile("^/horse")}
                )
                for a in horse_a_list:
                    horse_id = re.findall(r"\d+", a["href"])
                    horse_id_list.append(horse_id[0])
                jockey_id_list = []
                jockey_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
                    "a", attrs={"href": re.compile("^/jockey")}
                )
                for a in jockey_a_list:
                    jockey_id = re.findall(r"\d+", a["href"])
                    jockey_id_list.append(jockey_id[0])
                df["horse_id"] = horse_id_list
                df["jockey_id"] = jockey_id_list
                #インデックスをrace_idにする
                df.index = [race_id] * len(df)
                race_results[race_id] = df
            #存在しないrace_idを飛ばす
            except IndexError:
                continue
            except AttributeError: #存在しないrace_idでAttributeErrorになるページもあるので追加
                continue
            #wifiの接続が切れた時などでも途中までのデータを返せるようにする
            except Exception as e:
                print(e)
                break
            #Jupyterで停止ボタンを押した時の対処
            except:
                break
        #pd.DataFrame型にして一つのデータにまとめる
        race_results_df = pd.concat([race_results[key] for key in race_results])
        return race_results_df
    
race_id_list = []
"""
for year in range(2019, 2020, 1):
    for place in range(1, 11, 1):
        for kai in range(1, 7, 1):
            for day in range(1, 13, 1):
                for r in range(1, 13, 1):
                    race_id = "2019" + str(place).zfill(2) + str(kai).zfill(2) + str(day).zfill(2) + str(r).zfill(2)
                    race_id_list.append(race_id)
"""

months_days = {
    1:  [15]
    #,
    #2:  [18, 19, 20, 21, 22],
    #3:  [11, 12, 13, 14, 15],
    #4:  [15, 16, 17, 18, 19],
    #5:  [6,  7,  8,  9,  10]
    #,
    #6:  [17, 18, 19, 20, 21],
    #7:  [14, 15, 16, 17],
    #8:  [7,  8,  9,  10, 29, 30, 31],
    #9:  [2,  23, 24, 25, 26, 27],
    #10: [28, 29, 30, 31],
    #11: [1],
    #12: [9,  10, 11, 12, 13],
}

year       = 2019      # 20xx の xx 部分
track_code = 43        # 船橋競馬場のコード

race_id_list = []
for month, days in months_days.items():
    for day in days:
        for race_no in range(1, 2):  # レース番号 01～12
            race_id_list.append(
                f"{year}"            # 4桁の年
                f"{track_code:02d}"  # 2桁の競馬場コード
                f"{month:02d}"       # 2桁の月
                f"{day:02d}"         # 2桁の日
                f"{race_no:02d}"     # 2桁のレース番号
            )


#print(f"生成された race_id は {len(race_id)} 件です。")
# 動作確認例
#print(race_id[:])

#実行
#results = Results.scrape(race_id_list)
#保存
#results.to_pickle('results.pickle')
#読み込み
results = pd.read_pickle('results.pickle')
#① DataFrame を文字列表現で書き出し（整形済テーブル風）
with open('results.txt', 'w', encoding='utf-8') as f:
    f.write(results.to_string())

###====================================================

#馬の過去成績データを処理するクラス
class HorseResults:
    @staticmethod
    def scrape(horse_id_list):
        """
        馬の過去成績データをスクレイピングする関数

        Parameters:
        ----------
        horse_id_list : list
            馬IDのリスト

        Returns:
        ----------
        horse_results_df : pandas.DataFrame
            全馬の過去成績データをまとめてDataFrame型にしたもの
        """

        #horse_idをkeyにしてDataFrame型を格納
        horse_results = {}
        for horse_id in tqdm(horse_id_list):
            time.sleep(1)
            try:
                url = 'https://db.netkeiba.com/horse/' + horse_id
                headers = {'User-Agent': random.choice(USER_AGENTS)}
                html = requests.get(url, headers=headers)
                html.encoding = "EUC-JP"
                df = pd.read_html(html.text)[2]
                df.index = [horse_id] * len(df)
                horse_results[horse_id] = df
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        #pd.DataFrame型にして一つのデータにまとめる        
        horse_results_df = pd.concat([horse_results[key] for key in horse_results])

        return horse_results_df
    
#horse_id_list = results['horse_id'].unique()
#horse_results = HorseResults.scrape(horse_id_list)
#horse_results #jupyterで出力
#horse_results.to_pickle('horse_results.pickle')

horse_results = pd.read_pickle('horse_results.pickle')

with open('horse_results.txt', 'w', encoding='utf-8') as f:
    f.write(horse_results.to_string())