import random
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.75 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:65.0) Gecko/20100101 Firefox/65.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/18.17763",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3864.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:62.0) Gecko/20100101 Firefox/62.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:67.0) Gecko/20100101 Firefox/67.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:68.0) Gecko/20100101 Firefox/68.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:61.0) Gecko/20100101 Firefox/61.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:62.0) Gecko/20100101 Firefox/62.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:92.0) Gecko/20100101 Firefox/92.0"
]


#鳥待月スプリント(B2)
#Raceid = 202543040409
#エイプリルダッシュ(C2)
Raceid = 202543040408
#春和景明特別(B1B2)
Raceid = 202543040411
#3歳九　未出走
Raceid = 202543040401

#「出馬表」の「出走馬」のURL
#https://nar.netkeiba.com/race/shutuba.html?race_id=202543040406&rf=top_pickup
#https://nar.netkeiba.com/race/shutuba.html?race_id=202543040406

#「出馬表」の「馬柱」のURL
#https://nar.netkeiba.com/race/shutuba_past.html?race_id=202543040406&rf=shutuba_submenu
#https://nar.netkeiba.com/race/shutuba_past.html?race_id=202543040406

#「出馬表」の「競馬新聞」のURL
#https://nar.netkeiba.com/race/newspaper.html?race_id=202543040406

#「出馬表」の「タイム指数」のURL
#https://nar.netkeiba.com/race/speed.html?race_id=202542042210


def Scraping_Race(Race_ID):
    #URLとヘッダーの設定
    #URL = "https://db.netkeiba.com/race/" + str(Race_ID)
    URL = "https://nar.netkeiba.com/race/shutuba_past.html?race_id=202543040406"
    HEADERS = {'User-Agent': random.choice(USER_AGENTS)}
    try:
        #リクエストの送信
        HTML = requests.get(URL, headers=HEADERS)
        HTML.encoding = "EUC-JP"
        # メインとなるテーブルデータを取得
        df = pd.read_html(HTML.text)[0]
        #df.to_csv("test.csv", index=False, encoding="utf-8-sig")

        Soup = BeautifulSoup(HTML.text, "html.parser")
        
        #h1タグを取得
        #h1_tags = Soup.find_all('h1')
        #h1タグの中から、正規表現で「(B2)」のようなタイトルを含むh1タグを取得
        #race_h1 = next(h for h in h1_tags if re.search(r'\(.+?\)', h.get_text(strip=True)))
        #h1タグのテキストを取得
        #title = race_h1.get_text(strip=True)        # "鳥待月スプリント(B2)"
        ##タイトルからクラスを取得
        #b2 = re.search(r'\((.+?)\)', title).group(1) # "B2"
        #print(b2)
        
        #title = h1.get_text(strip=True)
        #テスト用・データの確認
        print(df)
        #print(h1)
        #print(title)
        
    except IndexError:
        print("IndexError: URLの取得に失敗しました。")
        return None
    
def Scraping_Horse():
    pass

def Sleeping():
    time.sleep(random.randint(1, 3))

Scraping_Race(Raceid)