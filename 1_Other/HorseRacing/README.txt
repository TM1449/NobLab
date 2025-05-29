現段階で考えていること

【小目標】
・船橋競馬場専用のAIを作る。
・「各距離」「各馬場状態」「各馬齢」ごとに「各頭数ごと」の平均の【走破タイム】と【上がり3F】を計算する。
・レース結果をスクレイピングした際に、見出しタグも確認する。（B2やB3などを抜き出す）
例：
https://db.netkeiba.com/?sort_key=member&sort_type=asc&page=&serial=a%3A17%3A%7Bs%3A3%3A%22pid%22%3Bs%3A9%3A%22race_list%22%3Bs%3A4%3A%22word%22%3Bs%3A0%3A%22%22%3Bs%3A10%3A%22start_year%22%3Bs%3A4%3A%222016%22%3Bs%3A9%3A%22start_mon%22%3Bs%3A4%3A%22none%22%3Bs%3A8%3A%22end_year%22%3Bs%3A4%3A%22none%22%3Bs%3A7%3A%22end_mon%22%3Bs%3A4%3A%22none%22%3Bs%3A3%3A%22jyo%22%3Ba%3A1%3A%7Bi%3A0%3Bs%3A2%3A%2243%22%3B%7Ds%3A4%3A%22baba%22%3Ba%3A2%3A%7Bi%3A0%3Bs%3A1%3A%221%22%3Bi%3A1%3Bs%3A1%3A%222%22%3B%7Ds%3A5%3A%22barei%22%3Ba%3A1%3A%7Bi%3A0%3Bs%3A2%3A%2211%22%3B%7Ds%3A9%3A%22kyori_min%22%3Bs%3A0%3A%22%22%3Bs%3A9%3A%22kyori_max%22%3Bs%3A0%3A%22%22%3Bs%3A4%3A%22sort%22%3Bs%3A4%3A%22date%22%3Bs%3A4%3A%22list%22%3Bs%3A3%3A%22100%22%3Bs%3A9%3A%22style_dir%22%3Bs%3A17%3A%22style%2Fnetkeiba.ja%22%3Bs%3A13%3A%22template_file%22%3Bs%3A14%3A%22race_list.html%22%3Bs%3A9%3A%22style_url%22%3Bs%3A18%3A%22%2Fstyle%2Fnetkeiba.ja%22%3Bs%3A6%3A%22search%22%3Bs%3A64%3A%22%B4%FC%B4%D6%5B2016%C7%AF%A1%C1%CC%B5%BB%D8%C4%EA%5D%A1%A2%B6%A5%C7%CF%BE%EC%5B%C1%A5%B6%B6%5D%A1%A2%C7%CF%BE%EC%BE%F5%C2%D6%5B%CE%C9%A1%A2%E3%C4%5D%A1%A2%C7%CF%CE%F0%5B%A3%B2%BA%D0%5D%22%3B%7D&pid=race_list

例1：「1200m」「良・稍重」の「2歳馬」の2016年からの結果（頭数順で1ページ目）
https://db.netkeiba.com//?sort_key=member&sort_type=asc&page=1&serial=a%3A18%3A%7Bs%3A3%3A%22pid%22%3Bs%3A9%3A%22race_list%22%3Bs%3A4%3A%22word%22%3Bs%3A0%3A%22%22%3Bs%3A10%3A%22start_year%22%3Bs%3A4%3A%222016%22%3Bs%3A9%3A%22start_mon%22%3Bs%3A4%3A%22none%22%3Bs%3A8%3A%22end_year%22%3Bs%3A4%3A%22none%22%3Bs%3A7%3A%22end_mon%22%3Bs%3A4%3A%22none%22%3Bs%3A3%3A%22jyo%22%3Ba%3A1%3A%7Bi%3A0%3Bs%3A2%3A%2243%22%3B%7Ds%3A4%3A%22baba%22%3Ba%3A2%3A%7Bi%3A0%3Bs%3A1%3A%221%22%3Bi%3A1%3Bs%3A1%3A%222%22%3B%7Ds%3A5%3A%22barei%22%3Ba%3A1%3A%7Bi%3A0%3Bs%3A2%3A%2211%22%3B%7Ds%3A9%3A%22kyori_min%22%3Bs%3A0%3A%22%22%3Bs%3A9%3A%22kyori_max%22%3Bs%3A0%3A%22%22%3Bs%3A5%3A%22kyori%22%3Ba%3A1%3A%7Bi%3A0%3Bs%3A4%3A%221200%22%3B%7Ds%3A4%3A%22sort%22%3Bs%3A4%3A%22date%22%3Bs%3A4%3A%22list%22%3Bs%3A3%3A%22100%22%3Bs%3A9%3A%22style_dir%22%3Bs%3A17%3A%22style/netkeiba.ja%22%3Bs%3A13%3A%22template_file%22%3Bs%3A14%3A%22race_list.html%22%3Bs%3A9%3A%22style_url%22%3Bs%3A18%3A%22/style/netkeiba.ja%22%3Bs%3A6%3A%22search%22%3Bs%3A77%3A%22%B4%FC%B4%D6%5B2016%C7%AF%A1%C1%CC%B5%BB%D8%C4%EA%5D%A1%A2%B6%A5%C7%CF%BE%EC%5B%C1%A5%B6%B6%5D%A1%A2%C7%CF%BE%EC%BE%F5%C2%D6%5B%CE%C9%A1%A2%E3%C4%5D%A1%A2%C7%CF%CE%F0%5B%A3%B2%BA%D0%5D%A1%A2%B5%F7%CE%A5%5B1200m%5D%22%3B%7D&pid=race_list


・・・・「単回帰分析」をして、【平均走破タイム】よりも上の馬を計算する。

・・・・・・・・重回帰分析も行う？前処理

【中目標】
・他の競馬場も同様の距離ごとの【平均走破タイム】と【平均上がり3Fタイム】を計算する。


船橋競馬場（過去5年）(良・稍重)
1000m(203)
1200m(1235)
1500m(669)
1600m(420)
1700m(55)
1800m(79)
2200m(105)


地方競馬（南関東・船橋競馬場を含む）では、２歳・３歳馬には「A・B・C」といった格付けクラスが付かないため、能力の近い馬同士を「組（くみ）」という単位で分けています。
このときレース名に付く漢数字（「一」「二」「三」…）がその組番号を示しており、例えば「２歳一」は２歳馬の１組（＝最も力のあるグループ）、「２歳六」は同じく２歳馬の６組（＝比較的力の劣るグループ）を表します。

漢数字による組分けは、数字が小さいほど上位（より強い馬が集まる）という順序になっており、「一＞二＞三＞四＞五＞六＞…」の順でランク付けされます。

――つまり、船橋競馬場で「2歳六」とあれば、２歳馬のうち６番目に能力が低いグループのレース、「2歳一」であれば最上位のグループのレース、という意味になります。
