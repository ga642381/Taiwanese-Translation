from 臺灣言語工具.解析整理.拆文分析器 import 拆文分析器
from 臺灣言語工具.音標系統.閩南語.臺灣閩南語羅馬字拼音相容教會羅馬字音標 import 臺灣閩南語羅馬字拼音相容教會羅馬字音標
from 臺灣言語工具.音標系統.閩南語.臺灣閩南語羅馬字拼音 import 臺灣閩南語羅馬字拼音
from 臺灣言語工具.基本物件.公用變數 import 分字符號
from 臺灣言語工具.基本物件.公用變數 import 分詞符號

"""
複製以下取代集裡面的看語句

def 看語句(self):
    詞的型陣列 = []
    頂一詞上尾是羅馬字 = False
    for 一詞 in self.內底詞:
        詞型 = 一詞.看語句()
        #if (
        #    頂一詞上尾是羅馬字
        #    and (敢是拼音字元(詞型[0]) or 詞型[0].isdigit() or 詞型[0] == 分字符號)
        #):
            #詞的型陣列.append(分詞符號)
        # 輕聲詞 '--sui2' => '--sui2 '
        # 一般詞 'sui2' => 'sui2 '
        詞的型陣列.append(詞型)
        頂一詞上尾是羅馬字 = 敢是拼音字元(詞型[-1]) or 詞型[-1].isdigit()
    return ' '.join(詞的型陣列)
    
"""
教羅檔案 = "./icorpus/閩"
臺羅檔案 = "./icorpus/閩_臺羅"
with open(教羅檔案, "r") as in_file:
    with open(臺羅檔案, "w") as out_file:
        for 教羅 in in_file:
            臺羅物件 = 拆文分析器.建立句物件(教羅).轉音(臺灣閩南語羅馬字拼音相容教會羅馬字音標)
            out_file.write(臺羅物件.看語句())
    