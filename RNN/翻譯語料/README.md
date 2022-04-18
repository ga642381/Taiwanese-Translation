總共使用兩種華臺語料
1.iCorpus台華平行新聞語料庫
2.TGB通訊


## TGB
原始資料:
1. TGB_華語 : https://github.com/sih4sing5hong5/huan1-ik8_gian2-kiu3/tree/master/%E8%AA%9E%E6%96%99/TGB/對齊平行華語資料
2. TGB_閩南語 : https://github.com/sih4sing5hong5/huan1-ik8_gian2-kiu3/tree/master/%E8%AA%9E%E6%96%99/TGB/對齊平行閩南語


## 處理資料
1.1. python 轉臺羅icorpus.py
1.2. python 處理icorpus.py
2. python 處理TGB.py

資料會放進 tmp 資料夾內

3. python gen_data.py

產生 training, testing, validation data
資料會放進 data 資料夾內

4. python build_dataset.py
資料會放進 data 資料夾內
