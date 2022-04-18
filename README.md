# Taiwanese-Translation

* Allennlp 與 RNN 兩個資料夾是兩個獨立的 Project

## Allenlp
* Allenlp資料夾內僅包含使用allennlp架設server的script
* 訓練模型等相關資源請參考: https://github.com/Chung-I/mandarin_to_tsm
* pre-trained model 請於此下載: https://drive.google.com/file/d/1XTpGU9EwWviIbZjg6bhdLMiWL8d1VI9i/view?usp=sharing
* pre-trained model 請放置於 Allennlp/model/ 資料夾內
* allennlp的版本不能過高，需為v1，若僅想執行此處程式，建議使用包好的conda env **tts_demo.tar.gz**: https://drive.google.com/file/d/1rWjPna__lJLH9uK_2WMsn95j6ALCS_E4/view?usp=sharing
* **tts_demo.tar.gz** 是使用 conda pack 工具export的，使用方式很簡單，請參考文件: https://conda.github.io/conda-pack/

## RNN
* RNN資料夾內包含文字的訓練資料
* RNN/翻譯語料內包含 **TGB** 及 **icorpus** 兩個來源
* RNN/翻譯語料內包含前處理的程式碼，處理完成的資料放在 RNN/翻譯語料/tmp/ 裡
* 根據經驗 (此處沒有數據)，RNN表現並沒有Allennlp的BERT模型好，但這邊提供了翻譯語料及前處理的script，可以好好利用

##

## 其他台語資源 
* Taiwanese-Corpus: https://github.com/Taiwanese-Corpus/hue7jip8
* 臺灣言語工具: https://github.com/i3thuan5/tai5-uan5_gian5-gi2_kang1-ku7
* iTaigi 愛台語 github: https://github.com/i3thuan5/itaigi
* iTaigi 愛台語 網站: https://itaigi.tw/
* ChhoeTaigi 找台語 github (看起來非常有價值): https://github.com/ChhoeTaigi/ChhoeTaigiDatabase
* ChhoeTaigi 找台語 網站:https://chhoe.taigi.info/
* T-BERT (有包含台語和國語的BERT): https://github.com/DeepqEducation/t-bert

## Contact
張凱爲 f09921048@ntu.edu.tw
