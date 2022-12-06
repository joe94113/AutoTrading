# MLDS HW1 - AutoTrading

Window CMD
```
py -m venv venv
.\venv\Scripts\activate.bat
pip install -r requirements.txt
python trader.py
```

### 想法
1. 載入訓練資料
2. 將資料做正規化
3. 將資料進行訓練集與測試集的切割
4. 使用`Keras` 的 `LSTM` 進行預測，加上兩層 256 個神經元 `LSTM` 層，並加上`Dropout`層來防止資料過度擬合，最後加上不同數目神經元的全連結層
5. 訓練模型
6. 載入測試資料
7. 迴圈讀取測試資料，並將資料寫回，對明日股價進行預測，並決定是否要購買，並輸出到`output.csv`(例: 利用`8/1\~8/31`預測`9/1`號股價，並決定動作，再利用`8/1~9/1`的資料預測`9/2`號股價，並決定動作，依此類推)，並對輸出動作進行檢查，防止重複購買及重複放空。

[colab](https://colab.research.google.com/drive/1ur3W8rgm4m9kFGrBbiOdDScGD4DiqeZq?usp=sharing#scrollTo=ByxByJ6n_m9L)

> 在`Windows`使用`Poetry`安裝`tensorflow@2.10.0`時一直報錯，查無原因，固無使用`Poetry`，拍謝
