# You can write code above the if-main block.
from predict import Trader

if __name__ == "__main__":
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="training_data.csv", help="input training data file name")
    parser.add_argument("--testing", default="testing_data.csv", help="input testing data file name")
    parser.add_argument("--output", default="output.csv", help="output file name")
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    trader = Trader()
    training_data = trader.load_data(args.training)
    foxconndf_norm = trader.normalize(training_data)
    X_train, y_train, X_test, y_test = trader.data_helper(foxconndf_norm, trader.time_frame)
    model = trader.build_model( trader.time_frame, trader.dimension )
    model.fit( X_train, y_train, batch_size=trader.batch_size, epochs=trader.epochs, validation_split=0.1, verbose=1)

    testing_data = trader.load_data(args.testing)
    with open(args.output, "w") as output_file:
        for idx, row in enumerate(testing_data.values):
            if(len(testing_data.values) - 1 == idx):
                continue

            # 將新資料寫回並正規化
            training_data = trader.add_row_to_df(row, training_data)
            foxconndf_norm = trader.normalize(training_data)
            X_train, y_train, X_test, y_test = trader.data_helper(foxconndf_norm, 20)
            
            # 取得明日開盤價以及隔天動作
            pred = model.predict(X_test)
            denorm_pred = trader.denormalize(training_data, pred)
            denorm_ytest = trader.denormalize(training_data, y_test)
            trader.draw_predict_result(denorm_pred, denorm_ytest)
            forecast_price = denorm_pred[-1][0]
            action = trader.predict_action(forecast_price)

            # 寫入檔案
            output_file.write(f"{action}")
            output_file.write("\n")