from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method=='GET':
        return render_template('index.html')

    else:
        data = CustomData(
            Age=int(request.form.get('Age')),
            Gender=request.form.get('Gender'),
            Location=request.form.get('Location'),
            Subscription_Length_Months=int(request.form.get('Subscription_Length_Months')),
            Monthly_Bill=float(request.form.get('Monthly_Bill')),
            Total_Usage_GB=int(request.form.get('Total_Usage_GB'))
        )
        data_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(data_df)

        results = round(pred[0], 2)

        return render_template('index.html', final_result=results)


if __name__ == '__main__':
    app.run(debug=True)
