from flask import Flask, render_template, request
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict_point():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data = CustomData(
            mean_radius = float(request.form.get('mean_radius')),
            mean_texture = float(request.form.get('mean_texture')),
            mean_perimeter = float(request.form.get('mean_perimeter')),
            mean_area = float(request.form.get('mean_area')),
            mean_smoothness = float(request.form.get('mean_smoothness')),
            mean_compactness = float(request.form.get('mean_compactness')),
            mean_concavity = float(request.form.get('mean_concavity')),
            mean_concave_points = float(request.form.get('mean_concave_points')),
            mean_symmetry = float(request.form.get('mean_symmetry')),
            mean_fractal_dimension = float(request.form.get('mean_fractal_dimension')),
            radius_error = float(request.form.get('radius_error')),
            texture_error = float(request.form.get('texture_error')),
            perimeter_error = float(request.form.get('perimeter_error')),
            area_error = float(request.form.get('area_error')),
            smoothness_error = float(request.form.get('smoothness_error')),
            compactness_error = float(request.form.get('compactness_error')),
            concavity_error = float(request.form.get('concavity_error')),
            concave_points_error = float(request.form.get('concave_points_error')),
            symmetry_error = float(request.form.get('symmetry_error')),
            fractal_dimension_error = float(request.form.get('fractal_dimension_error')),
            worst_radius = float(request.form.get('worst_radius')),
            worst_texture = float(request.form.get('worst_texture')),
            worst_perimeter = float(request.form.get('worst_perimeter')),
            worst_area = float(request.form.get('worst_area')),
            worst_smoothness = float(request.form.get('worst_smoothness')),
            worst_compactness = float(request.form.get('worst_compactness')),
            worst_concavity = float(request.form.get('worst_concavity')),
            worst_concave_points = float(request.form.get('worst_concave_points')),
            worst_symmetry = float(request.form.get('worst_symmetry')),
            worst_fractal_dimension = float(request.form.get('worst_fractal_dimension'))
        )

        data_df = data.get_data_as_dataframe()        
        predict_pipeline = PredictPipeline()

        prediction, probability = predict_pipeline.predict(data_df)

        return render_template('prediction.html',prediction=prediction, probability=probability)  

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)