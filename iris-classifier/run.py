import gradio as gr
import numpy as np
from sklearn import datasets
from joblib import load 

iris = datasets.load_iris()
iris_labels = iris.target_names

clf = load("iris_svc.joblib")


def iris_predict(sepal_length, sepal_width, petal_length, petal_width):
    X_in = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    y_hat = clf.predict(X_in)[0]
    return iris_labels[y_hat]

demo = gr.Blocks()

with demo:
    gr.Markdown("Please input iris dimensions below and click Run to get prediction.")
    with gr.Row():
        sepal_length_in = gr.Number(label="Sepal Length (cm)")
        sepal_width_in = gr.Number(label="Sepal Width (cm)")
        petal_length_in = gr.Number(label="Petal Length (cm)")
        petal_width_in = gr.Number(label="Petal Width (cm)")
        # out = gr.Textbox()
    inp = [sepal_length_in, sepal_width_in, petal_length_in, petal_width_in]
    out = gr.Textbox(label="Predicted iris type")
    
    btn = gr.Button("Predict")
    btn.click(fn=iris_predict, inputs=inp, outputs=out)

demo.launch()
