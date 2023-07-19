import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('cuttest_dogs_model.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "The Cuttest Dogs Classifier 🐶🐕🦮🐕‍🦺"
description = "Classifier trainded on images of huskies, retrievers, pomeranians, corgis and samoyeds. Created as a demo for Deep Learning app using HuggingFace Spaces & Gradio."
examples = ['husky.jpg', 'retriever.jpg', 'corgi.jpg', 'pomeranian.jpg', 'samoyed.jpg']
interpretation='default'
enable_queue=True

gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(shape=(512, 512)),
    outputs=gr.outputs.Label(num_top_classes=5),
    title=title,
    description=description,
    examples=examples,
    interpretation=interpretation,
    enable_queue=enable_queue).launch()