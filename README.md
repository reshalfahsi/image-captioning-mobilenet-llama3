# Image Captioning With MobileNet-LLaMA 3

 <div align="center">
    <a href="https://colab.research.google.com/github/reshalfahsi/image-captioning-mobilenet-llama3/blob/master/Image_Captioning_MobileNet_LLaMA3.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a>
    <br />
 </div>


<div align="center">
    <img src="https://github.com/reshalfahsi/image-captioning-mobilenet-llama3/blob/master/assets/architecture.png" alt="architecture" >
    </img>
    MobileNet V3 + LLaMA 3 architecture.
    <br />
</div>


Image captioning is one of the problems in computer vision, constituting two kinds of modalities, i.e., image and text. Given a particular image, a caption regarding it is automatically generated. One can easily leverage a CNN-based architecture to draw the numerical representation out of the image. When interacting with the text, the long-range dependencies method has to be employed. Uplifted by the recent success of LLaMA 3, this project utilizes its computational block called the LLaMA 3 Transformer block. This block comprises RMSNorm, Grouped Multi-Query Attention, Feed Forward SwiGLU, and Rotary Position Embedding. Anyhow, in the original implementation, the Transformer block was only used as the decoder. In this project, the Transformer block is used as both the encoder and the decoder. In the encoder, before image data is funneled into the architecture, a CNN-based architecture, MobileNet-V3, is leveraged, acting similarly to the text embedding. Therefore, this architecture is dubbed MobileNet-LLaMA 3. To get knowledge on the performance of the model, the Flickr-8k dataset is used. The dataset is separated into the train, validation, and test sets in the 80-10-10 rule. Quantitatively, the performance of the model is measured via the ROUGE score, to be precise, the ROUGE-1 F-measure.



## Experiment

Proceed to this [notebook](https://github.com/reshalfahsi/image-captioning-mobilenet-llama3/blob/master/Image_Captioning_MobileNet_LLaMA3.ipynb) to vacate and answer your confusion and questions about this project by contemplating each line of code.


## Result

## Quantitative Result

The MobileNet-LLaMA3 performance on the test set is quantitatively displayed by the following table.

Test Metric                   | Score
----------------------------- | -------------
ROUGE-1 F-measure             | 36.69%


## Loss Curve

<p align="center"> <img src="https://github.com/reshalfahsi/image-captioning-mobilenet-llama3/blob/master/assets/loss_curve.png" alt="loss_curve" > <br /> Loss curves of the MobileNet-LLaMA 3 model on the train and validation sets. </p>


## Qualitative Result

The following image shows the qualitative results of MobileNet-LLaMA 3 on the test set.

<p align="center"><img src="https://github.com/reshalfahsi/image-captioning-mobilenet-llama3/blob/master/assets/qualitative.png" alt="qualitative"><br/> The image-caption pairs yielded from MobileNet-LLaMA 3. </p>

The MobileNet-LLaMA 3 model is also assessed in the wild.

<p align="center"><img src="https://github.com/reshalfahsi/image-captioning-mobilenet-llama3/blob/master/assets/qualitative-in-the-wild.png" alt="qualitative"><br/> The result of MobileNet-LLaMA 3 in the wild. </p>


## Citation

Feel free to cite this repository:

```
@misc{mobilenet-llama3,
   title = {Image Captioning With MobileNet-LLaMA 3},
   url = {https://github.com/reshalfahsi/image-captioning-mobilenet-llama3},
   author = {Resha Dwika Hefni Al-Fahsi},
}
```


## Credit

- [Introducing Meta Llama 3: The most capable openly available LLM to date](https://ai.meta.com/blog/meta-llama-3/)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288)
- [Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467)
- [GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202)
- [Roformer: Enhanced Transformer With Rotary Position Embedding](https://arxiv.org/pdf/2104.09864)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245)
- [Efficiently Scaling Transformer Inference](https://arxiv.org/pdf/2211.05102)
- [Transformers Optimization: Part 1 - KV Cache](https://r4j4n.github.io/blogs/posts/kv/)
- [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244)
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)
- [torchtune](https://github.com/pytorch/torchtune)
- [Exploring and building the LLaMA 3 Architecture : A Deep Dive into Components, Coding, and Inference Techniques](https://medium.com/@vi.ai_/exploring-and-building-the-llama-3-architecture-a-deep-dive-into-components-coding-and-43d4097cfbbb)
- [LLaMA 2 from scratch 🦙](https://github.com/viai957/llama-inference)
- [aladdinpersson's Image Captioning](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/image_captioning)
- [Keras' Image Captioning](https://keras.io/examples/vision/image_captioning/)
- [Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics (Extended Abstract)](https://www.ijcai.org/Proceedings/15/Papers/593.pdf)
- [jbrownlee's Flickr8k Dataset](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
