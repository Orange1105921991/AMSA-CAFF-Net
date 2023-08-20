# AMSA-CAFF-Net

Accurately estimating object numbers plays a crucial role in industrial applications. However, object counting in computer vision poses a formidable challenge, particularly when dealing with images containing tiny, similar, and stacked objects. Therefore, this paper proposes an encoder–decoder-based convolutional neural network named AMSA-CAFF Net to accurately count and regress high-quality density maps from X-ray images of highly dense microscopic components.

First run create_json.py to create the image path 

Then run make_dataset.py to generate ground truth density map

Finally configure the parameters in train.py and run



![a36de590a7f5f861f415adba16215ce](https://github.com/Orange1105921991/AMSA-CAFF-Net/assets/115998667/9dfb1fd1-ba13-4b42-879b-4c3abb49cdee)
Figure 1: The structure of AMSA-CAFF Net. AMSA-CAFF Net mainly consists of three major components: an adaptive multi-scale feature aggregation (AMSA) module for encoding adaptive multi-scale features in the encoder, a channel-wise adaptive feature fusion (CAFF) module to ensure smooth feature fusion, and a continuous feature enhancement mechanism to promote information integration in the decoder.

![0bdb12ee03b2083a8b7bc32bc03d63a](https://github.com/Orange1105921991/AMSA-CAFF-Net/assets/115998667/45e1014b-bf89-4e22-a92d-47dd94ebcbdc)
Figure 2: Density maps generated by the proposed method and other counting methods on XRAY-IECCD. The first column shows the input images of the 10 types of electronic components, the second column shows the ground truth density maps corresponding to their input images, and the third to eighth columns show the density maps predicted by the proposed method and the other five counting methods.

![6662740015378b66b6c4fec2fd15f8e](https://github.com/Orange1105921991/AMSA-CAFF-Net/assets/115998667/aeab3c02-94b3-4190-b496-b7d251281fc9)
Figure 3: Prediction results of AMSA-CAFF Net on the ShanghaiTech dataset. The first column shows the input images of the ShanghaiTech dataset, the second column shows the corresponding ground truth, and the third column shows the prediction results of the proposed method.

![115cb8c532ab8664873d33b88b62d15](https://github.com/Orange1105921991/AMSA-CAFF-Net/assets/115998667/e8bf0f10-fde0-4c1c-98b1-0b176ecabef5)
Figure 4: Prediction results of AMSA-CAFF Net on the CARPK dataset. The first column shows the input images of the CARPK dataset (each line represents a scene), the second column shows the corresponding ground truth, and the third column shows the prediction results of the proposed method.
