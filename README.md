# Image Processing and Classification Model for Screening Mammography in Automated Detection of Breast Cancer

This project aimed to enhance breast cancer detection from mammography images by leveraging deep learning techniques, specifically Convolutional Neural Networks (CNN) and Vision Transformers (ViT). The goal was to create a robust image processing and classification model capable of identifying malignancies in mammograms, improving early cancer detection and potentially saving lives.

Methodology:

1. Data Collection and Preparation:
- The dataset consisted of 54,706 mammogram images from 11,913 patients, provided by the Radiological Society of North America (RSNA).
- Given the data imbalance (only 2% of images were cancerous), techniques such as ROI (Region of Interest) cropping, data augmentation, and weighted loss functions were implemented to enhance model performance.
2. Model Architecture:
- Baseline Models: Initial models included pre-trained CNN (ResNeXt) and ViT architectures, which were fine-tuned for this specific medical imaging task.
- Advanced Techniques: The project incorporated ROI cropping to focus on relevant image areas, data augmentation to increase the robustness of the model, and auxiliary predictions to improve classification by considering additional features like patient age, breast density, and biopsy results.
3. Training and Evaluation:
- A 5-fold cross-validation approach was used to train and evaluate the models.
- Various combinations of preprocessing techniques were tested to find the optimal model setup.

Results:

The best-performing model was a CNN (ResNeXt) with all preprocessing steps (ROI cropping, augmentation, auxiliary predictions). This model achieved an F1 score of 0.345, significantly outperforming the baseline models.
Despite the improvements, challenges remained due to the substantial data imbalance, highlighting areas for future research and development.

Conclusion:

This project demonstrated the potential of using deep learning to assist in breast cancer detection from mammography images. By systematically improving model architecture and addressing data imbalance, the project achieved a meaningful performance boost, laying the groundwork for further advancements in automated medical diagnostics.
