# 2.2 and Appendix A (paste-ready)

## 2.2 Locating Regions for Visual Feature Measurement via Grad-CAM

Thumbnails associated with the two age groups can be compared on familiar visual cues, including people, text, and color. A thumbnail, however, layers people, text, and background in a single frame, so the same feature yields different values depending on which region it is read from. We therefore first train a thumbnail-only classifier to separate the two age groups, then use Grad-CAM (Selvaraju et al. 2017) to locate the regions the model relies on. Those regions set the focus for the visual feature measurements in Section 2.3.

Image classification used a frozen DINOv3 ViT-B/16 backbone (Siméoni et al. 2025) and a classification head trained to separate 18–34 from 65+ on the 9,586 thumbnails in Table 1. Age labels were assigned at the channel level, and the model predicted that label from the thumbnail alone. We used an 8:2 video-level split stratified by age label. Test-set accuracy was 78.4% (balanced accuracy .783), so the thumbnail carries a discriminative signal associated with audience age, which is what makes the heatmaps worth reading. Token handling, the top-50 sampling rule, OCR and YOLO thresholds, the ROI overlap rule, and training hyperparameters are reported in Appendix A. Per-category accuracy and class-wise precision, recall, and F1 are in Table A2.

In correctly classified high-confidence cases, 65+ activation concentrated on large text and on people, including faces and upper bodies, whereas 18–34 activation spread more widely across background, people, text, and objects (Figure 1). On the same top-50 samples, text and person together held 56.1% of 65+ heatmap energy, while 72.4% of 18–34 heatmap energy fell on the background (Table A3). Deletion and insertion tests of explanation faithfulness are shown in Figure A1 and Table A1. Additional exemplars, three per category, are in Figures A2 and A3.

In Overleaf two-column mode, do not use `center` + `\captionof`. That is not a float, so the two grids split from the caption and wrap into the next column. Use `figure*` (full width, top of page) with `\caption` inside the environment:

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth,height=0.26\textheight,keepaspectratio]{fig1_older_gradcam.png}\\[0.35em]
\includegraphics[width=\textwidth,height=0.26\textheight,keepaspectratio]{fig1_younger_gradcam.png}
\caption{Grad-CAM exemplars from correctly classified, channel-deduplicated top-50 cases (one thumbnail per category). Top: 65+, activation on text and people. Bottom: 18--34, activation spread more widely across the scene. Columns: original, predicted-class overlay, predicted-class heatmap. Rows: EDU, HEALTH, LIFESTYLE, SOCIETY.}
\label{fig:gradcam}
\end{figure*}
```

Figure 1. Grad-CAM exemplars from correctly classified, channel-deduplicated top-50 cases (one thumbnail per category). Top: 65+, activation on text and people. Bottom: 18–34, activation spread more widely across the scene. Columns: original, predicted-class overlay, predicted-class heatmap. Rows: EDU, HEALTH, LIFESTYLE, SOCIETY.

Later analyses follow these localizations. Where activation concentrates on thumbnail text, we measure text area and text–background contrast. Where it concentrates on people, we measure person area and estimated face age. Where it spreads across the scene, we compare captions describing people, actions, objects, and setting.

---

## Appendix A. Grad-CAM Implementation, ROI Tables, and Full Grids

\paragraph{Implementation.}
Grad-CAM was applied to the last Transformer block. CLS and register tokens were dropped and the remaining patch tokens were reshaped into a spatial map. Heatmaps were computed for the predicted class. For qualitative inspection we ranked correctly classified 18–34 and 65+ test images by prediction confidence, kept one image per channel, and used the top 50 in each group. Figure 1 shows the clearest instance of each group's pattern in that sample: for 65+, the image with the largest combined text and person share of heatmap energy; for 18–34, the image with the most dispersed heatmap, measured as the image fraction needed to accumulate half of the heatmap energy.

Text, person, and background regions were obtained with EasyOCR (Korean and English; confidence ≥ 0.3) and YOLOv8 person boxes (confidence ≥ 0.3). Overlapping pixels were assigned in the order text, then person, then background, so exclusive shares sum to 1. The classifier was trained for 5 epochs with Adam (learning rate = 0.001; batch size = 16). Thumbnails were resized to 224 × 224. The backbone identifier is `facebook/dinov3-vitb16-pretrain-lvd1689m`.

Test-set accuracy was .784 overall, .822 for EDU, .721 for HEALTH, .867 for LIFESTYLE, and .749 for SOCIETY. Recall was higher for 18–34 (.842) than for 65+ (.725), so more 65+ images were labeled 18–34 (257) than the reverse (150). Class-wise scores are in Table A1.

Table A1. Test-set prediction of the age-associated label.

| Group | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| 18–34 | .757 | .842 | .797 |
| 65+ | .818 | .725 | .769 |

\paragraph{Exclusive ROI energy.}
Table A2 reports where predicted-class heatmap energy fell on the same top-50 correct images. In 65+ cases, text and person together held 56.1% of heatmap energy (text 31.0%, person 25.1%). Text occupied 33.9% of the frame, so it is a large part of the layout rather than a small hot spot (concentration = 0.86). Person occupied 20.9% of the frame but 31.4% of pixels with CAM ≥ 0.5 (concentration = 1.42), so discriminative information was more densely packed in that smaller region. In 18–34 cases, 72.4% of heatmap energy and 74.5% of high-activation pixels fell on the background.

Table A2. Grad-CAM energy by exclusive ROI (top-50 correct; predicted-class heatmap). Concentration is energy share divided by area. Overlapping pixels are assigned to text, then person, then background.

| Group | ROI | Area | Energy | Conc. | CAM ≥ .5 |
| --- | --- | ---: | ---: | ---: | ---: |
| 65+ | text | 33.9% | 31.0% | 0.86 | 28.3% |
| 65+ | person | 20.9% | 25.1% | 1.42 | 31.4% |
| 65+ | background | 45.1% | 43.9% | 0.90 | 40.3% |
| 18–34 | text | 6.2% | 7.0% | 1.06 | 5.6% |
| 18–34 | person | 19.3% | 20.6% | 1.61 | 19.8% |
| 18–34 | background | 74.5% | 72.4% | 0.98 | 74.5% |

\paragraph{Faithfulness.}
Deletion removes pixels in order of Grad-CAM importance and records the drop in predicted probability for the target class. Insertion starts from a blurred image and restores the same pixels in the same order. For the 65+ top-50 correct sample, deletion AUC was .494 and insertion AUC was .725. For the 18–34 top-50 correct sample, deletion AUC was .871 and insertion AUC was .900: predicted probability stayed high even after important regions were removed, which is consistent with evidence spread across the image.

[Figure A1 here]

Figure A1. Deletion and insertion curves for the top-50 correct samples (left: 65+; right: 18–34).

Table A3 repeats the deletion and insertion tests one ROI at a time, using the original overlapping OCR and YOLO boxes (text + person + background can exceed 100%). Deletion masks the ROI and insertion shows only the ROI. delN = (baseline probability - deletion probability) / area; insN = insertion probability / area.

Table A3. Overlapping ROI area and area-normalized deletion/insertion (top-50 correct, n = 50 per group). Baseline probability was .999 for 65+ and 1.000 for 18–34.

| Group | ROI | Area | Deletion (raw) | Insertion (raw) | delN | insN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 65+ | text | 33.9% | .948 | .521 | 0.15 | 1.54 |
| 65+ | person | 25.9% | .854 | .543 | 0.56 | 2.10 |
| 65+ | background | 45.1% | .889 | .566 | 0.24 | 1.26 |
| 18–34 | text | 6.2% | 1.000 | .989 | 0.00 | 15.95 |
| 18–34 | person | 20.6% | .997 | .993 | 0.02 | 4.82 |
| 18–34 | background | 74.5% | .974 | .995 | 0.03 | 1.33 |

The two groups behave differently. In 65+ cases, masking a single ROI lowers the predicted probability and showing a single ROI recovers only about half of it, so person and text each carry part of the evidence, with person the densest per unit area. In 18–34 cases, deletion of any single ROI leaves the probability near baseline and insertion of any single ROI already returns a probability above .98, so no single region is necessary and each is close to sufficient on its own. This is the pattern expected when the evidence is redundant across the frame, and it is why the 18–34 area-normalized values, especially insN = 15.95 for the small text region, should not be read as a ranking of ROI importance.

[Figure A2 here]

Figure A2. Grad-CAM for correctly classified 65+ thumbnails (three per category; rows EDU, HEALTH, LIFESTYLE, SOCIETY).

[Figure A3 here]

Figure A3. Grad-CAM for correctly classified 18–34 thumbnails (three per category; rows EDU, HEALTH, LIFESTYLE, SOCIETY).

Model identifier (for replication): `facebook/dinov3-vitb16-pretrain-lvd1689m`.
