## 2.2 Locating Regions for Visual Feature Measurement via Grad-CAM

Thumbnails associated with the two age groups can be compared on familiar visual cues, including people, text, and color. The difficulty is not only which cues to record, but which regions of the frame those cues should be read from. Measuring text, people, or color without first seeing where the age-associated label is carried can spread the measurement across parts of the image that the classifier does not use. This section therefore begins with a localization step rather than with a new feature vocabulary.

A thumbnail-only classifier is trained to distinguish content associated with people aged 34 and under (~34) from content associated with people aged 65 and over (65+). Grad-CAM (Selvaraju et al., 2017) is then used to inspect where the model treats evidence as concentrated. The maps do not replace later feature extraction, and they are not treated as a conclusion about age-related style. They indicate where subsequent measures should focus. When activation lies on overlay text, later analyses emphasize text area and text–background contrast. When it lies on people, they emphasize person area, count, expression, and head direction. When it spreads across the scene, they turn to captions of people, actions, and setting. Global color statistics and an AI-generation score remain conventional whole-image measures; they are not derived from Grad-CAM.

### Data, model, and explanation procedure

**Data and split.** The classifier used 9,586 thumbnails from EDU, HEALTH, LIFESTYLE (meditation), and SOCIETY (Table 2.1). Each category is balanced 50:50 on the age-associated label. Age labels are assigned at the channel level, as described in the data collection section; the model predicts that channel-level label from the thumbnail, not the age of a depicted person or of an individual viewer. The data were split 8:2 at the video level, stratified on the age label only (random seed = 42). Category was not stratified, so test-set category counts differ. Because the split is by video, the same channel can appear in both sets, and most test videos come from channels that also appear in training. Classification accuracy is therefore not interpreted as a generalization estimate. It is used only to check that the thumbnail carries information about the age-associated label, so that the maps are worth reading as a guide to where later features should be measured.

**Table 2.1.** Thumbnails used in the Grad-CAM classifier, by age

| Category | n | ~34 | ~34 % | 65+ | 65+ % |
| --- | ---: | ---: | ---: | ---: | ---: |
| SOCIETY | 2,600 | 1,300 | 50.0 | 1,300 | 50.0 |
| EDU | 2,496 | 1,248 | 50.0 | 1,248 | 50.0 |
| HEALTH | 2,500 | 1,250 | 50.0 | 1,250 | 50.0 |
| LIFESTYLE (meditation) | 1,990 | 995 | 50.0 | 995 | 50.0 |
| Total | 9,586 | 4,793 | 50.0 | 4,793 | 50.0 |

**Model.** Image classification used a frozen DINOv3 ViT-B/16 backbone (Siméoni et al., 2025) and a classification head trained to separate ~34 from 65+ (5 epochs; Adam; learning rate = 0.001; batch size = 16). Thumbnails were resized by the model's image processor to 224 × 224. After training, gradients were enabled on the last Transformer block so that Grad-CAM could be computed.

**Grad-CAM and ROIs.** Grad-CAM was applied to the last Transformer block. CLS and register tokens were dropped and the remaining patch tokens were reshaped into a spatial map. Heatmaps were computed for the predicted class (for correctly classified images this is also the true label). For qualitative inspection we ranked correctly classified ~34 and 65+ test images by prediction confidence, kept one image per channel, and retained the top 50 in each group. These are high-confidence cases, not a random sample, and they are not treated as typical of the full group. Incorrect predictions were not interpreted.

Text, person, and background regions were obtained with EasyOCR (Korean and English; confidence ≥ 0.3) and YOLOv8 person boxes (confidence ≥ 0.3). These ROIs are the same familiar regions that later visual measures will use. They are applied here only to summarize where heatmap energy falls, not to introduce a new feature set. For heatmap-energy counts, overlapping pixels were assigned in the order text, then person, then background, so shares sum to 1. Deletion and insertion tests of explanation faithfulness, and area-normalized ROI deletion/insertion scores, are reported in Appendix A.

### Classification and localization results

Test-set accuracy was .784 and balanced accuracy was .783 (Table 2.2). Under this split, the thumbnail is predictive of the age-associated channel label. That result does not isolate visual style from channel identity, given the overlap reported above. Accuracy was highest for LIFESTYLE (meditation) (.867) and EDU (.822), and lower for SOCIETY (.749) and HEALTH (.721). Recall was higher for ~34 (.842) than for 65+ (.725), so more 65+ images were labeled ~34 (257) than the reverse (150). The maps below are therefore read as a guide to measurement, not as proof that a particular visual style generalizes beyond these channels.

**Table 2.2.** Test-set prediction of the age-associated label

| Group | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| ~34 | .757 | .842 | .797 |
| 65+ | .818 | .725 | .769 |

In the top-50 correct 65+ cases, activation tended to lie on large overlay text and on people, including faces and upper bodies. In the top-50 correct ~34 cases, activation was more widely spread across background, people, text, and objects (Figure 1). The main text shows a small set of exemplars; the full grids are in Appendix A.

**Figure 1.** Grad-CAM exemplars from correctly classified, channel-deduplicated top-50 cases.

(a) 65+: activation on overlay text and people.

(b) ~34: activation spread more widely across the scene.

Table 2.3 reports where predicted-class heatmap energy fell on the same top-50 images. Figures are descriptive; we do not test ~34 versus 65+ differences on these n = 50 sets. In 65+ cases, text and person together held 56.1% of heatmap energy (text 31.0%, person 25.1%). Text occupied 33.9% of the frame, so it is a large part of the layout rather than a small hot spot (concentration = 0.86). Person occupied 20.9% of the frame but 31.4% of pixels with CAM ≥ 0.5 (concentration = 1.42). In ~34 cases, 72.4% of heatmap energy and 74.5% of high-activation pixels fell on the background. Person concentration for ~34 is also above 1 (1.61) but varies widely across images, so we do not treat it as a stable group pattern.

**Table 2.3.** Grad-CAM energy by exclusive ROI (top-50 correct; predicted-class heatmap)

| Group | ROI | Area | Energy share | Concentration (share / area) | Share of pixels with CAM ≥ 0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| 65+ | text | 33.9% | 31.0% | 0.86 | 28.3% |
| 65+ | person | 20.9% | 25.1% | 1.42 | 31.4% |
| 65+ | background | 45.1% | 43.9% | 0.90 | 40.3% |
| ~34 | text | 6.2% | 7.0% | 1.06 | 5.6% |
| ~34 | person | 19.3% | 20.6% | 1.61 | 19.8% |
| ~34 | background | 74.5% | 72.4% | 0.98 | 74.5% |

*Note.* Overlapping pixels are assigned to text, then person, then background. Concentration > 1 means more heatmap energy than expected from area alone.

Deletion and insertion curves for the 65+ top-50 sample were consistent with a usable map (deletion AUC = .494; insertion AUC = .725; Appendix A). After area normalization, person had the highest deletion and insertion importance. For ~34, predicted probability stayed high after important regions were removed (deletion AUC = .871). That may reflect evidence spread across the image, or saturated probabilities. We therefore do not interpret ~34 ROI deletion/insertion scores in the main text.

### Where later visual measures are focused

The maps do not add new feature types. They assign emphasis among cues that later sections already measure. Because 65+ activation concentrates on overlay text and on people, later analyses give priority to text area, text–background contrast, person area, person count, facial expression, and head direction. Estimated face age is added as a related person measure; Grad-CAM does not itself estimate age. Because ~34 activation spreads across the scene, later analyses also read image captions as semantic units (people, actions, objects, expression, and setting), rather than restricting measurement to a single boxed object. Global color statistics and an AI-generation score remain conventional whole-image measures. They are not derived from Grad-CAM. Title length, form, grammar, and vocabulary are a separate linguistic layer, used to test whether the information-heavy presentation seen in many 65+ thumbnails also appears in language.

---

## Appendix A. Grad-CAM faithfulness and full grids

Deletion removes pixels in order of Grad-CAM importance and records the drop in predicted probability for the target class. Insertion starts from a blurred image and restores the same pixels in the same order. For the 65+ top-50 correct sample, deletion AUC was .494 and insertion AUC was .725. For the ~34 top-50 correct sample, deletion AUC was .871 and insertion AUC was .900.

Table A1 uses the original overlapping OCR and YOLO boxes (text + person + background can exceed 100%). delN = (original probability − deletion probability) / area; insN = insertion probability / area. Person has the highest area-normalized scores. These figures complement Table 2.3 and are not a second area definition for the main argument.

**Table A1.** Overlapping ROI area and area-normalized deletion/insertion (65+ correct, n = 50)

| ROI | Area | Deletion (raw) | Insertion (raw) | delN | insN |
| --- | ---: | ---: | ---: | ---: | ---: |
| text | 33.9% | .948 | .521 | 0.15 | 1.54 |
| person | 25.9% | .854 | .543 | 0.56 | 2.10 |
| background | 45.1% | .889 | .566 | 0.24 | 1.26 |

The full 50-image Grad-CAM grids for correct ~34 and correct 65+ are omitted from the main text. Incorrect high-confidence cases were generated in the same pipeline and are not interpreted.

*Model identifier (for replication):* `facebook/dinov3-vitb16-pretrain-lvd1689m`.
