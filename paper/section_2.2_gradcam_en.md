## 2.2 Locating Regions for Visual Feature Measurement via Grad-CAM

Thumbnails associated with the two age groups can be compared on familiar visual cues, including people, text, and color. A thumbnail, however, layers people, text, and background in a single frame, so the same feature yields different values depending on which region it is read from. This section therefore first identifies which parts of the frame the model relies on when it separates the two age groups, and uses that result to focus the visual feature analysis that follows.

A thumbnail-only classifier is trained to distinguish content associated with people aged 34 and under (~34) from content associated with people aged 65 and over (65+). Grad-CAM (Selvaraju et al., 2017) then shows which regions of the thumbnail the model attends to. Where activation concentrates on thumbnail text, later analyses focus on text area and text–background contrast; where it concentrates on people, on person area, count, expression, and head direction; and where it spreads across the scene, on captions describing people, actions, and setting.

### Data, model, and explanation procedure

**Data and split.** The classifier used 9,586 thumbnails from EDU, HEALTH, LIFESTYLE, and SOCIETY. As defined in Section 2.1, the four samples are the Psychology education, Wellness, Meditation, and Current Affairs subcategories. Each category is balanced 50:50 across the two age groups (Table 2.1). Age labels are assigned at the channel level, as described in the data collection section, and the model predicts that channel-level label from the thumbnail. The data were split 8:2 at the video level, stratified on the age label (random seed = 42).

**Table 2.1.** Thumbnails used in the Grad-CAM classifier, by age

| Category | Subcategory | n | ~34 | ~34 % | 65+ | 65+ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EDU | Psychology education | 2,496 | 1,248 | 50.0 | 1,248 | 50.0 |
| HEALTH | Wellness | 2,500 | 1,250 | 50.0 | 1,250 | 50.0 |
| LIFESTYLE | Meditation | 1,990 | 995 | 50.0 | 995 | 50.0 |
| SOCIETY | Current Affairs | 2,600 | 1,300 | 50.0 | 1,300 | 50.0 |
| Total |  | 9,586 | 4,793 | 50.0 | 4,793 | 50.0 |

**Model.** Image classification used a frozen DINOv3 ViT-B/16 backbone (Siméoni et al., 2025) and a classification head trained to separate ~34 from 65+ (5 epochs; Adam; learning rate = 0.001; batch size = 16). Thumbnails were resized by the model's image processor to 224 × 224. After training, gradients were enabled on the last Transformer block so that Grad-CAM could be computed.

**Grad-CAM and ROIs.** Grad-CAM was applied to the last Transformer block. CLS and register tokens were dropped and the remaining patch tokens were reshaped into a spatial map. Heatmaps were computed for the predicted class. For qualitative inspection we ranked correctly classified ~34 and 65+ test images by prediction confidence, kept one image per channel, and used the top 50 in each group.

To count where heatmap energy falls, text, person, and background regions were obtained with EasyOCR (Korean and English; confidence ≥ 0.3) and YOLOv8 person boxes (confidence ≥ 0.3). Overlapping pixels were assigned in the order text, then person, then background, so shares sum to 1. Deletion and insertion tests of explanation faithfulness are reported in Appendix A (Figure A1), and area-normalized ROI scores for the 65+ top-50 sample are in Table A1.

### Classification and localization results

Test-set accuracy was .784 and balanced accuracy was .783 (Table 2.2). The thumbnail alone therefore carries a signal that separates the two age groups, which is what makes the heatmaps worth reading. Accuracy was .822 for EDU, .721 for HEALTH, .867 for LIFESTYLE, and .749 for SOCIETY. Recall was higher for ~34 (.842) than for 65+ (.725), so more 65+ images were labeled ~34 (257) than the reverse (150).

**Table 2.2.** Test-set prediction of the age-associated label

| Group | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| ~34 | .757 | .842 | .797 |
| 65+ | .818 | .725 | .769 |

In the top-50 correct 65+ cases, activation tended to lie on large text and on people, including faces and upper bodies. In the top-50 correct ~34 cases, activation was more widely spread across background, people, text, and objects. Figure 1 illustrates this pattern with one exemplar from each of EDU, HEALTH, LIFESTYLE, and SOCIETY, drawn from the channel-deduplicated high-confidence top 50. The full 50-image grids are in Appendix A (Figures A2 and A3).

**[Insert Figure 1 here]**  
**Figure 1.** Grad-CAM exemplars from correctly classified, channel-deduplicated top-50 cases (one image per category; order EDU, HEALTH, LIFESTYLE, SOCIETY). Columns: original, predicted-class overlay, predicted-class heatmap.

(a) 65+: activation on text and people.

(b) ~34: activation spread more widely across the scene.

Table 2.3 reports where predicted-class heatmap energy fell on the same top-50 images. In 65+ cases, text and person together held 56.1% of heatmap energy (text 31.0%, person 25.1%). Text occupied 33.9% of the frame, so it is a large part of the layout rather than a small hot spot (concentration = 0.86). Person occupied 20.9% of the frame but 31.4% of pixels with CAM ≥ 0.5 (concentration = 1.42), so discriminative information was more densely packed in that smaller region. In ~34 cases, 72.4% of heatmap energy and 74.5% of high-activation pixels fell on the background.

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

Deletion and insertion curves for the 65+ top-50 sample were consistent with a usable map (deletion AUC = .494; insertion AUC = .725; Figure A1). After area normalization, person had the highest deletion and insertion importance: text shapes the overall impression through its large area, while person holds discriminative information in a smaller region. The same tests on ~34 cases show no single region to be necessary, which again points to evidence distributed across the frame (Table A1).

### Where later visual measures are focused

In sum, discriminative evidence lay on text and people in 65+ thumbnails and across the scene in ~34 thumbnails. Later analyses follow three directions. First, they measure text area and text–background contrast. Second, they measure person area, person count, facial expression, head direction, and estimated face age. Third, they compare scene-level differences through image captions, as semantic units of people, actions, objects, expression, and setting. The analysis then turns to title length, form, grammar, and vocabulary, to see whether the information-dense presentation found in 65+ thumbnails continues in language.

---

## Appendix A. Grad-CAM faithfulness and full grids

Deletion removes pixels in order of Grad-CAM importance and records the drop in predicted probability for the target class. Insertion starts from a blurred image and restores the same pixels in the same order. For the 65+ top-50 correct sample, deletion AUC was .494 and insertion AUC was .725. For the ~34 top-50 correct sample, deletion AUC was .871 and insertion AUC was .900: predicted probability stayed high even after important regions were removed, which is consistent with evidence spread across the image.

**[Insert Figure A1 here]**  
**Figure A1.** Deletion and insertion curves for the top-50 correct samples (left: 65+; right: ~34).

Table A1 repeats the deletion and insertion tests one ROI at a time, using the original overlapping OCR and YOLO boxes (text + person + background can exceed 100%). Deletion masks the ROI and insertion shows only the ROI. delN = (baseline probability − deletion probability) / area; insN = insertion probability / area.

**Table A1.** Overlapping ROI area and area-normalized deletion/insertion (top-50 correct, n = 50 per group)

| Group | ROI | Area | Deletion (raw) | Insertion (raw) | delN | insN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 65+ | text | 33.9% | .948 | .521 | 0.15 | 1.54 |
| 65+ | person | 25.9% | .854 | .543 | 0.56 | 2.10 |
| 65+ | background | 45.1% | .889 | .566 | 0.24 | 1.26 |
| ~34 | text | 6.2% | 1.000 | .989 | 0.00 | 15.95 |
| ~34 | person | 20.6% | .997 | .993 | 0.02 | 4.82 |
| ~34 | background | 74.5% | .974 | .995 | 0.03 | 1.33 |

*Note.* Baseline probability was .999 for 65+ and 1.000 for ~34.

The two groups behave differently. In 65+ cases, masking a single ROI lowers the predicted probability and showing a single ROI recovers only about half of it, so person and text each carry part of the evidence, with person the densest per unit area. In ~34 cases, deletion of any single ROI leaves the probability near baseline and insertion of any single ROI already returns a probability above .98, so no single region is necessary and each is close to sufficient on its own. This is the pattern expected when the evidence is redundant across the frame, and it is why the ~34 area-normalized values, especially insN = 15.95 for the small text region, should not be read as a ranking of ROI importance.

**[Insert Figure A2 here]**  
**Figure A2.** Full 50-image Grad-CAM grid, correctly classified 65+ (`gradcam_correct_65_top50.png`).

**[Insert Figure A3 here]**  
**Figure A3.** Full 50-image Grad-CAM grid, correctly classified ~34 (`gradcam_correct_34_top50.png`).

*Model identifier (for replication):* `facebook/dinov3-vitb16-pretrain-lvd1689m`.
