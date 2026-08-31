## 2.2 Bottom-up Exploratory Analysis via Grad-CAM

### Research Objective

The central question of this study is how the multimodal style of YouTube thumbnails and titles differs by age group. If the analysis begins with a predefined set of features such as text, people, and color, its scope is limited to those features, and other visual cues used for age classification may be missed. Before the main feature analysis, we therefore trained a model to distinguish the 34-and-under (~34) and 65-and-over (65~) groups from the thumbnail alone, and used Grad-CAM to inspect which image regions the model treated as evidence for age classification.

Grad-CAM is an explanation method that shows, as a heatmap, which parts of an image the model used as its main evidence when predicting a given class. The regions identified here were not taken as a conclusion about age-related style. They were used to set candidate features for later tests. If a person region is highlighted, for example, person count, facial expression, and head direction can be taken up next. If a text region is highlighted, text area and text–background contrast can be measured. If activation spreads across the scene, the meaning of people, actions, and setting can be examined.

### Data and Model Configuration

The classifier used 9,416 thumbnails collected from four categories: EDU, HEALTH, LIFESTYLE (meditation), and SOCIETY (~34: 4,751; 65~: 4,665). The data were split 8:2 so that the two age groups kept similar proportions in the training and test sets (7,532 training images, 1,884 test images; random seed = 42). The category-level sample is reported in the data collection section. Image classification used the pretrained vision model DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`). A pretrained vision model learns general visual structure, including shape, texture, objects, and spatial layout, from large image collections. To use these representations, the DINOv3 backbone was frozen and a classification head was added to separate ~34 from 65~. The head was trained for 5 epochs (Adam, learning rate = 0.001, batch size = 16).

### Explainability Evaluation

Grad-CAM was applied to the last Transformer block of DINOv3. For qualitative analysis, the test set was divided into three groups: correctly classified ~34, correctly classified 65~, and incorrect predictions. Cases in each group were ranked by prediction confidence. To reduce repeated inclusion of one channel's visual style, duplicate channels were removed, and the top 50 cases were retained in each group. The faithfulness of the Grad-CAM explanations was evaluated with deletion and insertion analyses (Appendix). Deletion measures how quickly the predicted probability of the target class falls when pixels are removed in order of Grad-CAM importance. Insertion measures how quickly that probability rises when the same pixels are restored, in the same order, on a blurred image. In addition, EasyOCR and YOLOv8 were used to partition each thumbnail into text, person, and background ROIs. We then compared changes in predicted probability when each ROI was removed, and when only that ROI was retained.

### Age Classification Performance

**Table 1.** Test-set classification performance

| Group | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| ~34 | .757 | .842 | .797 |
| 65~ | .818 | .725 | .769 |

Test-set accuracy was .784, and balanced accuracy was .783. The two groups could be separated from the thumbnail image alone, without predefined features. This indicates that visual style signals associated with age are present in the images. Accuracy by category was LIFESTYLE (meditation) .867, EDU .822, SOCIETY .749, and HEALTH .721. Misclassifications from 65~ to ~34 (257) also outnumbered those from ~34 to 65~ (150). In borderline cases, 65~ thumbnails were more often judged as the younger group.

### Grad-CAM and ROI Results

In correctly classified 65~ cases, activation tended to concentrate around large overlay text and around people, including faces and upper bodies. In correctly classified ~34 cases, activation was more widely distributed across background, people, text, and objects, rather than concentrated in one region (Figure 1). This observation is limited to the channel-deduplicated top 50 cases with high prediction confidence. It is not treated as a general property of each age group.

**Figure 1.** Grad-CAM on correctly classified cases by age group.

(a) In 65~ correct cases, activation concentrated around overlay text and people.

(b) In ~34 correct cases, activation was more widely distributed across elements of the scene.

To quantify where the heatmap energy fell, each top-50 correct Grad-CAM map was overlapped with exclusive text, person, and background masks (overlapping pixels were assigned in that order). In 65~ cases, text and person together accounted for 56.1% of heatmap energy (text 31.0%, person 25.1%), and 59.7% of pixels with CAM ≥ 0.5. Person occupied 20.9% of the frame but 31.4% of those high-activation pixels (concentration = 1.42). In ~34 cases, 72.4% of heatmap energy and 74.5% of high-activation pixels fell on the background (Table 2).

**Table 2.** Grad-CAM energy by ROI on the top-50 correct cases (predicted-class heatmap; exclusive masks)

| Group | ROI | Area | Energy share | Concentration (share / area) | Share of pixels with CAM ≥ 0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| 65~ | text | 33.9% | 31.0% | 0.86 | 28.3% |
| 65~ | person | 20.9% | 25.1% | 1.42 | 31.4% |
| 65~ | background | 45.1% | 43.9% | 0.90 | 40.3% |
| ~34 | text | 6.2% | 7.0% | 1.06 | 5.6% |
| ~34 | person | 19.3% | 20.6% | 1.61 | 19.8% |
| ~34 | background | 74.5% | 72.4% | 0.98 | 74.5% |

*Note.* Concentration > 1 indicates more heatmap energy than expected from area alone.

Faithfulness checks for the 65~ correct sample (n = 50) gave a deletion AUC of .494 and an insertion AUC of .725 (Appendix). Predicted probability for 65~ fell as important regions were removed and rose as they were restored, which is consistent with a match between the activation map and the model's evidence. For the ~34 sample, predicted probability remained high after important regions were removed (deletion AUC = .871; insertion AUC = .900). Discriminative information may be spread across many regions, or predicted probabilities may be saturated. ~34 ROI results were therefore excluded from the main interpretation.

For the 65~ correct sample, text occupied 33.9% of the frame on average, person 25.9%, and background 45.1%. After area normalization, person had the highest deletion and insertion importance (delN = .56, insN = 2.10; Table 3). Text occupied a large share of the frame and contributed to overall style, whereas person occupied a smaller region in which age-classification information was more concentrated.

**Table 3.** ROI area and area-normalized deletion/insertion importance (65~ correct, n = 50)

| ROI | Area | Deletion (raw) | Insertion (raw) | delN | insN |
| --- | ---: | ---: | ---: | ---: | ---: |
| text | 33.9% | .948 | .521 | 0.15 | 1.54 |
| person | 25.9% | .854 | .543 | 0.56 | 2.10 |
| background | 45.1% | .889 | .566 | 0.24 | 1.26 |

*Note.* Area in this table uses overlapping text and person boxes, as in the original ROI pipeline. delN = (original − deletion) / area; insN = insertion / area. Person area is therefore higher than the exclusive person area in Table 2.

### Transition to Subsequent Analyses

The regions observed with Grad-CAM were used as the starting point for later candidate features. Activation around people was operationalized as person area, person count, facial expression, head direction, and estimated person age. Activation on overlay text was linked to text placement and readability features, including text area and text–background contrast.

Scene-level activation that a single feature cannot capture was converted, through image captioning, into semantic units such as people, actions, objects, expression, and setting. To test whether the information-centered presentation seen in thumbnails also appears in language, titles were analyzed separately for length, form, grammar, and vocabulary.

Grad-CAM therefore functions as an exploratory step. It derives candidate features from the regions the model actually used, rather than stating a conclusion about age-related style. Because the split was made at the video level, videos from the same channel may appear in both the training and test sets. Classification performance is interpreted as a result that supports later feature search, not as a firm estimate of generalization.
