## Numbering (keep current Appendix A labels)

Main text
- Table 1: subcategory counts (Section 2.1, unchanged)
- Figure 1: compact 2x4 Grad-CAM overlays (main text only)

Appendix A, already in the PDF
- Figure A1: deletion and insertion curves
- Table A1: overlapping ROI deletion/insertion
- Figure A2: 65+ grids, three per category
- Figure A3: 18--34 grids, three per category

Appendix A, moved from the main text
- Table A2: test-set precision, recall, and F1 (was Table 2)
- Table A3: exclusive ROI energy (was Table 3)

No Figure A4. Extra exemplars are already Figures A2 and A3.

---

## 2.2 Locating Regions for Visual Feature Measurement via Grad-CAM

Thumbnails associated with the two age groups can be compared on familiar visual cues, including people, text, and color. A thumbnail, however, layers people, text, and background in a single frame, so the same feature yields different values depending on which region it is read from. We therefore first train a thumbnail-only classifier to separate the two age groups, then use Grad-CAM \citep{selvaraju2017} to locate the regions the model relies on. Those regions set the focus for the visual feature measurements in Section 2.3.

Image classification used a frozen DINOv3 ViT-B/16 backbone \citep{simeoni2025} and a classification head trained to separate 18--34 from 65+ on the 9,586 thumbnails in Table~\ref{tab:subcats}. Age labels were assigned at the channel level, and the model predicted that label from the thumbnail alone. We used an 8:2 video-level split stratified by age label. Test-set accuracy was 78.4\% (balanced accuracy .783), so the thumbnail carries a discriminative signal associated with audience age, which is what makes the heatmaps worth reading. Token handling, the top-50 sampling rule, OCR and YOLO thresholds, the ROI overlap rule, and training hyperparameters are reported in Appendix~A. Per-category accuracy and class-wise precision, recall, and F1 are in Table~A2.

In correctly classified high-confidence cases, 65+ activation concentrated on large text and on people, including faces and upper bodies, whereas 18--34 activation spread more widely across background, people, text, and objects (Figure~\ref{fig:gradcam}). On the same top-50 samples, text and person together held 56.1\% of 65+ heatmap energy, while 72.4\% of 18--34 heatmap energy fell on the background (Table~A3). Deletion and insertion tests of explanation faithfulness are shown in Figure~A1 and Table~A1. Additional exemplars, three per category, are in Figures~A2 and~A3.

Later analyses follow these localizations. Where activation concentrates on thumbnail text, we measure text area and text--background contrast. Where it concentrates on people, we measure person area and estimated face age. Where it spreads across the scene, we compare captions describing people, actions, objects, and setting.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig1_gradcam_compact.png}
\caption{Grad-CAM overlays for correctly classified, channel-deduplicated top-50 cases (one image per category). Top: 65+, activation on text and people. Bottom: 18--34, activation spread more widely across the scene. Columns: EDU, HEALTH, LIFESTYLE, SOCIETY.}
\label{fig:gradcam}
\end{figure}

---

## Appendix A. Grad-CAM Implementation, ROI Tables, and Full Grids

Prepend the implementation paragraph and Tables A2--A3. Do not renumber Figure A1, Table A1, Figure A2, or Figure A3.

\paragraph{Implementation.}
Grad-CAM was applied to the last Transformer block. CLS and register tokens were dropped and the remaining patch tokens were reshaped into a spatial map. Heatmaps were computed for the predicted class. For qualitative inspection we ranked correctly classified 18--34 and 65+ test images by prediction confidence, kept one image per channel, and used the top 50 in each group. Figure~\ref{fig:gradcam} shows the clearest instance of each group's pattern in that sample: for 65+, the image with the largest combined text and person share of heatmap energy; for 18--34, the image with the most dispersed heatmap, measured as the image fraction needed to accumulate half of the heatmap energy.

Text, person, and background regions were obtained with EasyOCR (Korean and English; confidence $\geq$ 0.3) and YOLOv8 person boxes (confidence $\geq$ 0.3). Overlapping pixels were assigned in the order text, then person, then background, so exclusive shares sum to 1. The classifier was trained for 5 epochs with Adam (learning rate = 0.001; batch size = 16). Thumbnails were resized to 224 $\times$ 224. The backbone identifier is \url{facebook/dinov3-vitb16-pretrain-lvd1689m}.

Test-set accuracy was .784 overall, .822 for EDU, .721 for HEALTH, .867 for LIFESTYLE, and .749 for SOCIETY. Recall was higher for 18--34 (.842) than for 65+ (.725), so more 65+ images were labeled 18--34 (257) than the reverse (150). Class-wise scores are in Table~A2.

\captionof{table}{Test-set prediction of the age-associated label.}
\label{tab:a2}

Then the existing deletion/insertion paragraph, Figure A1, Table A1, and the two-group interpretation.

\paragraph{ROI energy.}
Table~A3 reports where predicted-class heatmap energy fell on the top-50 correct images. In 65+ cases, text occupied 33.9\% of the frame, so it is a large part of the layout rather than a small hot spot (concentration = 0.86). Person occupied 20.9\% of the frame but 31.4\% of pixels with CAM $\geq$ 0.5 (concentration = 1.42). In 18--34 cases, 72.4\% of heatmap energy and 74.5\% of high-activation pixels fell on the background.

\captionof{table}{Grad-CAM energy by exclusive ROI (top-50 correct; predicted-class heatmap). Concentration is energy share divided by area. Overlapping pixels are assigned to text, then person, then background.}
\label{tab:a3}

Then Figures A2 and A3, unchanged.
