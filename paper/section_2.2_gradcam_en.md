## 2.2 Locating Regions for Visual Feature Measurement via Grad-CAM

Thumbnails associated with the two age groups can be compared on familiar visual cues, including people, text, and color. A thumbnail, however, layers people, text, and background in a single frame, so the same feature yields different values depending on which region it is read from. We therefore first train a thumbnail-only classifier to separate the two age groups, then use Grad-CAM \citep{selvaraju2017} to locate the regions the model relies on. Those regions set the focus for the visual feature measurements in Section 2.3.

Image classification used a frozen DINOv3 ViT-B/16 backbone \citep{simeoni2025} and a classification head trained to separate 18--34 from 65+ on the 9,586 thumbnails in Table~\ref{tab:subcats}. Age labels were assigned at the channel level, and the model predicted that label from the thumbnail alone. We used an 8:2 video-level split stratified by age label. Test-set accuracy was 78.4\% (balanced accuracy .783), so the thumbnail carries a discriminative signal associated with audience age, which is what makes the heatmaps worth reading. Per-category accuracy, class-wise precision and recall, and implementation details are reported in Appendix~B.

In correctly classified high-confidence cases, 65+ activation concentrated on large text and on people, including faces and upper bodies, whereas 18--34 activation spread more widely across background, people, text, and objects (Figure~\ref{fig:gradcam}). On the same top-50 samples, text and person together held 56.1\% of 65+ heatmap energy, while 72.4\% of 18--34 heatmap energy fell on the background. Full ROI tables, faithfulness tests, and additional grids are in Appendix~B.

Later analyses follow these localizations. Where activation concentrates on thumbnail text, we measure text area and text--background contrast; where it concentrates on people, person area and estimated face age; where it spreads across the scene, captions describing people, actions, objects, and setting.

\begin{center}
\includegraphics[width=\columnwidth]{fig1_gradcam_compact.png}
\captionof{figure}{Grad-CAM exemplars from correctly classified, channel-deduplicated top-50 cases (one image per category; columns EDU, HEALTH, LIFESTYLE, SOCIETY). Top: 65+, activation on text and people. Bottom: 18--34, activation spread more widely across the scene.}
\label{fig:gradcam}
\end{center}

---

## Appendix B. Grad-CAM implementation, ROI tables, and full grids

\paragraph{Implementation.}
Grad-CAM was applied to the last Transformer block of the frozen DINOv3 backbone. CLS and register tokens were dropped and the remaining patch tokens were reshaped into a spatial map. Heatmaps were computed for the predicted class. For qualitative inspection we ranked correctly classified 18--34 and 65+ test images by prediction confidence, kept one image per channel, and used the top 50 in each group. Exemplars in Figure~\ref{fig:gradcam} are the clearest instance of each group's pattern within that sample rather than the single most confident prediction: for 65+, the image with the largest combined text and person share of heatmap energy; for 18--34, the image with the most dispersed heatmap, measured as the image fraction needed to accumulate half of the heatmap energy.

Text, person, and background regions were obtained with EasyOCR (Korean and English; confidence $\geq$ 0.3) and YOLOv8 person boxes (confidence $\geq$ 0.3). Overlapping pixels were assigned in the order text, then person, then background, so exclusive shares sum to 1. Training used 5 epochs, Adam, learning rate 0.001, and batch size 16. Thumbnails were resized by the model's image processor to 224 $\times$ 224. The backbone identifier is \url{facebook/dinov3-vitb16-pretrain-lvd1689m}.

Test-set accuracy was .784 overall, .822 for EDU, .721 for HEALTH, .867 for LIFESTYLE, and .749 for SOCIETY. Recall was higher for 18--34 (.842) than for 65+ (.725), so more 65+ images were labeled 18--34 (257) than the reverse (150).

\begin{center}
\small
\begin{tabular}{lrrr}
\hline
\textbf{Group} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\
\hline
18--34 & .757 & .842 & .797 \\
65+ & .818 & .725 & .769 \\
\hline
\end{tabular}
\captionof{table}{Test-set prediction of the age-associated label.}
\label{tab:pred}
\end{center}

\paragraph{ROI energy.}
Table~\ref{tab:roi} reports where predicted-class heatmap energy fell on the top-50 correct images. In 65+ cases, text occupied 33.9\% of the frame, so it is a large part of the layout rather than a small hot spot (concentration = 0.86). Person occupied 20.9\% of the frame but 31.4\% of pixels with CAM $\geq$ 0.5 (concentration = 1.42). In 18--34 cases, 72.4\% of heatmap energy and 74.5\% of high-activation pixels fell on the background.

\begin{center}
\scriptsize
\begin{tabular}{llrrrr}
\hline
\textbf{Group} & \textbf{ROI} & \textbf{Area} & \textbf{Energy} & \textbf{Conc.} & \textbf{CAM $\geq$ .5} \\
\hline
65+ & text & 33.9\% & 31.0\% & 0.86 & 28.3\% \\
65+ & person & 20.9\% & 25.1\% & 1.42 & 31.4\% \\
65+ & background & 45.1\% & 43.9\% & 0.90 & 40.3\% \\
18--34 & text & 6.2\% & 7.0\% & 1.06 & 5.6\% \\
18--34 & person & 19.3\% & 20.6\% & 1.61 & 19.8\% \\
18--34 & background & 74.5\% & 72.4\% & 0.98 & 74.5\% \\
\hline
\end{tabular}
\captionof{table}{Grad-CAM energy by exclusive ROI (top-50 correct). Concentration is energy share divided by area. Overlapping pixels are assigned to text, then person, then background.}
\label{tab:roi}
\end{center}

\paragraph{Faithfulness.}
Deletion removes pixels in order of Grad-CAM importance and records the drop in predicted probability for the target class. Insertion starts from a blurred image and restores the same pixels in the same order. For the 65+ top-50 correct sample, deletion AUC was .494 and insertion AUC was .725. For the 18--34 top-50 correct sample, deletion AUC was .871 and insertion AUC was .900: predicted probability stayed high even after important regions were removed, which is consistent with evidence spread across the image.

[Figure A1: deletion and insertion curves]

Table~\ref{tab:a1} repeats the tests one ROI at a time, using the original overlapping OCR and YOLO boxes (text + person + background can exceed 100\%). Deletion masks the ROI and insertion shows only the ROI. delN = (baseline probability $-$ deletion probability) / area; insN = insertion probability / area.

[Table A1: overlapping ROI deletion/insertion]

The two groups behave differently. In 65+ cases, masking a single ROI lowers the predicted probability and showing a single ROI recovers only about half of it, so person and text each carry part of the evidence, with person the densest per unit area. In 18--34 cases, deletion of any single ROI leaves the probability near baseline and insertion of any single ROI already returns a probability above .98, so no single region is necessary. This is why the 18--34 area-normalized values, especially insN = 15.95 for the small text region, should not be read as a ranking of ROI importance.

[Figures A2--A3: three exemplars per category]
