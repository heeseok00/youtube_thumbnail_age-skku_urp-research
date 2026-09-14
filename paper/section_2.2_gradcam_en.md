# 2.2 and Appendix B (paste-ready)

Appendix letters: A = data collection; B = Grad-CAM; C = visual features; D = semantics; E = textual features; F = lexical visualization.

After faithfulness moved into Section 2.2:
- Main text: Figure~\ref{fig:gradcam}, Figure~\ref{fig:faithfulness}, Table~\ref{tab:faithfulness}
- Appendix B: Table~\ref{tab:b1}, Table~\ref{tab:b2}, Figure~\ref{fig:b1} (18--34 grid), Figure~\ref{fig:b2} (65+ grid)

Delete the old 2.2 block (from `\subsection{2.2 ...}` through the paragraph that ends with Section~2.4) and the old Appendix B block (from the B counter reset through the end of Appendix B, before Appendix C). Also remove any Figure 1 `figure*` that still sits inside 2.1.

## Paste: Section 2.2

```latex
\subsection{2.2 Locating Regions for Visual Feature Measurement via Grad-CAM}

To examine age-associated differences in the visual presentation of YouTube thumbnails, we first sought to identify the image regions that distinguish the two audience groups without relying solely on a predefined set of visual features. We therefore trained a thumbnail-only classifier to distinguish the 18--34 and 65+ groups and applied Grad-CAM \citep{selvaraju2017} to explore which image regions contributed to the model's predictions. Grad-CAM visualizes the regions that contribute most strongly to a model's prediction for a given class, allowing us to identify where age-discriminative information appears before conducting detailed visual-feature analysis.

Image classification used a frozen DINOv3 ViT-B/16 backbone \citep{simeoni2025} with a classification head trained on the 9,586 thumbnails in Table~\ref{tab:subcats}. Age labels were assigned at the channel level, and the model predicted the corresponding label from the thumbnail alone. We used an 8:2 video-level split stratified by age label. The model achieved a test-set accuracy of 78.4\% and a balanced accuracy of .783, indicating that thumbnail images contain visual information predictive of the channel-level audience-age label. Token handling, the top-50 sampling procedure, OCR and YOLO thresholds, the ROI overlap rule, and training hyperparameters are reported in Appendix~B. Per-category accuracy and class-wise precision, recall, and F1 are reported in Table~\ref{tab:b1}.

Among correctly classified high-confidence cases, the two groups showed different localization patterns. For the 65+ group, activation tended to concentrate on large on-image text regions and people, including faces and upper bodies. For the 18--34 group, activation was distributed more broadly across backgrounds, people, text, and objects (Figure~\ref{fig:gradcam}). In the same top-50 samples, text and person regions together accounted for 56.1\% of the Grad-CAM heatmap energy in the 65+ group, whereas 72.4\% of the heatmap energy in the 18--34 group fell on background regions; full exclusive-ROI statistics are reported in Table~\ref{tab:b2} in Appendix~B.

\begin{figure*}[t]
\centering
\includegraphics[width=0.48\textwidth]{fig1_younger_gradcam.png}\hfill
\includegraphics[width=0.48\textwidth]{fig1_older_gradcam.png}
\caption{Grad-CAM exemplars from correctly classified, channel-deduplicated top-50 cases (one thumbnail per category). Left: 18--34, activation spread more widely across the scene. Right: 65+, activation concentrated on text and people. Columns: Original thumbnail, Grad-CAM overlay, Grad-CAM heatmap. Rows: EDU, HEALTH, LIFESTYLE, SOCIETY.}
\label{fig:gradcam}
\end{figure*}

We further evaluated the faithfulness of these localization patterns using deletion and insertion tests. Deletion progressively removes pixels in order of Grad-CAM importance, whereas insertion begins with a blurred image and progressively restores the same pixels. The 65+ group yielded a deletion AUC of .494 and an insertion AUC of .725, while the 18--34 group yielded corresponding AUCs of .871 and .900 (Figure~\ref{fig:faithfulness}). ROI-level tests using overlapping OCR and YOLO boxes showed a similar contrast (Table~\ref{tab:faithfulness}): for the 65+ group, masking any single ROI lowered the predicted probability, with the largest drop for person; for the 18--34 group, predicted probability remained near baseline after masking any single ROI. The large 18--34 area-normalized insertion value for text should not be read as a ranking of ROI importance; it follows from that region's small area when predicted probability is already near ceiling. These results are consistent with more localized evidence for the 65+ group and more distributed, redundant evidence for the 18--34 group. Additional Grad-CAM examples, with three thumbnails per category, are provided in Figures~\ref{fig:b1} and~\ref{fig:b2} in Appendix~B.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth,height=0.36\textheight,keepaspectratio]{figB1_delins_2x2.png}
\caption{Deletion and insertion curves for the top-50 correctly classified samples. Top: 18--34; bottom: 65+. Left: deletion; right: insertion.}
\label{fig:faithfulness}
\end{figure*}

\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{llrrrrr}
\toprule
\textbf{Group} & \textbf{ROI} & \textbf{Area} & \textbf{Del.} & \textbf{Ins.} & \textbf{delN} & \textbf{insN} \\
\midrule
65+ & text & 33.9\% & .948 & .521 & 0.15 & 1.54 \\
65+ & person & 25.9\% & .854 & .543 & 0.56 & 2.10 \\
65+ & background & 45.1\% & .889 & .566 & 0.24 & 1.26 \\
18--34 & text & 6.2\% & 1.000 & .989 & 0.00 & 15.95 \\
18--34 & person & 20.6\% & .997 & .993 & 0.02 & 4.82 \\
18--34 & background & 74.5\% & .974 & .995 & 0.03 & 1.33 \\
\bottomrule
\end{tabular}
\caption{ROI-level deletion and insertion results for the top-50 correctly classified samples ($n=50$ per group). Regions are overlapping OCR and YOLO boxes, so area shares need not sum to 100\%. Del.\ and Ins.\ are predicted probabilities after ROI deletion and insertion. delN = (baseline probability $-$ deletion probability) / area; insN = insertion probability / area. Baseline probability was .999 for 65+ and 1.000 for 18--34.}
\label{tab:faithfulness}
\end{table}

These localization patterns guided the subsequent visual and semantic analyses. Because activation for the 65+ group often concentrated on on-image text and people, we examined text prominence and text--background contrast, as well as person area, person count, and estimated face age. We complemented these region-focused measures with global visual properties, including color characteristics and the probability of AI-generated imagery, to characterize the overall visual form of thumbnails. Because activation for the 18--34 group was more broadly distributed across the scene, we additionally examined scene-level semantics using VLM-generated captions describing people, expressions, actions, objects, and settings. In this way, the Grad-CAM analysis served as an exploratory step that guided the visual-feature analysis in Section~2.3 and the semantic analysis in Section~2.4.
```

## Paste: Appendix B

Replace from the B counter reset through the end of Appendix B (before Appendix C).

```latex
\setcounter{figure}{0}
\setcounter{table}{0}
\renewcommand{\thefigure}{B\arabic{figure}}
\renewcommand{\thetable}{B\arabic{table}}

\section{Appendix B. Grad-CAM Implementation, ROI Tables, and Full Grids}

\paragraph{Implementation.}
Grad-CAM was applied to the last Transformer block. CLS and register tokens were dropped and the remaining patch tokens were reshaped into a spatial map. Heatmaps were computed for the predicted class. For qualitative inspection we ranked correctly classified 18--34 and 65+ test images by prediction confidence, kept one image per channel, and used the top 50 in each group. Figure~\ref{fig:gradcam} shows the clearest instance of each group's pattern in that sample: for 65+, the image with the largest combined text and person share of heatmap energy; for 18--34, the image with the most dispersed heatmap, measured as the image fraction needed to accumulate half of the heatmap energy.

Text, person, and background regions were obtained with EasyOCR (Korean and English; confidence $\geq$ 0.3) and YOLOv8 person boxes (confidence $\geq$ 0.3). Overlapping pixels were assigned in the order text, then person, then background, so exclusive shares sum to 1. The classifier was trained for 5 epochs with Adam (learning rate = 0.001; batch size = 16; random seed = 42). Thumbnails were resized to $224 \times 224$. The backbone identifier is \url{facebook/dinov3-vitb16-pretrain-lvd1689m}.

Test-set accuracy was .784 overall, .822 for EDU, .721 for HEALTH, .867 for LIFESTYLE, and .749 for SOCIETY. Recall was higher for 18--34 (.842) than for 65+ (.725), so more 65+ images were labeled 18--34 (257) than the reverse (150). Class-wise scores are in Table~\ref{tab:b1}.

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
\label{tab:b1}
\end{center}

\paragraph{Exclusive ROI energy.}
Table~\ref{tab:b2} reports where predicted-class heatmap energy fell on the same top-50 correct images. In 65+ cases, text and person together held 56.1\% of heatmap energy (text 31.0\%, person 25.1\%). Text occupied 33.9\% of the frame, so it is a large part of the layout rather than a small hot spot (concentration = 0.86). Person occupied 20.9\% of the frame but 31.4\% of pixels with CAM $\geq$ 0.5 (concentration = 1.42), so discriminative information was more densely packed in that smaller region. In 18--34 cases, 72.4\% of heatmap energy and 74.5\% of high-activation pixels fell on the background.

\begin{center}
\small
\setlength{\tabcolsep}{3.5pt}
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
\captionof{table}{Grad-CAM energy by exclusive ROI (top-50 correct; predicted-class heatmap). Concentration is energy share divided by area. Overlapping pixels are assigned to text, then person, then background.}
\label{tab:b2}
\end{center}

\paragraph{Faithfulness evaluation.}
Deletion and insertion curves and ROI-level overlapping-box results are reported in Figure~\ref{fig:faithfulness} and Table~\ref{tab:faithfulness} in Section~2.2. For the deletion test, pixels were removed in descending order of Grad-CAM importance and the predicted probability of the target class was recorded after each removal step. For insertion, the procedure began with a blurred image and restored pixels in the same importance order. ROI-level tests used the original overlapping OCR and YOLO regions: deletion masked a given ROI, whereas insertion retained only that ROI. Because these regions can overlap, their area shares need not sum to 100\%.

\paragraph{Additional Grad-CAM Exemplars.}
Additional examples from the same channel-deduplicated top-50 samples are shown in Figures~\ref{fig:gradcam-younger-grid} and~\ref{fig:gradcam-older-grid}, with two thumbnails per category that are distinct from Figure~\ref{fig:gradcam}. The 18--34 examples illustrate activation distributed across multiple parts of the scene, whereas the 65+ examples more often show activation around large thumbnail text and people.

% Do not use figure[t] here: it defers B1/B2 and lets Appendix C start beside them.
\twocolumn[{%
\centering
\setlength{\tabcolsep}{8pt}
\begin{tabular}{@{}>{\centering\arraybackslash}p{0.48\textwidth}>{\centering\arraybackslash}p{0.48\textwidth}@{}}
\includegraphics[width=\linewidth]{figA3_younger_grid.png}
\captionof{figure}{Grad-CAM for correctly classified 18--34 thumbnails (two per category, distinct from Figure~\ref{fig:gradcam}; rows: EDU, HEALTH, LIFESTYLE, SOCIETY). Each pair shows the original thumbnail and the predicted-class overlay.}
\label{fig:gradcam-younger-grid}
&
\includegraphics[width=\linewidth]{figA2_older_grid.png}
\captionof{figure}{Grad-CAM for correctly classified 65+ thumbnails (two per category, distinct from Figure~\ref{fig:gradcam}; rows: EDU, HEALTH, LIFESTYLE, SOCIETY). Each pair shows the original thumbnail and the predicted-class overlay.}
\label{fig:gradcam-older-grid}
\end{tabular}

\setcounter{figure}{0}
\setcounter{table}{0}
\renewcommand{\thefigure}{C\arabic{figure}}
\renewcommand{\thetable}{C\arabic{table}}
\section{Appendix C. Continuous Visual Feature Details}
}]
```
