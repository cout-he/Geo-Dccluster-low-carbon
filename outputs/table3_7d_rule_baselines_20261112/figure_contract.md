# Figure contract

Question: How do the five built-in RBCs compare with the three saved SAC strategies under matched seven-day inputs?

Provisional claim: Controllers trade cost, carbon and service quality; comparisons must show all seeds and distinguish existing execution paths.

Backend: Python/matplotlib, continuing the experiment's Python workflow. Preference stored within this output directory.

Archetype: quantitative grid; standalone numerical table; matched-seed difference plot.

Evidence hierarchy:
- Nine-metric overview: all eight controllers, raw five-seed points, mean and sample SD (ddof=1).
- Paired comparison: RBC minus SAC within each of five identical seeds, for cost, CO2 and SLA. Zero reference; no significance stars.
- Table 3: all nine metrics with mean ± sample SD, editable vector text and CSV source.

Dimensions: overview 11.5 x 8.2 inches, paired differences 11.5 x 7 inches; wide report table 15 x 4.8 inches. These are report exports, not a claim of journal submission compliance.

Export: PDF/SVG with editable text and 300dpi PNG previews. Fonts at least 6pt. All multi-panel axes measured after layout with 1.5pt alignment tolerance. Source validation, PDF font and collision audit, and visual inspection before delivery.

Source: metrics_per_seed.csv, paired_differences_vs_sac.csv, table3_combined_numeric.csv. No exclusions, smoothing, rescaling or synthetic observations. n=5 seeds per controller; descriptive uncertainty only. No new SAC run.

Reviewer risks: five seeds do not establish significance; Most Available uses absolute available cores; RBC direct queue and SAC delivery-buffer timing differ. Low SLA among completed tasks must be read with unfinished-task counts.
