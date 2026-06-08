#!/usr/bin/env python3
"""
generate_reports.py

Generates one PDF clinical report per EEG analysis paradigm, aggregating results
from every patient found in the results directory. Deletes any existing report PDF
before writing the new one.

Uses fpdf2 for direct PNG embedding — no matplotlib rendering per page.

Usage (from the analysis/ directory, with the venv active):
    python generate_reports.py
"""

import json
import textwrap
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    raise SystemExit("fpdf2 not installed — run: pip install fpdf2")
from PIL import Image as _PIL

# ── Paths ──────────────────────────────────────────────────────────────────────
ANALYSIS_ROOT = Path(__file__).parent.resolve()
RESULTS_DIR   = ANALYSIS_ROOT / 'results'
REPORTS_DIR   = ANALYSIS_ROOT / 'reports'
REPORTS_DIR.mkdir(exist_ok=True)

PAGE_W = 8.5    # letter, inches
PAGE_H = 11.0
MARGIN = 0.75
TEXT_W = PAGE_W - 2 * MARGIN   # 7.0 in

# ── Paradigm catalogue ─────────────────────────────────────────────────────────
# Each entry defines the output PDF name, display titles, a plain-English overview
# of the paradigm, and an ordered list of expected figure files with captions and
# citations. The script will include a figure only if the PNG file actually exists.

ANALYSES = {
    'oddball': {
        'pdf_name':   'report_oddball_p300.pdf',
        'title':      'Auditory Awareness Test',
        'full_title': 'Auditory Awareness Test (Oddball P300)',
        'overview': (
            'Background: One of the central challenges in caring for patients with severe brain '
            'injury is determining whether they retain any awareness of their surroundings, even '
            'when they are unable to speak or move. Standard neurological exams rely on visible '
            'responses, but research has shown that some patients who appear completely unresponsive '
            'still have significant brain activity occurring beneath the surface. This test uses '
            'EEG (electroencephalography), which records electrical signals from the brain through '
            'small sensors placed on the scalp, to look for signs of awareness that cannot be seen '
            'from the outside.\n\n'
            'What we did: The patient listened to a series of beeps through headphones. Most beeps '
            'were the same low tone (80% of the time), but occasionally a higher tone was played '
            '(20% of the time). The patient was not asked to do anything.\n\n'
            'What we are looking for: A brain that is aware of its surroundings automatically '
            'reacts when something unexpected happens. When a healthy, aware brain hears the '
            'unexpected higher tone, it generates a characteristic electrical signal roughly '
            '300 to 600 milliseconds later, known as the P300 response. This happens without '
            'any conscious effort. Finding this response in a patient who cannot communicate '
            'suggests their brain is still actively processing the sounds around them, which is '
            'an important indicator of residual awareness.\n\n'
            'Scoring: each rare tone evokes up to four overlapping brain responses in sequence. '
            'N1 (50-100 ms) confirms the auditory pathway is intact and is present even in coma. '
            'MMN (100-200 ms) reflects automatic detection of the tone change and requires no '
            'conscious effort. P3a (200-300 ms) is an automatic orienting response also present '
            'regardless of attention. P3b (300-600 ms) is the only component that requires active '
            'conscious processing and is absent in patients in a vegetative state. The Fischer '
            'hierarchy score counts how many of these four components reach statistical significance '
            'in the expected direction (0 to 4). Higher scores correlate with a higher level of '
            'conscious state and better recovery probability.\n\n'
            'Signal processing: the raw EEG was bandpass filtered (0.1-30 Hz) and re-referenced '
            'to the scalp average across all electrodes. Artifact rejection used autoreject: '
            'a data-driven algorithm that learns a separate amplitude threshold for each electrode '
            'and, when only one or two electrodes are bad in a given epoch, reconstructs them '
            'from neighbouring channels by spherical spline interpolation rather than discarding '
            'the entire epoch. This preserves more trials than a fixed threshold approach, which '
            'matters because each session yields only 10-20 rare tones. Epochs where too many '
            'electrodes were simultaneously bad are fully excluded. A companion bandwidth-comparison '
            'figure re-plots the N1 at Cz under 30, 100, and 200 Hz low-pass settings to show '
            'whether the early response is stable or filter-sensitive. The patient divider page '
            'lists which electrodes were most often corrected and how many epochs were saved '
            'by interpolation versus fully rejected.'
        ),
        'figures': [
            # ── Whole-brain overview first ────────────────────────────────────
            {
                'suffix': '_oddball_butterfly.png',
                'title':  'Whole-Brain Response: All Electrodes',
                'description': (
                    'Pz (red, parietal midline) rising above all other electrodes in the gold window '
                    'while Fz (blue, frontal midline) dips negative simultaneously is the strongest '
                    'evidence of genuine P3b.'
                ),
                'citations': [
                    'Bekinschtein, T. A. et al. (2009). Neural signature of the conscious processing '
                    'of auditory regularities. PNAS, 106(5), 1672-1677.',
                    'Polich, J. (2007). Updating P300: An integrative theory of P3a and P3b. '
                    'Clinical Neurophysiology, 118(10), 2128-2148.',
                ],
            },
            # ── Per-component ERP waveforms ──────────────────────────────────
            {
                'suffix': '_oddball_erp_n1.png',
                'title':  'N1: Obligatory Auditory Response (50–100 ms)',
                'description': (
                    'Both tones drive N1 equally, so the blue (standard) and red (rare) lines dip '
                    'together at Cz inside the lavender window. A near-zero green difference line is '
                    'the expected result and confirms the auditory pathway is intact. T3 and T4 show '
                    'the bilateral auditory cortex response.'
                ),
                'citations': [
                    'Polich, J. (2007). Updating P300: An integrative theory of P3a and P3b. '
                    'Clinical Neurophysiology, 118(10), 2128-2148.',
                ],
            },
            {
                'suffix': '_oddball_n1_bandwidth.png',
                'title':  'N1 Bandwidth Comparison: Is the Early Response Stable?',
                'description': (
                    'The same Cz N1 is replotted after applying 30, 100, and 200 Hz low-pass '
                    'filters. If the early negative deflection stays in the same 50–100 ms window '
                    'across all three panels, the N1 is robust and not just a by-product of one '
                    'particular filter choice. If the shape changes dramatically as the bandwidth '
                    'widens, the result is more likely to reflect noise or filter sensitivity than '
                    'a stable auditory response. This is a descriptive quality check, not a new '
                    'statistical test.'
                ),
                'citations': [
                    'Polich, J. (2007). Updating P300: An integrative theory of P3a and P3b. '
                    'Clinical Neurophysiology, 118(10), 2128-2148.',
                ],
            },
            {
                'suffix': '_oddball_erp_mmn.png',
                'title':  'MMN: Automatic Mismatch Negativity (100–200 ms)',
                'description': (
                    'A negative dip in the green dashed line (rare minus standard) at Fz inside the '
                    'blue window means the brain automatically detected the tone change. Cz is '
                    'shown as a secondary reference. This response does not require conscious '
                    'awareness.'
                ),
                'citations': [
                    'Shao, R. et al. (2025). Mismatch negativity and P300 in diagnosis and prognostic '
                    'assessment of disorders of consciousness. Neurocritical Care.',
                    'Khusakul, S. et al. (2026). Auditory evoked potentials in disorders of '
                    'consciousness: a systematic review. Clinical Neurophysiology Practice.',
                ],
            },
            {
                'suffix': '_oddball_erp_p3a.png',
                'title':  'P3a: Automatic Orienting (200–300 ms)',
                'description': (
                    'A positive rise in the green dashed line at Cz inside the green window, with a '
                    'similar rise at Fz, reflects automatic orienting to the novel tone. This is '
                    'expected but is not by itself a marker of consciousness.'
                ),
                'citations': [
                    'Polich, J. (2007). Updating P300: An integrative theory of P3a and P3b. '
                    'Clinical Neurophysiology, 118(10), 2128-2148.',
                    'Bekinschtein, T. A. et al. (2009). Neural signature of the conscious processing '
                    'of auditory regularities. PNAS, 106(5), 1672-1677.',
                ],
            },
            {
                'suffix': '_oddball_erp_p3b.png',
                'title':  'P3b: Conscious Cognitive Updating (300–600 ms)',
                'description': (
                    'This is the primary clinical finding. A positive rise at Pz (parietal midline) '
                    'inside the gold window, with Fz (frontal midline) dipping negative at the same '
                    'time, is the topographic signature of genuine P3b. That simultaneous '
                    'parietal-positive and frontal-negative pattern cannot be produced by automatic '
                    'processes alone.'
                ),
                'citations': [
                    'Bekinschtein, T. A. et al. (2009). Neural signature of the conscious processing '
                    'of auditory regularities. PNAS, 106(5), 1672-1677.',
                    'Fischer, C. et al. (2016). Long-term prognosis of patients in unresponsive '
                    'wakefulness syndrome after brain injury. NeuroImage Clinical, 12, 462-468.',
                    'Shao, R. et al. (2025). Mismatch negativity and P300 in diagnosis and prognostic '
                    'assessment of disorders of consciousness. Neurocritical Care.',
                ],
            },
            {
                'suffix': '_oddball_p3b_bandwidth.png',
                'title':  'P3b Bandwidth Check: Filter Choice Validation (300-600 ms)',
                'description': (
                    'The Pz waveform is replotted at three progressively narrower low-pass settings '
                    '(30 Hz, 15 Hz, 10 Hz). If the P3b peak in the gold window stays in roughly the '
                    'same place and direction across all three panels, the 10 Hz display filter used '
                    'in the main P3b figure is not distorting the component -- it is simply removing '
                    'alpha-band ripple that would otherwise obscure the slow positive wave. A P3b '
                    'that shrinks or flips sign as the bandwidth narrows would be a red flag that '
                    'the signal is filter-sensitive rather than a true slow potential.'
                ),
                'citations': [
                    'Polich, J. (2007). Updating P300: An integrative theory of P3a and P3b. '
                    'Clinical Neurophysiology, 118(10), 2128-2148.',
                ],
            },
            {
                'suffix': '_oddball_erp_fn.png',
                'title':  'P3b Dipole Index: Parietal vs. Frontal Contrast (300-600 ms)',
                'description': (
                    'Rare-minus-standard difference wave averaged across parietal electrodes '
                    '(P3, Pz, P4) in red and frontal electrodes (F7, F3, Fz, F4, F8) in blue. '
                    'A genuine P3b produces a simultaneous parietal rise and frontal dip, separating '
                    'the two traces in opposite directions across the gold 300-600 ms window. '
                    'A large gap between the lines confirms the full dipole and rules out P3a bleed, '
                    'which produces parietal positivity without a corresponding frontal negativity. '
                    'Traces converging near zero indicates no dipole structure.'
                ),
                'citations': [
                    'Bekinschtein, T. A. et al. (2009). Neural signature of the conscious processing '
                    'of auditory regularities. PNAS, 106(5), 1672-1677.',
                ],
            },
            # ── Per-component null distributions ─────────────────────────────
            {
                'suffix': '_oddball_null_n1.png',
                'title':  'N1 Statistical Test',
                'description': (
                    'The red line falling to the left of the dashed 5th-percentile line means the '
                    'standard-evoked amplitude at Cz is more negative than expected by chance.'
                ),
                'citations': [
                    'Maris, E. and Oostenveld, R. (2007). Nonparametric statistical testing of '
                    'EEG- and MEG-data. Journal of Neuroscience Methods, 164(1), 177-190.',
                ],
            },
            {
                'suffix': '_oddball_null_mmn.png',
                'title':  'MMN Statistical Test',
                'description': (
                    'The red line falling to the left of the dashed 5th-percentile line means the '
                    'rare-minus-standard amplitude at Fz is more negative than expected by chance.'
                ),
                'citations': [
                    'Maris, E. and Oostenveld, R. (2007). Nonparametric statistical testing of '
                    'EEG- and MEG-data. Journal of Neuroscience Methods, 164(1), 177-190.',
                    'Fischer, C. et al. (2016). Long-term prognosis of patients in unresponsive '
                    'wakefulness syndrome after brain injury. NeuroImage Clinical, 12, 462-468.',
                ],
            },
            {
                'suffix': '_oddball_null_p3a.png',
                'title':  'P3a Statistical Test',
                'description': (
                    'The red line falling to the right of the dashed 95th-percentile line means the '
                    'rare-minus-standard amplitude at Cz is more positive than expected by chance.'
                ),
                'citations': [
                    'Maris, E. and Oostenveld, R. (2007). Nonparametric statistical testing of '
                    'EEG- and MEG-data. Journal of Neuroscience Methods, 164(1), 177-190.',
                ],
            },
            {
                'suffix': '_oddball_null_p3b.png',
                'title':  'P3b Statistical Test: Primary Clinical Finding',
                'description': (
                    'The red line falling to the right of the dashed 95th-percentile line means the '
                    'rare-minus-standard amplitude at Pz is more positive than expected by chance. '
                    'The Bonferroni threshold for four simultaneous tests is p < 0.0125. '
                    'The Fischer hierarchy score counts how many of the four components reach '
                    'p < 0.05; a higher score correlates with conscious state and recovery.'
                ),
                'citations': [
                    'Maris, E. and Oostenveld, R. (2007). Nonparametric statistical testing of '
                    'EEG- and MEG-data. Journal of Neuroscience Methods, 164(1), 177-190.',
                    'Fischer, C. et al. (2016). Long-term prognosis of patients in unresponsive '
                    'wakefulness syndrome after brain injury. NeuroImage Clinical, 12, 462-468.',
                ],
            },
            {
                'suffix': '_oddball_null_fn.png',
                'title':  'P3b Dipole Index Statistical Test',
                'description': (
                    'Permutation test for the parietal-frontal contrast: mean(P3, Pz, P4) minus '
                    'mean(F7, F3, Fz, F4, F8) in the 300-600 ms window of the rare-minus-standard '
                    'difference wave. The red line falling to the right of the dashed 95th-percentile '
                    'line means the parietal strip is significantly more positive than the frontal '
                    'strip (p < 0.05), confirming the dipole in both directions simultaneously. '
                    'This test is not part of the four-component Fischer score but provides '
                    'independent confirmation of the P3b topographic signature.'
                ),
                'citations': [
                    'Maris, E. and Oostenveld, R. (2007). Nonparametric statistical testing of '
                    'EEG- and MEG-data. Journal of Neuroscience Methods, 164(1), 177-190.',
                    'Bekinschtein, T. A. et al. (2009). Neural signature of the conscious processing '
                    'of auditory regularities. PNAS, 106(5), 1672-1677.',
                ],
            },
            {
                'suffix': '_oddball_svm_null.png',
                'title':  'Single-Trial SVM: Rare vs. Standard Classification Accuracy',
                'description': (
                    'Each individual epoch (0-600 ms, all channels) is treated as a feature '
                    'vector. A linear classifier is trained on all epochs except one, then asked '
                    'to predict whether the held-out epoch is rare or standard. This is repeated '
                    'for every epoch (leave-one-out cross-validation). The final accuracy is the '
                    'proportion of correct predictions. The null distribution is built by repeating '
                    'the same procedure 500 times with shuffled rare/standard labels. The red line '
                    'falling to the right of the dashed 95th-percentile line means the brain '
                    'responses to rare and standard tones are reliably distinguishable at the '
                    'single-trial level, independent of averaging. Higher accuracy correlates with '
                    'CRS-R score and predicts 3-month recovery outcome.'
                ),
                'citations': [
                    'Claassen, J. et al. (2019). Detection of brain activation in unresponsive '
                    'patients with acute brain injury. New England Journal of Medicine, 380(26), '
                    '2497-2505.',
                ],
            },
            {
                'suffix': '_oddball_gamma.png',
                'title':  'Induced Gamma Power (30–80 Hz): Non-Phase-Locked Conscious Signal',
                'description': (
                    'Left: box plots of mean 30–80 Hz envelope power (200–600 ms post-tone) '
                    'for rare and standard tones. Right: permutation null distribution. '
                    'Unlike the ERP average, this captures induced (non-phase-locked) gamma '
                    'activity that is invisible to averaging. Global ignition theory predicts '
                    'a burst of distributed gamma activity accompanying the conscious P3b — '
                    'higher gamma power for rare vs standard supports this. '
                    'Red line to the right of the dashed 95th-percentile line is significant.'
                ),
                'citations': [
                    'Bekinschtein, T. A. et al. (2009). Neural signature of the conscious '
                    'processing of auditory regularities. PNAS, 106(5), 1672-1677.',
                ],
            },
            {
                'suffix': '_oddball_alpha_corr.png',
                'title':  'Pre-Stimulus Alpha vs P3b Amplitude (Trial-Level Correlation)',
                'description': (
                    'Scatter plot of pre-stimulus alpha power (8–12 Hz, −200 to 0 ms, at Cz) '
                    'against P3b amplitude at Pz (300–600 ms), one point per rare-tone trial. '
                    'A negative correlation — trials with lower pre-stimulus alpha producing '
                    'larger P3b — is present in minimally conscious state patients but absent '
                    'in vegetative state. This is a trial-level measure that does not require '
                    'averaging and remains informative even with very few rare tones. '
                    'The Pearson r and p-value are shown in the title.'
                ),
                'citations': [
                    'Polich, J. (2007). Updating P300: An integrative theory of P3a and P3b. '
                    'Clinical Neurophysiology, 118(10), 2128-2148.',
                ],
            },
            {
                'suffix': '_oddball_pac.png',
                'title':  'Delta-Gamma Phase-Amplitude Coupling (Pz)',
                'description': (
                    'Tort et al. (2010) modulation index: how strongly the amplitude of fast '
                    'gamma oscillations (30-80 Hz) is modulated by the phase of slow delta '
                    'oscillations (1-4 Hz) at Pz, 200-600 ms post-tone. Computed from the '
                    'continuous filtered signal — not the averaged ERP — separately for rare '
                    'and standard tones, then compared by permutation test. Stronger '
                    'delta-gamma coupling for rare than standard tones is reported in MCS '
                    'patients but not UWS patients on standard two-tone oddball recordings — a '
                    'non-phase-locked, single-trial measure that complements the amplitude-'
                    'based ERP findings above.'
                ),
                'citations': [
                    'Tort, A. B. L. et al. (2010). Measuring phase-amplitude coupling between '
                    'neuronal oscillations of different frequencies. Journal of '
                    'Neurophysiology, 104(2), 1195-1210.',
                ],
            },
            {
                'suffix': '_oddball_lzc.png',
                'title':  'Lempel-Ziv Complexity: Rare vs. Standard (0-600 ms)',
                'description': (
                    'Normalized Lempel-Ziv complexity (Zhang et al. 2009) of each individual '
                    'epoch (0-600 ms post-tone, signal binarized by median split, averaged '
                    'across channels), compared between rare and standard tones by permutation '
                    'test. Unlike the averaged ERP, this measure is sensitive to non-phase-'
                    'locked dynamics that cancel out under averaging — a higher complexity '
                    'response to the rare tone would indicate a richer, less stereotyped '
                    'single-trial neural reaction to the unexpected stimulus.'
                ),
                'citations': [
                    'Zhang, Y. et al. (2009). Normalized Lempel-Ziv complexity and its '
                    'application in bio-sequence analysis. Journal of Mathematical Chemistry, '
                    '46(4), 1203-1212.',
                ],
            },
            {
                'suffix': '_oddball_xdawn_null.png',
                'title':  'XDAWN + Riemannian MDM: P300-Specific Classifier',
                'description': (
                    'XDAWN learns spatial filters that maximise the signal-to-noise ratio of '
                    'the ERP template in each training epoch; the resulting augmented covariance '
                    'matrices are classified by Minimum Distance to Riemannian Mean (MDM). '
                    'This pipeline is designed specifically for P300 paradigms and is more '
                    'stable than LinearSVC at small trial counts. Leave-one-out accuracy and '
                    'permutation p-value are shown. A result above the 95th-percentile dashed '
                    'line converges with the SVM finding and provides independent confirmation.'
                ),
                'citations': [
                    'Barachant, A. et al. (2013). Classification of covariance matrices using '
                    'a Riemannian-based kernel for BCI applications. Neurocomputing, 112, 172-178.',
                    'Haufe, S. et al. (2014). On the interpretation of weight vectors of linear '
                    'models in multivariate neuroimaging. NeuroImage, 87, 96-110.',
                ],
            },
            {
                'suffix': '_oddball_svm_haufe.png',
                'title':  'SVM Haufe Spatial Patterns: Where the Classifier Looks',
                'description': (
                    'The classifier trained on all epochs is converted into a brain map using the '
                    'Haufe transform, which shows which scalp regions carry the most discriminative '
                    'information for separating rare from standard tones. Each panel averages the '
                    'pattern across one component window: N1 (50-100 ms), MMN (100-200 ms), '
                    'P3a (200-300 ms), and P3b (300-600 ms). Red regions are positively associated '
                    'with the rare-tone response; blue regions are negatively associated. A '
                    'parietal-positive (red) pattern at P3b time is consistent with P3b and '
                    'converges with the ERP findings. Unlike raw SVM weights, Haufe patterns are '
                    'neurophysiologically interpretable and are not distorted by correlated '
                    'electrode activity.'
                ),
                'citations': [
                    'Haufe, S. et al. (2014). On the interpretation of weight vectors of linear '
                    'models in multivariate neuroimaging. NeuroImage, 87, 96-110.',
                    'Claassen, J. et al. (2019). Detection of brain activation in unresponsive '
                    'patients with acute brain injury. New England Journal of Medicine, 380(26), '
                    '2497-2505.',
                ],
            },
            # ── Whole-brain and spatial views ────────────────────────────────
            {
                'suffix': '_p300_topomap.png',
                'title':  'Where on the Scalp is the Response Strongest?',
                'description': (
                    'Warm colours (red or orange) at Pz (back of the head) in the 300 to 600 ms '
                    'panels, with cool colours (blue) at Fz (forehead) at the same time, is the '
                    'topographic signature of P3b. Earlier panels at 100 ms and 200 ms show N1 '
                    'and MMN centred over the vertex and frontal regions.'
                ),
                'citations': [
                    'Sutton, S., Braren, M., Zubin, J., and John, E. R. (1965). '
                    'Evoked-potential correlates of stimulus uncertainty. Science, 150(3700), 1187-1188.',
                    'Bekinschtein, T. A. et al. (2009). Neural signature of the conscious processing '
                    'of auditory regularities. PNAS, 106(5), 1672-1677.',
                    'Shao, R. et al. (2025). Mismatch negativity and P300 in diagnosis and prognostic '
                    'assessment of disorders of consciousness. Neurocritical Care.',
                ],
            },
            {
                'suffix': '_oddball_johnsen_reactivity.png',
                'title':  'Brain Wave Power Before vs After Each Beep',
                'description': (
                    'Dots falling below the lower dashed line in the alpha (8 to 13 Hz) and beta '
                    '(14 to 30 Hz) rows at central electrodes (C3, Cz, C4) indicate a consistent '
                    'power decrease after each beep, meaning the cortex is actively engaging with '
                    'the auditory stimuli.'
                ),
                'citations': [
                    'Johnsen, L. G. et al. (2014). EEG power spectrum and coherence in disorders '
                    'of consciousness. Clinical Neurophysiology, 125(4), 623-633.',
                    'Della Bella, G. et al. (2025). EEG-based assessment of disorders of '
                    'consciousness: a multicentre study. Communications Biology.',
                ],
            },
        ],
    },

    'language': {
        'pdf_name':   'report_language_tracking.pdf',
        'title':      'Language Comprehension Test',
        'full_title': 'Language Comprehension Test (Speech Rhythm Tracking)',
        'overview': (
            'Background: Understanding whether a patient can still comprehend language is critical '
            'for clinical decision-making, but patients with severe brain injury often cannot '
            'demonstrate comprehension through any outward behaviour. Research has shown that the '
            'brain has a distinctive way of processing speech: when it understands language, its '
            'electrical rhythms naturally synchronise with the rhythmic structure of what is being '
            'said, at the pace of sentences, phrases, and individual words. This synchronisation '
            'can be measured from scalp EEG recordings and provides a window into covert language '
            'processing that does not require any response from the patient. This approach was '
            'validated by Sokoliuk et al. (2021) as a reliable method for detecting language '
            'comprehension in unresponsive patients.\n\n'
            'What we did: The patient listened to recordings of spoken sentences through headphones '
            'across 72 trials, each containing 12 sentences. We recorded the brain\'s electrical '
            'activity throughout.\n\n'
            'What we are looking for: We measure how consistently the brain\'s rhythms lock on '
            'to the pace of speech at three specific rates: the sentence rate (roughly one sentence '
            'every 1.3 seconds, 0.78 Hz), the phrase rate (1.56 Hz), and the word rate (3.125 Hz). '
            'Significant synchronisation at these rates, compared to what would be expected by '
            'chance, suggests the patient is covertly tracking and comprehending the speech.'
        ),
        'figures': [
            {
                'suffix': '_lang_itpc_avg.png',
                'title':  'How Well Does the Brain Track Speech Rhythms?',
                'description': (
                    'This chart shows the degree of brain synchronisation (vertical axis) at each '
                    'frequency from 0.5 to 4 Hz (horizontal axis), averaged across all scalp '
                    'electrodes. The three coloured bands mark the speech rates being tested: '
                    'teal (0.78 Hz, sentence rate), purple (1.56 Hz, phrase rate), and red '
                    '(3.125 Hz, word rate). The dotted horizontal lines show the threshold for '
                    'chance-level synchronisation at each frequency. A peak rising above the '
                    'dotted line at any of the three speech rates, marked with an asterisk (*), '
                    'means the brain is tracking that level of speech structure. The sentence rate '
                    'is the most clinically significant.'
                ),
                'citation': (
                    'Sokoliuk, R. et al. (2021). Two approaches to assess language comprehension '
                    'in unresponsive patients. Annals of Neurology, 90(1), 89-103.'
                ),
            },
            {
                'suffix': '_lang_itpc_channels.png',
                'title':  'Speech Tracking Across Individual Electrodes',
                'description': (
                    'The same analysis shown separately for each of the 19 scalp electrodes. Each '
                    'small chart represents one electrode location; the coloured bands again mark '
                    'the three speech rates. Electrodes over the sides of the head (temporal '
                    'regions, labelled T3, T4, T5, T6) are where we expect the strongest response, '
                    'as those areas sit closest to the brain\'s auditory and language-processing '
                    'regions. Strong, consistent peaks at the temporal electrodes are a particularly '
                    'convincing sign of language comprehension.'
                ),
                'citation': (
                    'Sokoliuk, R. et al. (2021). Two approaches to assess language comprehension '
                    'in unresponsive patients. Annals of Neurology, 90(1), 89-103.'
                ),
            },
            {
                'suffix': '_lang_itpc_topomap.png',
                'title':  'Where on the Scalp is the Speech Tracking Strongest?',
                'description': (
                    'These head maps show the strength of brain synchronisation at each of the '
                    'three speech rates across the entire scalp. Warmer colours (yellow/orange) '
                    'indicate stronger synchronisation at that electrode location. We expect to '
                    'see the strongest activity over the sides of the head (temporal regions) and '
                    'possibly more on the left side, which handles language in most people. '
                    'Activity concentrated in those areas supports a finding of covert language '
                    'comprehension.'
                ),
                'citation': (
                    'Bekinschtein, T. A. et al. (2009). Neural signature of the conscious '
                    'processing of auditory regularities. PNAS, 106(5), 1672-1677. '
                    'Sokoliuk, R. et al. (2021). Annals of Neurology, 90(1), 89-103.'
                ),
            },
        ],
    },

    'command': {
        'pdf_name':   'report_command_following.pdf',
        'title':      'Command Following Test',
        'full_title': 'Command Following Test (Motor Imagery Brain Response)',
        'overview': (
            'Background: Some patients with severe brain injury retain the ability to understand '
            'and follow instructions internally, even though they cannot produce any visible '
            'movement or speech. This condition is called Cognitive-Motor Dissociation (CMD). '
            'Identifying CMD is clinically important because it indicates a level of awareness '
            'and voluntary control that behavioural assessment alone would miss. Research '
            'published in the New England Journal of Medicine (Claassen et al., 2019) '
            'demonstrated that EEG-based motor imagery tasks can reliably detect CMD in patients '
            'with acute brain injury. When a person imagines moving their hand, even without '
            'actually moving it, the brain generates a distinctive pattern: electrical activity '
            'in the motor regions of the brain quiets down in specific frequency ranges (8 to '
            '30 Hz). This quieting is measurable from scalp electrodes.\n\n'
            'What we did: The patient was asked through headphones to either imagine repeatedly '
            'opening and closing one hand (the "keep" condition) or to rest and clear their mind '
            '(the "stop" condition). No physical movement was expected. We recorded brain '
            'activity throughout and used two methods to detect a response: a direct comparison '
            'of brain wave power between the two conditions, and a machine-learning classifier '
            'trained to tell them apart.\n\n'
            'What we are looking for: A significant difference in brain activity between the '
            '"keep" and "stop" conditions at the motor electrodes (top centre of the head), '
            'particularly a decrease in activity during "keep" compared to "stop." If the '
            'classifier can reliably tell the two conditions apart with above-chance accuracy, '
            'this constitutes evidence of CMD.'
        ),
        'figures': [
            {
                'suffix': '_command_erd.png',
                'title':  'Brain Wave Power: Keep vs Stop',
                'description': (
                    'Top panel: the strength of brain electrical activity at the motor electrodes '
                    '(C3, Cz, C4, positioned over the motor strip at the top of the head) during '
                    'the "keep squeezing" phase (blue) and the "stop" phase (red), across '
                    'frequencies 1 to 40 Hz. Bottom panel: the difference between the two '
                    'conditions in decibels. Negative values (below the dashed line) mean brain '
                    'activity was lower during "keep" than during "stop," which is the expected '
                    'pattern when someone is imagining movement. The gold band (8 to 12 Hz, mu '
                    'rhythm) and green band (14 to 30 Hz, beta rhythm) are the key frequency '
                    'ranges. Consistent negative values in those bands are a positive finding.'
                ),
                'citation': (
                    'Sokoliuk, R. et al. (2021). Two approaches to assess command-following in '
                    'unresponsive patients. Annals of Neurology, 90(1), 89-103.'
                ),
            },
            {
                'suffix': '_command_right_lateralization.png',
                'title':  'Right Hand Command: Which Side of the Brain Responds?',
                'description': (
                    'When a person imagines squeezing their right hand, the motor response should '
                    'be strongest on the left side of the brain (the brain controls the opposite '
                    'side of the body). This figure shows the brain wave difference (keep minus '
                    'stop) separately for three electrodes: C3 (left brain, labelled contra, '
                    'expected to show the strongest drop), Cz (centre), and C4 (right brain, '
                    'labelled ipsi, expected to show less change). If C3 shows a larger negative '
                    'response than C4, this left-right pattern confirms the patient is specifically '
                    'following the right-hand instruction rather than reacting generally to the sounds.'
                ),
                'citation': (
                    'Claassen, J. et al. (2019). Detection of brain activation in unresponsive '
                    'patients with acute brain injury. NEJM, 380(26), 2497-2505.'
                ),
            },
            {
                'suffix': '_command_left_lateralization.png',
                'title':  'Left Hand Command: Which Side of the Brain Responds?',
                'description': (
                    'The same analysis for the left-hand command. Here the response should be '
                    'strongest on the right side of the brain (C4, labelled contra). If both the '
                    'right-hand and left-hand commands produce the correct opposite-side brain '
                    'response, this double dissociation is strong evidence that the patient is '
                    'genuinely following each specific instruction.'
                ),
                'citation': (
                    'Claassen, J. et al. (2019). Detection of brain activation in unresponsive '
                    'patients with acute brain injury. NEJM, 380(26), 2497-2505.'
                ),
            },
            {
                'suffix': '_command_right_tfr.png',
                'title':  'Right Hand Command: When and at What Frequency Does Suppression Emerge?',
                'description': (
                    'This time-frequency map shows brain activity at each electrode across the '
                    'full 10-second imagery window, comparing keep versus stop. Blue regions mean '
                    'brain activity was lower during "keep" than "stop" (suppression); red means '
                    'it was higher. The gold dashed lines mark the mu rhythm (8-12 Hz) and the '
                    'green lines mark the beta rhythm (14-30 Hz). A positive response shows '
                    'clear blue bands in those frequency ranges beginning shortly after the '
                    'command and sustained through the imagery period, strongest at C3 (the '
                    'electrode over the left motor cortex, contralateral to the right hand).'
                ),
                'citation': (
                    'Claassen, J. et al. (2019). Detection of brain activation in unresponsive '
                    'patients with acute brain injury. NEJM, 380(26), 2497-2505.'
                ),
            },
            {
                'suffix': '_command_left_tfr.png',
                'title':  'Left Hand Command: When and at What Frequency Does Suppression Emerge?',
                'description': (
                    'The same time-frequency analysis for the left-hand command. A positive '
                    'response should show suppression (blue) in the mu and beta bands at C4 '
                    '(right motor cortex, contralateral to the left hand).'
                ),
                'citation': (
                    'Claassen, J. et al. (2019). Detection of brain activation in unresponsive '
                    'patients with acute brain injury. NEJM, 380(26), 2497-2505.'
                ),
            },
            {
                'suffix': '_command_svm_null.png',
                'title':  'Can a Computer Tell "Keep" from "Stop" Using the Brain Signal?',
                'description': (
                    'We trained a machine-learning algorithm to read the EEG signal and decide '
                    'whether the patient was in the "keep squeezing" or "stop" phase. Its accuracy '
                    'is expressed as an AUC score: 0.5 means pure chance (the algorithm cannot '
                    'tell the conditions apart), while 1.0 means perfect classification. To '
                    'confirm the result is real and not a statistical fluke, we repeated the test '
                    '500 times with randomly shuffled labels. The blue histogram shows the range '
                    'of scores expected by chance. The red line sitting clearly above the bulk of '
                    'the histogram (p < 0.05) means the brain signal reliably distinguishes the '
                    'two commands, meeting the published criterion for Cognitive-Motor Dissociation '
                    '(Claassen et al., 2019).'
                ),
                'citation': (
                    'Claassen, J. et al. (2019). NEJM, 380(26), 2497-2505. '
                    'Haufe, S. et al. (2014). On the interpretation of weight vectors of linear '
                    'models in multivariate neuroimaging. NeuroImage, 87, 96-110.'
                ),
            },
            {
                'suffix': '_command_psd_features.png',
                'title':  'What the Classifier Sees: Brain Power Across Sub-Epochs and Electrodes',
                'description': (
                    'Each panel shows the raw power measurements that the machine-learning '
                    'algorithm uses to make its decision, across four frequency bands. Each '
                    'column is one 2-second sub-epoch; each row is one electrode. Columns '
                    'alternate between "keep" and "stop" sub-epochs within each trial pair '
                    '(thin vertical lines mark trial boundaries). If the classifier is picking '
                    'up a genuine motor imagery signal, the keep and stop columns should appear '
                    'visibly different in the alpha and beta panels, particularly at the central '
                    'electrodes (C3, Cz, C4). Uniform colouring across all sub-epochs suggests '
                    'the classifier is working at chance level.'
                ),
                'citation': (
                    'Claassen, J. et al. (2019). Detection of brain activation in unresponsive '
                    'patients with acute brain injury. NEJM, 380(26), 2497-2505.'
                ),
            },
            {
                'suffix': '_command_decoding.png',
                'title':  'Decoding Time-Course: Does the Brain Track the Command Sequence?',
                'description': (
                    'Replicating Figure 3 from Claassen et al. (2019). Each numbered unit on '
                    'the x-axis is one keep+stop trial pair. Within each trial the orange-shaded '
                    'left half is the motor imagery (move) period and the unshaded right half is '
                    'the rest period; the solid and dotted lines below the axis mark these '
                    'periods. The y-axis shows the classifier\'s prediction that the brain is in '
                    'the "move" state: values near 1.0 mean the brain signal looked like motor '
                    'imagery, values near 0.0 mean it looked like rest, and 0.5 is chance. A '
                    'patient whose brain is following the commands will show the curve rising '
                    'during orange periods and falling during white periods, producing the '
                    'characteristic oscillating pattern seen in healthy controls in the paper. '
                    'A flat line near 0.5 throughout indicates no detectable response.'
                ),
                'citation': (
                    'Claassen, J. et al. (2019). Detection of brain activation in unresponsive '
                    'patients with acute brain injury. NEJM, 380(26), 2497-2505.'
                ),
            },
            {
                'suffix': '_command_svm_patterns.png',
                'title':  'Which Brain Regions Did the Classifier Use?',
                'description': (
                    'These head maps reveal which scalp locations the algorithm relied on most '
                    'heavily to tell "keep" from "stop," shown separately for four frequency '
                    'bands. Red and blue indicate regions that were most informative; grey '
                    'indicates regions that contributed little. A trustworthy result should show '
                    'the strongest activity over the top centre of the head (the motor strip) in '
                    'the alpha and beta bands (8 to 30 Hz). If the algorithm instead relied '
                    'heavily on forehead electrodes (which pick up eye movement artefacts) or '
                    'the back of the head, the result may not reflect genuine command following.'
                ),
                'citation': (
                    'Haufe, S. et al. (2014). On the interpretation of weight vectors of linear '
                    'models in multivariate neuroimaging. NeuroImage, 87, 96-110.'
                ),
            },
        ],
    },

    'spindles': {
        'pdf_name':   'report_spindles.pdf',
        'title':      'Sleep Spindle Detection',
        'full_title': 'Sleep Spindle Detection (Passive Thalamocortical Integrity)',
        'overview': (
            'Background: Sleep spindles are brief 12-15 Hz oscillatory bursts generated by '
            'thalamocortical circuits, visible as characteristic waxing-and-waning bursts on '
            'scalp EEG lasting 0.5-2 seconds. They require no patient cooperation and can be '
            'detected from any resting EEG recording. Claassen et al. (Nature Medicine 2025, '
            'n=226 acute brain injury) showed that well-formed sleep spindles independently '
            'predict CMD positivity and recovery: 76% of patients with both spindles and CMD '
            'recovered consciousness before hospital discharge.\n\n'
            'Note: This analysis is optional and not run by default. Run with: '
            'python run_all.py --analyses spindles'
        ),
        'figures': [
            {
                'suffix': '_spindles_trace.png',
                'title':  'Resting EEG Spindle Trace (12-15 Hz Envelope)',
                'description': (
                    'Mean 12-15 Hz Hilbert envelope across all scalp electrodes during the '
                    'pre-stimulus resting period. The dashed red line is the detection threshold '
                    '(mean + 2 SD). Gold shading marks each detected spindle (0.5-2 s above '
                    'threshold). More spindle events indicate more intact thalamocortical '
                    'spindle generation.'
                ),
                'citations': [
                    'Claassen, J. et al. (2025). Sleep spindles as a passive predictor of '
                    'command following in acute brain injury. Nature Medicine.',
                ],
            },
            {
                'suffix': '_spindles_summary.png',
                'title':  'Spindle Rate and Mean Amplitude Summary',
                'description': (
                    'Left: spindle rate in events per minute. Right: mean peak envelope '
                    'amplitude. Higher rate and amplitude indicate more robust thalamocortical '
                    'spindle activity. Claassen et al. (2025): 76% of patients with both '
                    'spindles and CMD recovered consciousness before hospital discharge.'
                ),
                'citations': [
                    'Claassen, J. et al. (2025). Sleep spindles as a passive predictor of '
                    'command following in acute brain injury. Nature Medicine.',
                ],
            },
        ],
    },

    'resting': {
        'pdf_name':   'report_resting_state.pdf',
        'title':      'Resting-State EEG Suite',
        'full_title': 'Resting-State EEG Suite (Passive Complexity, Connectivity & Quality)',
        'overview': (
            'Background: The three task-evoked paradigms above all require the patient to stay '
            'awake and engaged through several minutes of stimuli — a real risk in patients '
            'with severe brain injury, whose level of arousal can swing within and across '
            'sessions. The brain\'s spontaneous activity at rest, recorded in the few minutes '
            'before any stimulus is delivered, carries diagnostic information of its own and '
            'needs no task, no patient cooperation, and cannot produce a false negative caused '
            'by drowsiness.\n\n'
            'What we did: The resting EEG segment recorded immediately before the first '
            'stimulus of the session (up to 5 minutes) was analysed with five complementary, '
            'purely passive measures.\n\n'
            'What we are looking for: (1) Signal complexity — permutation entropy and '
            'Lempel-Ziv complexity both quantify how rich and unpredictable the brain\'s '
            'electrical activity is. Della Bella et al. (2025) found this kind of complexity '
            'decreases in a graded fashion across consciousness levels — healthy, minimally '
            'conscious, unresponsive, and acutely injured — and predicts six-month recovery. '
            '(2) EEG microstates — the brain\'s scalp-wide electrical pattern does not drift '
            'continuously but jumps between a small number of stable spatial configurations '
            'tens of times per second; how long the brain dwells in each one is altered in '
            'disorders of consciousness (Michel and Koenig 2018). (3) Weighted symbolic mutual '
            'information (wSMI) — measures how much two brain regions share genuinely joint '
            'information, after excluding the part attributable to a shared source or volume '
            'conduction; reduced sharing is a hallmark of unconsciousness (King et al. 2013). '
            '(4) Background EEG quality screen — a brief automated check for patterns '
            '(suppressed background, inter-hemispheric asymmetry, burst-suppression) that the '
            'Claassen group (medRxiv 2025) found predict zero chance of detecting command '
            'following no matter how the active-task analysis turns out, flagging sessions '
            'whose results deserve extra caution.\n\n'
            'Note: This suite is optional, not run by default, and purely descriptive — none '
            'of these five measures replace or change the active-paradigm findings above; they '
            'add independent, task-free context. Run with: '
            'python run_all.py --analyses resting'
        ),
        'figures': [
            {
                'suffix': '_resting_complexity.png',
                'title':  'Signal Complexity: Permutation Entropy and Lempel-Ziv Complexity',
                'description': (
                    'Two complementary measures of how rich and unpredictable the resting EEG '
                    'is, computed per electrode (bars) with the cross-electrode mean shown as '
                    'a dashed red line. Left: permutation entropy (order-3 ordinal patterns, '
                    'normalized 0-1) — how often the relative ordering of neighbouring samples '
                    'changes over time. Right: normalized Lempel-Ziv complexity — how '
                    'compressible the signal is once reduced to a binary sequence. Both '
                    'increase with the richness of brain dynamics; Della Bella et al. (2025) '
                    'found they decrease in a graded fashion from healthy controls through '
                    'MCS, UWS, and acute injury, and predict six-month recovery.'
                ),
                'citations': [
                    'Della Bella, G. et al. (2025). EEG-based assessment of disorders of '
                    'consciousness: a multicentre study. Communications Biology.',
                    'Bandt, C. and Pompe, B. (2002). Permutation entropy: a natural complexity '
                    'measure for time series. Physical Review Letters, 88(17), 174102.',
                    'Zhang, Y. et al. (2009). Normalized Lempel-Ziv complexity and its '
                    'application in bio-sequence analysis. Journal of Mathematical Chemistry, '
                    '46(4), 1203-1212.',
                ],
            },
            {
                'suffix': '_resting_microstates.png',
                'title':  'EEG Microstates: Recurring Scalp-Wide Activity Patterns',
                'description': (
                    'The resting EEG is segmented into brief (tens-of-milliseconds) periods of '
                    'stable scalp topography, each assigned to one of four data-driven spatial '
                    'classes (top: scalp maps; bottom: how long the brain stays in each class '
                    'on average, and how often per minute it switches into that class). GEV '
                    '(global explained variance) reports how well these four patterns '
                    'summarise the whole recording. These classes are statistical clusters '
                    'specific to this recording, not the canonical A/B/C/D microstate classes '
                    'reported in the literature — no normative template-matching is performed. '
                    'Altered microstate dynamics (duration, occurrence, coverage) are reported '
                    'as a task-independent marker in disorders of consciousness.'
                ),
                'citations': [
                    'Lehmann, D. et al. (1987). EEG alpha map series: brain micro-states by '
                    'space-oriented adaptive segmentation. Electroencephalography and '
                    'Clinical Neurophysiology, 67(3), 271-288.',
                    'Michel, C. M. and Koenig, T. (2018). EEG microstates as a tool for '
                    'studying the temporal dynamics of whole-brain neuronal networks: A '
                    'review. NeuroImage, 180, 577-593.',
                ],
            },
            {
                'suffix': '_resting_wsmi.png',
                'title':  'Weighted Symbolic Mutual Information: Information Sharing Between Regions',
                'description': (
                    'Each cell shows how much genuinely joint information a pair of electrodes '
                    'shares (yellow = more, dark purple = less), after symbolizing the signal '
                    'into short ordinal patterns and excluding symbol pairs that could reflect '
                    'a shared source or volume conduction rather than true information '
                    'exchange. King et al. (2013) showed this measure reliably separates '
                    'conscious from unconscious states and tracks recovery in disorders of '
                    'consciousness — reduced long-range sharing (e.g. frontal-parietal) is '
                    'characteristic of unconsciousness, while preserved sharing supports '
                    'integrated, brain-scale processing.'
                ),
                'citations': [
                    'King, J.-R. et al. (2013). Information sharing in the brain indexes '
                    'consciousness in noncommunicative patients. Current Biology, 23(19), '
                    '1914-1919.',
                ],
            },
            {
                'suffix': '_resting_quality.png',
                'title':  'Background EEG Quality Screen',
                'description': (
                    'A brief automated pre-flight check of the resting background — not a '
                    'clinical EEG read — for three patterns the Claassen group (medRxiv 2025) '
                    'found predict zero chance of detecting command following regardless of '
                    'how the active-task analysis turns out. Left: continuity (fraction of '
                    'windows with normal-amplitude activity; low values suggest a suppressed '
                    'background). Centre: inter-hemispheric asymmetry (right-versus-left '
                    'motor/temporal RMS ratio; values far from 1.0 suggest a lateralised '
                    'abnormality). Right: spread of window-by-window log-power (a bimodal, '
                    'high-spread distribution is characteristic of burst-suppression). A FLAG '
                    'on any panel does not invalidate the active-paradigm results — it means '
                    'they deserve extra caution and ideally corroboration with a repeat '
                    'session.'
                ),
                'citations': [
                    'Claassen, J. et al. (2025). Surface EEG background features as a passive '
                    'screen for cognitive motor dissociation. medRxiv.',
                ],
            },
        ],
    },

    'pooled_oddball': {
        'pdf_name':   'report_pooled_oddball.pdf',
        'title':      'Cross-Patient Pooled Oddball Analysis',
        'full_title': 'Cross-Patient Pooled Oddball Analysis (Cohort-Level P300)',
        'overview': (
            'Background: Per-patient analyses test whether each individual showed a P300 '
            'response. This pooled analysis asks a different question: does the population-level '
            'P300 signature generalise across patients? Leave-one-patient-out (LOPO) cross-'
            'validation trains a XDAWN+MDM classifier on all patients except one, then tests '
            'on the held-out patient. This evaluates cross-patient transfer — the foundation '
            'of building a clinical tool that works on new, unseen patients.\n\n'
            'The normative summary shows where each patient falls on key biomarkers relative '
            'to the cohort: P3b amplitude, Fischer hierarchy score, and rare tone count. '
            'Patients with fewer than 10 rare tones are flagged in red — their results should '
            'be interpreted with caution regardless of the statistical outcome.'
        ),
        'figures': [
            {
                'suffix': 'pooled_lopo_accuracy.png',
                'title':  'Leave-One-Patient-Out Classification Accuracy',
                'description': (
                    'Bar chart showing XDAWN+MDM classification accuracy for each patient '
                    'when the classifier was trained on all other patients and tested on that '
                    'patient. Red bars indicate accuracy above 0.6 (above-chance). The dashed '
                    'line at 0.5 is chance level. Above-chance LOPO accuracy on a held-out '
                    'patient means the population P300 signature transfers to that individual, '
                    'which is the prerequisite for a generalisable clinical tool. The overall '
                    'accuracy and permutation p-value appear in the title.'
                ),
                'citations': [
                    'Barachant, A. et al. (2013). Classification of covariance matrices using '
                    'a Riemannian-based kernel for BCI applications. Neurocomputing, 112, 172-178.',
                ],
            },
            {
                'suffix': 'pooled_normative_summary.png',
                'title':  'Cohort Normative Summary: P3b, Fischer Score, Rare Tone Count',
                'description': (
                    'Left: P3b amplitude at Pz for each patient. The red dashed line is the '
                    'Shao 2025 clinical threshold (1.095 µV). Centre: Fischer hierarchy score '
                    '(0-4 components significant); red dashed line at score 3. Right: number '
                    'of rare tones kept after autoreject; patients in red had fewer than 10 '
                    'rare tones, below the reliable averaging floor. This panel contextualises '
                    'individual patient results within the cohort and flags sessions where '
                    'data quality limits interpretation.'
                ),
                'citations': [
                    'Fischer, C. et al. (2016). Long-term prognosis in unresponsive wakefulness '
                    'syndrome. NeuroImage Clinical, 12, 462-468.',
                    'Shao, R. et al. (2025). Mismatch negativity and P300 in diagnosis and '
                    'prognostic assessment of disorders of consciousness. Neurocritical Care.',
                ],
            },
        ],
    },

    'voice': {
        'pdf_name':   'report_voice_familiarity.pdf',
        'title':      'Loved One Voice Recognition Test',
        'full_title': 'Loved One Voice Recognition Test (Familiarity Brain Response)',
        'overview': (
            'Background: Emotional memory and recognition of familiar people are among the most '
            'deeply preserved cognitive functions, even in patients with significant brain injury. '
            'Research has shown that the brain can produce a measurable electrical response to '
            'the voice of a loved one in patients who are otherwise completely unresponsive. This '
            'response reflects implicit emotional and memory processing that occurs below the '
            'level of conscious, voluntary behaviour. Detecting such a response provides '
            'evidence that aspects of the patient\'s identity, memory, and emotional life remain '
            'neurologically intact.\n\n'
            'What we did: The patient listened to short audio clips alternating between the voice '
            'of a loved one (a familiar person, such as a family member) and the voices of '
            'unknown speakers. No response was required.\n\n'
            'What we are looking for: A stronger brain electrical response to the loved one\'s '
            'voice than to unfamiliar voices, particularly a positive wave between 300 and 600 '
            'milliseconds after the voice begins. This familiarity response suggests the '
            'patient\'s brain is still recognising and emotionally responding to someone they '
            'know, which is a meaningful sign of preserved awareness and memory.'
        ),
        'figures': [
            {
                'suffix': '_voice_erp.png',
                'title':  'Brain Response: Familiar vs Unfamiliar Voice',
                'description': (
                    'This chart shows the average brain electrical activity following the start '
                    'of each voice clip. The red line shows the response to the loved one\'s '
                    'voice; the blue line shows the response to unfamiliar voices. The dashed '
                    'green line is the difference. The gold shaded region (300 to 600 ms) is '
                    'the window where a familiarity response is expected. A positive bump in the '
                    'green line within that window, meaning the brain responded more strongly to '
                    'the familiar voice, is a positive finding.'
                ),
                'citation': (
                    'Perrin, F. et al. (2006). Brain response to one\'s own name in vegetative '
                    'state, minimally conscious state, and locked-in syndrome. '
                    'Archives of Neurology, 63(4), 562-569.'
                ),
            },
            {
                'suffix': '_voice_null.png',
                'title':  'Is the Familiarity Response Real or Due to Chance?',
                'description': (
                    'We randomly shuffled which clips were labelled familiar and which were '
                    'labelled unfamiliar 1,000 times to build a picture of what results we would '
                    'expect by chance. The blue histogram shows this range of chance results. The '
                    'red vertical line shows the patient\'s actual response. The red line sitting '
                    'past the dashed 95th-percentile line means the familiarity response is '
                    'statistically significant and unlikely to have occurred by chance (p < 0.05).'
                ),
                'citation': (
                    'Maris, E. and Oostenveld, R. (2007). Nonparametric statistical testing of '
                    'EEG- and MEG-data. Journal of Neuroscience Methods, 164(1), 177-190.'
                ),
            },
            {
                'suffix': '_voice_topomap.png',
                'title':  'Where on the Scalp is the Familiarity Response Strongest?',
                'description': (
                    'Head maps showing the difference in brain activity (familiar minus unfamiliar) '
                    'at key time points across the 300 to 600 ms window. Warm colours indicate '
                    'stronger activity. A genuine familiarity response is typically strongest at '
                    'the top and back of the head (parietal region). Activity concentrated there, '
                    'rather than scattered across the scalp, confirms the response has the '
                    'expected brain signature.'
                ),
                'citation': (
                    'Fischer, C. et al. (2010). Improved prediction of awakening from severe '
                    'anoxic coma using tree-based classification analysis. Critical Care Medicine, '
                    '38(3), 745-754.'
                ),
            },
            {
                'suffix': '_voice_roc_bootstrap.png',
                'title':  'How Reliably Can We Classify Familiar vs Unfamiliar Trials?',
                'description': (
                    'Left: this curve shows how well the brain response alone can classify each '
                    'individual voice clip as familiar or unfamiliar. A curve that sweeps toward '
                    'the top-left corner indicates good discrimination; the AUC score summarises '
                    'this (0.5 means chance level, 1.0 means perfect). Right: we repeated this '
                    'classification 1,000 times on random subsets of the data (bootstrapping) to '
                    'estimate how reliable the result is. The gold band shows the 95% confidence '
                    'interval. The entire gold band sitting above 0.5 means the brain reliably '
                    'distinguishes familiar from unfamiliar voices across trials.'
                ),
                'citation': (
                    'Hajian-Tilaki, K. (2013). Receiver operating characteristic (ROC) curve '
                    'analysis for medical diagnostic test evaluation. Caspian Journal of Internal '
                    'Medicine, 4(2), 627-635.'
                ),
            },
        ],
    },
}


# ── fpdf2 helpers ──────────────────────────────────────────────────────────────

_CHAR_MAP = str.maketrans({
    "–": "-",    # en dash
    "—": "--",   # em dash
    "−": "-",    # mathematical minus sign
    "‘": "'",    # left single quotation mark
    "’": "'",    # right single quotation mark
    "“": '"',   # left double quotation mark
    "”": '"',   # right double quotation mark
    "…": "...",  # ellipsis
    "≥": ">=",   # greater-than or equal to
    "≤": "<=",   # less-than or equal to
})

def _n(text: str) -> str:
    """Normalise text to Latin-1 safe characters for built-in PDF fonts."""
    return text.translate(_CHAR_MAP)


def _make_pdf() -> FPDF:
    p = FPDF(unit='in', format='letter')
    p.set_auto_page_break(False)
    p.set_margins(0, 0, 0)
    return p


def _img_dims(path: Path) -> tuple:
    """Read PNG header only; return (w_in, h_in)."""
    with _PIL.open(path) as im:
        w_px, h_px = im.size
        dpi_info = im.info.get('dpi', (150, 150))
        dpi = float(dpi_info[0]) if isinstance(dpi_info, (tuple, list)) else float(dpi_info or 150)
        dpi = max(dpi, 1.0)
    return w_px / dpi, h_px / dpi


def _open_img(path: Path):
    """Open and fully decode a PNG once; return (PIL Image, w_in, h_in).

    Callers pass the returned PIL Image to p.image() to avoid a second file read.
    """
    im = _PIL.open(path)
    im.load()
    dpi_info = im.info.get('dpi', (150, 150))
    dpi = float(dpi_info[0]) if isinstance(dpi_info, (tuple, list)) else float(dpi_info or 150)
    dpi = max(dpi, 1.0)
    return im, im.width / dpi, im.height / dpi


def _rule(p: FPDF, y: float) -> None:
    p.set_draw_color(0, 0, 0)
    p.set_line_width(0.008)
    p.line(MARGIN, y, PAGE_W - MARGIN, y)


def _txt(p: FPDF, x: float, y: float, text: str,
         size: int = 10, style: str = '',
         color: tuple = (0, 0, 0), align: str = 'L') -> float:
    """Single line of text; returns line height used."""
    p.set_font('Helvetica', style=style, size=size)
    p.set_text_color(*color)
    lh = size * 1.4 / 72
    p.set_xy(x, y)
    p.cell(w=TEXT_W, h=lh, text=_n(text), align=align)
    return lh


def _wrap_txt(p: FPDF, x: float, y: float, text: str,
              size: int = 10, style: str = '',
              color: tuple = (0, 0, 0),
              max_w: float = TEXT_W, line_spacing: float = 1.55) -> float:
    """Wrapped multi-line text; returns total height consumed."""
    p.set_font('Helvetica', style=style, size=size)
    p.set_text_color(*color)
    lh = size * line_spacing / 72
    p.set_xy(x, y)
    p.multi_cell(w=max_w, h=lh, text=_n(text), align='L')
    return p.get_y() - y


def _place_img(p: FPDF, img_path: Path, top: float, max_h: float) -> None:
    """Embed PNG directly, centred in TEXT_W x max_h box, aspect-ratio preserved.

    Opens the image once via PIL and passes the object to fpdf2 — avoids a second
    file read compared to passing a path string.
    """
    im, w_nat, h_nat = _open_img(img_path)
    aspect = w_nat / h_nat
    if TEXT_W / max_h >= aspect:
        dh, dw = max_h, max_h * aspect
    else:
        dw, dh = TEXT_W, TEXT_W / aspect
    x = MARGIN + (TEXT_W - dw) / 2
    y = top   + (max_h - dh)  / 2
    p.image(im, x=x, y=y, w=dw, h=dh)
    im.close()


# ── Page builders ──────────────────────────────────────────────────────────────

def _fig_citations(fig_def: dict) -> list:
    if 'citations' in fig_def:
        return [c.strip() for c in fig_def['citations'] if c.strip()]
    legacy = fig_def.get('citation', '').strip()
    return [legacy] if legacy else []


def _resting_metadata_note(md: dict) -> str:
    parts = [f'Resting window analysed: {md["rest_duration_min"]:.1f} min.']
    pe  = md.get('permutation_entropy', {}).get('mean')
    lzc = md.get('lzc', {}).get('mean')
    if pe is not None and lzc is not None:
        parts.append(f'Complexity (mean across channels): permutation entropy={pe:.3f}, '
                     f'LZc={lzc:.3f}.')
    ms = md.get('microstates')
    if ms:
        parts.append(f'Microstates: GEV={ms["gev"]:.3f} across {len(ms["classes"])} classes.')
    wsmi = md.get('wsmi', {}).get('mean')
    if wsmi is not None:
        parts.append(f'wSMI mean={wsmi:.4f}.')
    qs = md.get('quality_screen')
    if qs:
        flags = [k for k, v in qs.get('flags', {}).items() if v]
        parts.append(f'Background quality flags: {", ".join(flags) if flags else "none"}.')
    return '  '.join(parts)


def _metadata_note(md: dict) -> str:
    if 'rest_duration_min' in md:
        return _resting_metadata_note(md)
    parts = []
    n_rare_pre  = md.get('n_rare_pre_rejection')
    n_rare_post = md.get('n_rare_post_rejection')
    n_std_pre   = md.get('n_std_pre_rejection')
    n_std_post  = md.get('n_std_post_rejection')
    method      = md.get('rejection_method', 'autoreject')
    n_interp    = md.get('n_epochs_with_interpolation', 0)
    interp_chs  = md.get('interpolated_channels', {})
    ref         = md.get('reference', '')
    hp          = md.get('highpass_hz')
    lp          = md.get('lowpass_hz')
    if n_rare_pre is not None and n_rare_post is not None:
        n_rare_rej = n_rare_pre  - n_rare_post
        n_std_rej  = (n_std_pre or 0) - (n_std_post or 0)
        tone_str = (f'Tones kept: {n_rare_post} rare ({n_rare_rej} fully rejected'
                    + (f', {n_interp} had channels interpolated' if n_interp else '')
                    + f') / {n_std_post} standard ({n_std_rej} fully rejected).')
        parts.append(tone_str)
        if interp_chs:
            top = sorted(interp_chs.items(), key=lambda x: -x[1])[:4]
            ch_str = ', '.join(f'{c} ({n}x)' for c, n in top)
            parts.append(f'Rejection: autoreject with channel interpolation; '
                          f'most corrected: {ch_str}.')
        else:
            parts.append('Rejection: autoreject with channel interpolation; '
                          'no channels required interpolation.')
    if ref:
        parts.append(f'Reference: {ref}.')
    if hp is not None and lp is not None:
        parts.append(f'Filter: {hp}-{lp} Hz.')
    fischer = md.get('fischer_score')
    n_comp  = md.get('n_components')
    if fischer is not None and n_comp is not None:
        parts.append(f'Fischer hierarchy: {fischer}/{n_comp} components significant.')
    shao_mmn = md.get('shao_mmn_positive')
    shao_p3b = md.get('shao_p3b_positive')
    if shao_mmn is not None and shao_p3b is not None:
        ms = 'Pass' if shao_mmn else 'Fail'
        ps = 'Pass' if shao_p3b else 'Fail'
        cs = 'Pass' if (shao_mmn and shao_p3b) else 'Fail'
        parts.append(f'Shao 2025: MMN {ms}  P3b {ps}  Combined {cs}.')
    pac = md.get('pac_result')
    if pac:
        parts.append(f'Delta-gamma PAC ({pac.get("channel", "Pz")}): '
                     f'diff={pac["obs_diff"]:+.4f}  p={pac["p_value"]:.3f}.')
    lzc = md.get('lzc_result')
    if lzc:
        parts.append(f'Lempel-Ziv complexity: diff={lzc["obs_diff"]:+.4f}  p={lzc["p_value"]:.3f}.')
    return '  '.join(parts)


def title_page(p: FPDF, adef: dict, patient_ids: list, date_str: str) -> None:
    # Page 1 — paradigm overview
    p.add_page()
    y = 0.55
    y += _txt(p, MARGIN, y, 'EEG Clinical Analysis Report', size=18, style='B') + 0.12
    y += _txt(p, MARGIN, y, adef['full_title'], size=13) + 0.08
    y += _txt(p, MARGIN, y, 'Harborview Medical Center  \xb7  University of Washington', size=10) + 0.06
    _txt(p, MARGIN, y, f'Generated: {date_str}', size=9, color=(100, 100, 100))
    y += 0.25
    _rule(p, y);  y += 0.18
    y += _txt(p, MARGIN, y, 'Paradigm Overview', size=11, style='B') + 0.14
    _wrap_txt(p, MARGIN, y, adef['overview'], size=10, line_spacing=1.6)

    # Page 2 — patients + figures index
    p.add_page()
    y = 0.55
    y += _txt(p, MARGIN, y, f'Patients in this report ({len(patient_ids)})',
              size=11, style='B') + 0.10
    y += _txt(p, MARGIN, y, '   '.join(patient_ids), size=10) + 0.22
    y += _txt(p, MARGIN, y, 'Figures included per patient', size=11, style='B') + 0.12
    for fd in adef['figures']:
        y += _txt(p, MARGIN + 0.15, y, f'- {fd["title"]}', size=10) + 0.04


def patient_divider(p: FPDF, patient_id: str, analysis_title: str,
                    metadata: dict = None) -> None:
    p.add_page()
    mid = PAGE_H / 2
    _rule(p, mid - 0.45)
    _txt(p, MARGIN, mid - 0.38, patient_id, size=28, style='B', align='C')
    _txt(p, MARGIN, mid + 0.08, analysis_title, size=14, align='C')
    _rule(p, mid + 0.35)
    if metadata:
        note = _metadata_note(metadata)
        if note:
            _wrap_txt(p, MARGIN + 0.5, mid + 0.48, note,
                      size=9, style='I', max_w=TEXT_W - 1.0)


def figure_page(p: FPDF, img_path: Path, patient_id: str,
                fig_def: dict, citation_nums: list = None) -> None:
    HEADER_H      = 0.55
    IMG_PAD       = 0.20
    BOTTOM_MARGIN = 0.50

    desc      = fig_def['description']
    n_lines   = len(textwrap.fill(desc, width=90).split('\n'))
    caption_h = max(1.0, 0.55 + n_lines * 9 * 1.55 / 72)

    rule_y     = MARGIN / 2 + HEADER_H
    img_top    = rule_y + IMG_PAD
    img_max_h  = PAGE_H - rule_y - caption_h - 2 * IMG_PAD - BOTTOM_MARGIN
    cap_rule_y = PAGE_H - BOTTOM_MARGIN - caption_h

    p.add_page()

    # Header: patient left, figure title right
    lh_hdr = 10 * 1.4 / 72
    _txt(p, MARGIN, MARGIN / 2, f'Patient: {patient_id}', size=10, style='B', align='L')
    _txt(p, MARGIN, MARGIN / 2, fig_def['title'], size=10, align='R')
    _rule(p, rule_y)

    # Image — direct PNG embed
    _place_img(p, img_path, top=img_top, max_h=img_max_h)

    # Caption
    _rule(p, cap_rule_y)
    title_text = fig_def['title']
    if citation_nums:
        title_text += '  ' + ', '.join(f'[{n}]' for n in sorted(citation_nums))
    y = cap_rule_y + 0.06
    y += _txt(p, MARGIN, y, title_text, size=10, style='B') + 0.06
    _wrap_txt(p, MARGIN, y, desc, size=9, line_spacing=1.55)


def references_page(p: FPDF, adef: dict) -> None:
    seen, unique = set(), []
    for fd in adef['figures']:
        for c in _fig_citations(fd):
            if c and c not in seen:
                seen.add(c); unique.append(c)
    if not unique:
        return
    p.add_page()
    y = 0.55
    y += _txt(p, MARGIN, y, 'References', size=13, style='B') + 0.08
    _rule(p, y);  y += 0.14
    for i, citation in enumerate(unique, 1):
        wrapped = textwrap.fill(f'[{i}]  {citation}', width=88,
                                subsequent_indent='     ')
        h = _wrap_txt(p, MARGIN, y, wrapped, size=9, line_spacing=1.55)
        y += h + 0.08


# ── Report builder + main ──────────────────────────────────────────────────────

def _collect_patients(results_dir: Path, analysis_key: str) -> dict:
    patients = {}
    for pd in sorted(results_dir.iterdir()):
        if not pd.is_dir() or pd.name == 'POOLED':
            continue
        ad = pd / analysis_key
        if not ad.is_dir():
            ad = pd   # voice files may live in patient root for older sessions
        pngs = {f.name: f for f in ad.glob('*.png')}
        if pngs:
            patients[pd.name] = pngs
    return patients


def build_report(analysis_key: str, adef: dict, date_str: str) -> None:
    pdf_path = REPORTS_DIR / adef['pdf_name']
    if pdf_path.exists():
        pdf_path.unlink()
        print(f'  Deleted old report: {pdf_path.name}')

    patients = _collect_patients(RESULTS_DIR, analysis_key)
    if not patients:
        print(f'  No results found for "{analysis_key}" — skipping.')
        return
    print(f'  Found {len(patients)} patient(s): {", ".join(patients)}')

    cit_to_num: dict = {}
    for fd in adef['figures']:
        for c in _fig_citations(fd):
            if c and c not in cit_to_num:
                cit_to_num[c] = len(cit_to_num) + 1

    p = _make_pdf()
    title_page(p, adef, list(patients.keys()), date_str)

    for patient_id, pngs in patients.items():
        meta_path = RESULTS_DIR / patient_id / analysis_key / 'metadata.json'
        metadata  = json.loads(meta_path.read_text()) if meta_path.exists() else None
        patient_divider(p, patient_id, adef['title'], metadata=metadata)

        included = 0
        for fd in adef['figures']:
            matches = [v for k, v in pngs.items() if k.endswith(fd['suffix'])]
            if not matches:
                continue
            nums = [cit_to_num[c] for c in _fig_citations(fd) if c in cit_to_num]
            figure_page(p, matches[0], patient_id, fd, citation_nums=nums)
            included += 1

        if included == 0:
            p.add_page()
            _txt(p, MARGIN, PAGE_H / 2, f'No figures found for {patient_id}.',
                 size=14, color=(136, 136, 136), align='C')

    references_page(p, adef)
    p.output(str(pdf_path))
    print(f'  Saved: {pdf_path}')


def build_pooled_report(analysis_key: str, adef: dict, date_str: str) -> None:
    """Build a report for pooled (cross-patient) analyses with no per-patient structure."""
    pdf_path = REPORTS_DIR / adef['pdf_name']
    if pdf_path.exists():
        pdf_path.unlink()
        print(f'  Deleted old report: {pdf_path.name}')

    pooled_dir = RESULTS_DIR / 'POOLED' / analysis_key.replace('pooled_', '')
    if not pooled_dir.is_dir():
        print(f'  No pooled results at {pooled_dir} — skipping.')
        return
    pngs = {f.name: f for f in pooled_dir.glob('*.png')}
    if not pngs:
        print(f'  No PNG figures found in {pooled_dir} — skipping.')
        return

    cit_to_num: dict = {}
    for fd in adef['figures']:
        for c in _fig_citations(fd):
            if c and c not in cit_to_num:
                cit_to_num[c] = len(cit_to_num) + 1

    p = _make_pdf()
    # Title page — no patient list for pooled
    p.add_page()
    y = 0.55
    y += _txt(p, MARGIN, y, 'EEG Clinical Analysis Report', size=18, style='B') + 0.12
    y += _txt(p, MARGIN, y, adef['full_title'], size=13) + 0.08
    y += _txt(p, MARGIN, y, 'Harborview Medical Center  \xb7  University of Washington',
              size=10) + 0.06
    _txt(p, MARGIN, y, f'Generated: {date_str}', size=9, color=(100, 100, 100))
    y += 0.25
    _rule(p, y);  y += 0.18
    y += _txt(p, MARGIN, y, 'Overview', size=11, style='B') + 0.14
    _wrap_txt(p, MARGIN, y, adef['overview'], size=10, line_spacing=1.6)

    for fd in adef['figures']:
        matches = [v for k, v in pngs.items() if k.endswith(fd['suffix'])]
        if not matches:
            continue
        nums = [cit_to_num[c] for c in _fig_citations(fd) if c in cit_to_num]
        figure_page(p, matches[0], 'POOLED', fd, citation_nums=nums)

    references_page(p, adef)
    p.output(str(pdf_path))
    print(f'  Saved: {pdf_path}')


def _build_one(args):
    key, adef, date_str = args
    if key.startswith('pooled_'):
        build_pooled_report(key, adef, date_str)
    else:
        build_report(key, adef, date_str)
    return key


def main() -> None:
    date_str = datetime.datetime.now().strftime('%B %d, %Y')
    print(f'Generating EEG analysis reports — {date_str}')
    print(f'Results directory: {RESULTS_DIR}')
    print(f'Reports directory: {REPORTS_DIR}\n')

    items = list(ANALYSES.items())
    n_workers = min(len(items), 4)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_build_one, (key, adef, date_str)): key
                   for key, adef in items}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                fut.result()
                print(f'[{key.upper()}] done')
            except Exception as e:
                print(f'[{key.upper()}] ERROR: {e}')

    print('\nDone.')


if __name__ == '__main__':
    main()
