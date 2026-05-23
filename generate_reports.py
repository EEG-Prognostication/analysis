#!/usr/bin/env python3
"""
generate_reports.py

Generates one PDF clinical report per EEG analysis paradigm, aggregating results
from every patient found in the results directory. Deletes any existing report PDF
before writing the new one.

Usage (from the analysis/ directory, with the venv active):
    python generate_reports.py
"""

import json
import textwrap
import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

# ── Paths ──────────────────────────────────────────────────────────────────────
ANALYSIS_ROOT = Path(__file__).parent.resolve()
RESULTS_DIR   = ANALYSIS_ROOT / 'results'
REPORTS_DIR   = ANALYSIS_ROOT / 'reports'
REPORTS_DIR.mkdir(exist_ok=True)

PAGE_W, PAGE_H = 8.5, 11.0   # letter, inches
MARGIN = 0.75                 # left/right margin, inches
GREY = '#888888'

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
            'Signal processing: the raw EEG was bandpass filtered (0.1-30 Hz), re-referenced '
            'to the scalp average across all electrodes, and tones where any electrode exceeded '
            '200 µV (indicating movement or electrical artifact) were excluded before averaging.'
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
                'suffix': '_command_svm_null.png',
                'title':  'Can a Computer Tell "Keep" from "Stop" Using the Brain Signal?',
                'description': (
                    'We trained a machine-learning algorithm to read the EEG signal and decide '
                    'whether the patient was in the "keep squeezing" or "stop" phase. Its accuracy '
                    'is expressed as an AUC score: 0.5 means pure chance (the algorithm cannot '
                    'tell the conditions apart), while 1.0 means perfect classification. To '
                    'confirm the result is real and not a statistical fluke, we repeated the test '
                    '200 times with randomly shuffled labels. The blue histogram shows the range '
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


# ── Rendering helpers ──────────────────────────────────────────────────────────
def _wrap(text, width=90):
    return textwrap.fill(text, width=width)


def _place_image(fig, img_path, left_in, bottom_in, max_w_in, max_h_in):
    """Add an image axes that preserves the image's aspect ratio."""
    img = mpimg.imread(str(img_path))
    h_px, w_px = img.shape[:2]
    img_aspect = w_px / h_px  # > 1 means wider than tall

    # Scale to fit within the available box, preserving aspect ratio
    if max_w_in / max_h_in >= img_aspect:
        # height is the limiting dimension
        draw_h = max_h_in
        draw_w = draw_h * img_aspect
    else:
        draw_w = max_w_in
        draw_h = draw_w / img_aspect

    # Centre within the available box
    left   = (left_in + (max_w_in - draw_w) / 2) / PAGE_W
    bottom = (bottom_in + (max_h_in - draw_h) / 2) / PAGE_H
    width  = draw_w / PAGE_W
    height = draw_h / PAGE_H

    ax = fig.add_axes([left, bottom, width, height])
    ax.imshow(img)
    ax.axis('off')


def title_page(pdf, analysis_def, patient_ids, date_str):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    x = MARGIN / PAGE_W          # left edge in figure fraction
    y = 0.95                      # start near top

    ax.text(x, y, 'EEG Clinical Analysis Report',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=18, fontweight='bold')
    ax.text(x, y - 0.055, analysis_def['full_title'],
            transform=ax.transAxes, ha='left', va='top', fontsize=13)
    ax.text(x, y - 0.09, 'Harborview Medical Center  ·  University of Washington',
            transform=ax.transAxes, ha='left', va='top', fontsize=10)
    ax.text(x, y - 0.115, f'Generated: {date_str}',
            transform=ax.transAxes, ha='left', va='top', fontsize=9, color='#555555')

    ax.plot([x, 1 - x], [y - 0.135, y - 0.135],
            color='black', linewidth=0.8, transform=ax.transAxes, clip_on=False)

    ax.text(x, y - 0.165, 'Paradigm Overview',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=11, fontweight='bold')

    overview_top = y - 0.205
    wrapped_overview = _wrap(analysis_def['overview'])
    ax.text(x, overview_top, wrapped_overview,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=10, linespacing=1.6)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # Page 2 — patients and figures list
    fig2 = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax2  = fig2.add_axes([0, 0, 1, 1])
    ax2.axis('off')

    y2 = 0.92
    ax2.text(x, y2, f'Patients in this report ({len(patient_ids)})',
             transform=ax2.transAxes, ha='left', va='top',
             fontsize=11, fontweight='bold')
    ax2.text(x, y2 - 0.04, '  '.join(patient_ids),
             transform=ax2.transAxes, ha='left', va='top',
             fontsize=10, family='monospace')

    figures_y = y2 - 0.10
    ax2.text(x, figures_y, 'Figures included per patient',
             transform=ax2.transAxes, ha='left', va='top',
             fontsize=11, fontweight='bold')
    for i, fig_def in enumerate(analysis_def['figures']):
        ax2.text(x + 0.02, figures_y - 0.045 - i * 0.042, f'- {fig_def["title"]}',
                 transform=ax2.transAxes, ha='left', va='top', fontsize=10)

    pdf.savefig(fig2, bbox_inches='tight')
    plt.close(fig2)


def patient_divider(pdf, patient_id, analysis_title, metadata=None):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    ax.text(0.5, 0.56, patient_id,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=28, fontweight='bold')
    ax.text(0.5, 0.50, analysis_title,
            transform=ax.transAxes, ha='center', va='center', fontsize=14)
    ax.plot([0.1, 0.9], [0.54, 0.54],
            color='black', linewidth=0.8, transform=ax.transAxes, clip_on=False)

    if metadata:
        note = _build_metadata_note(metadata)
        if note:
            ax.text(0.5, 0.44, _wrap(note, width=80),
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=9, style='italic', color='black', linespacing=1.5)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)




def _build_metadata_note(metadata):
    """Return a one-line analysis note from a metadata dict, or empty string."""
    parts = []
    n_rare_pre  = metadata.get('n_rare_pre_rejection')
    n_rare_post = metadata.get('n_rare_post_rejection')
    n_std_pre   = metadata.get('n_std_pre_rejection')
    n_std_post  = metadata.get('n_std_post_rejection')
    thresh      = metadata.get('rejection_threshold_uv')
    ref         = metadata.get('reference', '')
    hp          = metadata.get('highpass_hz')
    lp          = metadata.get('lowpass_hz')

    if n_rare_pre is not None and n_rare_post is not None:
        n_rare_rej = n_rare_pre - n_rare_post
        n_std_rej  = (n_std_pre or 0) - (n_std_post or 0)
        thresh_str = f'>{thresh} µV' if thresh else 'threshold'
        parts.append(
            f'Tones: {n_rare_post} rare / {n_std_post} standard '
            f'({n_rare_rej} rare and {n_std_rej} standard excluded, {thresh_str}).'
        )
    if ref:
        parts.append(f'Reference: {ref}.')
    if hp is not None and lp is not None:
        parts.append(f'Filter: {hp}–{lp} Hz.')

    fischer    = metadata.get('fischer_score')
    n_comp     = metadata.get('n_components')
    shao_mmn   = metadata.get('shao_mmn_positive')
    shao_p3b   = metadata.get('shao_p3b_positive')
    if fischer is not None and n_comp is not None:
        parts.append(f'Fischer hierarchy: {fischer}/{n_comp} components significant.')
    if shao_mmn is not None and shao_p3b is not None:
        mmn_sym = '✓' if shao_mmn else '✗'
        p3b_sym = '✓' if shao_p3b else '✗'
        combined = '✓' if (shao_mmn and shao_p3b) else '✗'
        parts.append(f'Shao 2025 thresholds: MMN {mmn_sym}  P3b {p3b_sym}  Combined {combined}.')

    return '  '.join(parts)


def figure_page(pdf, img_path, patient_id, fig_def, citation_nums=None):
    # Layout (bottom → top):
    #   BOTTOM_MARGIN | caption text | rule | IMG_PAD | image | IMG_PAD | header
    # CAPTION_H is sized to the actual text so blank space ends up between the
    # image and the caption rather than below the text.
    HEADER_H     = 0.55   # inches — patient/title strip at top
    IMG_PAD      = 0.20   # inches — vertical gap between image and caption rule
    BOTTOM_MARGIN = 0.50  # inches — whitespace below description text

    full_description = fig_def['description']
    n_desc_lines = len(_wrap(full_description).split('\n'))
    # fontsize 9, linespacing 1.55 → ~0.194 in per line; +0.55 for rule + bold title line
    CAPTION_H = max(1.0, 0.55 + n_desc_lines * 9 * 1.55 / 72)

    img_max_w   = PAGE_W - 2 * MARGIN
    img_max_h   = PAGE_H - HEADER_H - CAPTION_H - 2 * IMG_PAD - BOTTOM_MARGIN
    img_bottom_in = BOTTOM_MARGIN + CAPTION_H + IMG_PAD

    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # ── header ──
    header_top = (PAGE_H - MARGIN / 2) / PAGE_H
    ax.text(MARGIN / PAGE_W, header_top,
            f'Patient: {patient_id}',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=10, fontweight='bold')
    ax.text(1 - MARGIN / PAGE_W, header_top,
            fig_def['title'],
            transform=ax.transAxes, ha='right', va='top', fontsize=10)
    rule_y = (PAGE_H - HEADER_H) / PAGE_H
    ax.plot([MARGIN / PAGE_W, 1 - MARGIN / PAGE_W], [rule_y, rule_y],
            color='black', linewidth=0.6, transform=ax.transAxes, clip_on=False)

    # ── image (aspect-correct) ──
    _place_image(fig, img_path,
                 left_in=MARGIN, bottom_in=img_bottom_in,
                 max_w_in=img_max_w, max_h_in=img_max_h)

    # ── caption ──
    cap_top = (BOTTOM_MARGIN + CAPTION_H - 0.15) / PAGE_H
    ax.plot([MARGIN / PAGE_W, 1 - MARGIN / PAGE_W], [cap_top, cap_top],
            color='black', linewidth=0.6, transform=ax.transAxes, clip_on=False)

    cx = MARGIN / PAGE_W
    title_text = fig_def['title']
    if citation_nums:
        title_text += '  ' + ','.join(f'[{n}]' for n in sorted(citation_nums))
    ax.text(cx, cap_top - 0.005,
            title_text,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=10, fontweight='bold')
    ax.text(cx, cap_top - 0.045,
            _wrap(full_description),
            transform=ax.transAxes, ha='left', va='top',
            fontsize=9, linespacing=1.55)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _fig_citations(fig_def):
    """Return the list of citation strings for a figure, supporting both 'citations' (list) and legacy 'citation' (str)."""
    if 'citations' in fig_def:
        return [c.strip() for c in fig_def['citations'] if c.strip()]
    legacy = fig_def.get('citation', '').strip()
    return [legacy] if legacy else []


def references_page(pdf, analysis_def):
    """Render a single References page listing all unique citations from the figures."""
    # Deduplicate while preserving order
    seen = set()
    unique_citations = []
    for fig_def in analysis_def['figures']:
        for c in _fig_citations(fig_def):
            if c not in seen:
                seen.add(c)
                unique_citations.append(c)

    if not unique_citations:
        return

    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    cx = MARGIN / PAGE_W
    y = 0.93

    ax.text(cx, y, 'References',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=13, fontweight='bold')
    ax.plot([cx, 1 - cx], [y - 0.03, y - 0.03],
            color='black', linewidth=0.6, transform=ax.transAxes, clip_on=False)

    y -= 0.06
    text_w = PAGE_W - 2 * MARGIN        # inches
    chars_per_line = int(text_w / (9 / 72) * 1.6)   # rough estimate at fontsize 9

    for i, citation in enumerate(unique_citations, start=1):
        wrapped = textwrap.fill(f'[{i}]  {citation}', width=chars_per_line,
                                subsequent_indent='     ')
        n_lines = len(wrapped.split('\n'))
        line_h = 1.55 * 9 / (72 * PAGE_H)
        ax.text(cx, y, wrapped,
                transform=ax.transAxes, ha='left', va='top',
                fontsize=9, linespacing=1.55)
        y -= n_lines * line_h + 0.018

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def collect_patients(results_dir, analysis_key):
    """Return sorted list of (patient_id, {suffix: Path}) for a given analysis subdir."""
    patients = {}
    for patient_dir in sorted(results_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        analysis_dir = patient_dir / analysis_key
        if not analysis_dir.is_dir():
            # voice files may be in the patient root for older sessions
            analysis_dir = patient_dir
        pngs = {p.name: p for p in analysis_dir.glob('*.png')}
        if pngs:
            patients[patient_id] = pngs
    return patients


def build_report(analysis_key, analysis_def, results_dir, reports_dir, date_str):
    pdf_path = reports_dir / analysis_def['pdf_name']
    # Delete old report
    if pdf_path.exists():
        pdf_path.unlink()
        print(f'  Deleted old report: {pdf_path.name}')

    patients = collect_patients(results_dir, analysis_key)
    if not patients:
        print(f'  No results found for "{analysis_key}" — skipping.')
        return

    print(f'  Found {len(patients)} patient(s): {", ".join(patients)}')

    # Build citation number map (preserves insertion order, deduplicates)
    citation_to_num = {}
    for fig_def in analysis_def['figures']:
        for c in _fig_citations(fig_def):
            if c and c not in citation_to_num:
                citation_to_num[c] = len(citation_to_num) + 1

    with PdfPages(str(pdf_path)) as pdf:
        # Cover page
        title_page(pdf, analysis_def, list(patients.keys()), date_str)

        for patient_id, pngs in patients.items():
            # Load per-patient metadata sidecar if present
            meta_path = results_dir / patient_id / analysis_key / 'metadata.json'
            patient_metadata = None
            if meta_path.exists():
                with open(meta_path) as f:
                    patient_metadata = json.load(f)

            # Patient divider — includes metadata summary if available
            patient_divider(pdf, patient_id, analysis_def['title'], metadata=patient_metadata)

            included = 0
            for fig_def in analysis_def['figures']:
                matches = [p for name, p in pngs.items() if name.endswith(fig_def['suffix'])]
                if not matches:
                    continue
                citation_nums = [citation_to_num[c] for c in _fig_citations(fig_def)
                                 if c in citation_to_num]
                figure_page(pdf, matches[0], patient_id, fig_def,
                            citation_nums=citation_nums)
                included += 1

            if included == 0:
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis('off')
                ax.text(0.5, 0.55, f'No figures found for {patient_id}',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=14, color=GREY, style='italic')
                ax.text(0.5, 0.48,
                        f'Run notebooks/{analysis_def["title"].lower().replace(" ", "_")}.ipynb '
                        f'to generate results.',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=10, color=GREY, style='italic')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        # References page (all citations consolidated)
        references_page(pdf, analysis_def)

        # PDF metadata
        d = pdf.infodict()
        d['Title'] = analysis_def['full_title']
        d['Author'] = 'Harborview Medical Center EEG Analysis'
        d['Subject'] = 'Clinical EEG Report'
        d['CreationDate'] = datetime.datetime.now()

    print(f'  Saved: {pdf_path}')


def main():
    date_str = datetime.datetime.now().strftime('%B %d, %Y')
    print(f'Generating EEG analysis reports — {date_str}')
    print(f'Results directory: {RESULTS_DIR}')
    print(f'Reports directory: {REPORTS_DIR}\n')

    for key, definition in ANALYSES.items():
        print(f'[{key.upper()}] {definition["full_title"]}')
        build_report(key, definition, RESULTS_DIR, REPORTS_DIR, date_str)
        print()

    print('Done.')


if __name__ == '__main__':
    main()
