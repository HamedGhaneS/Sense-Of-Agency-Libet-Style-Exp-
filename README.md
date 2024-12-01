# Sense of Agency - Libet Style Experiment

<div align="center">
  <img src="thesis/figures/SoA.png" alt="Experimental Setup" width="600"/>
</div>

This repository contains the code and documentation for investigating the relationship between the "point of no return" in action initiation and the human sense of agency using real-time Brain-Computer Interfaces (BCI).

## Project Overview

This research explores how the timing of the "point of no return" (approximately 200ms before a voluntary action) influences our post-action sense of agency. Using BCI and machine learning techniques, we detect imminent actions in real-time to study how the temporal relationship between intention detection, action execution, and outcome presentation affects the perceived sense of control over actions.

## Research Hypothesis

We hypothesize that once the point of no return is crossed (~200ms before action), the brain considers the action as "initiated" in terms of agency attribution, even before physical execution. This temporal marker may significantly shape our post-action sense of agency.

## Repository Structure

```
├── data_collection/       # Scripts for the preparatory stage data collection
├── classifier/            # Implementation of the RLDA classifier
├── feature_extraction/    # EEG feature extraction and processing
├── real_time_experiment/ # Main experimental protocol implementation
├── utils/                # Helper functions and utilities
└── thesis/               # Master's thesis documentation
    ├── thesis.pdf       # Full thesis document
    └── figures/         # Original thesis figures
```

## Documentation

The complete research methodology, theoretical background, and experimental results are detailed in the master's thesis document located in the `thesis/` directory. The thesis provides comprehensive information about:

- Theoretical foundations of sense of agency
- Previous research on the point of no return
- Detailed experimental design and methodology
- Implementation of the BCI system
- Results from the preparatory stage
- Future research directions

## Technical Setup

### Hardware Requirements
- ENOBIO 20 5G EEG system
- Medical-grade touch-proof adapter
- Custom button-box with speaker for action-outcome tasks

### Software Dependencies
- MATLAB R2019b or later
- PsychToolbox
- Neuroelectrics Instrument Controller (NIC)

### EEG Configuration
- 17 passive electrodes (F3, Fz, F4, FC5, FC1, FC2, FC6, T7, C3, Cz, C4, T8, CP5, CP1, CP2, CP6, Oz)
- Reference: AFz electrode
- Ground: Right mastoid
- Sampling rate: 500 Hz

## Getting Started

1. Install required dependencies
2. Configure the EEG system according to the specified montage
3. Run the preparatory stage experiments for classifier training
4. Execute the main real-time experiment

## Current Status

The project has completed the preparatory stage, which includes:
- Data collection protocol implementation
- Feature extraction pipeline
- Classifier training framework
- Initial validation with 5 participants

The real-time experimental phase is currently being developed.

## Citation

If you use this code or build upon this research, please cite:
```
Ghane, H. (2023). The Point of No Return in Action Cancellation: 
Deciphering Its Influence on the Human Sense of Agency via Real-Time 
Brain-Computer Interfaces. Master's thesis, Pompeu Fabra University.
```

## Contact

For questions or collaboration inquiries, please get in touch with hmd.ghane.s@gmail.com
