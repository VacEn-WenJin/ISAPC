# Literature Facts and Verbatim Citations

This document lists key methodological facts used in the ISAPC + AIP workflow and provides a supporting verbatim sentence from authoritative papers. Each item is formatted as: Fact → Paper → Quoted sentence (verbatim).

---

1) Fact: We use S/N-aware Voronoi binning to achieve approximately constant S/N per spatial bin in IFU data.
- Paper: Cappellari & Copin (2003), “Adaptive spatial binning of integral-field spectroscopic data using Voronoi tessellations,” MNRAS, 342, 345. ADS: https://ui.adsabs.harvard.edu/abs/2003MNRAS.342..345C/abstract
- Verbatim: “We present new techniques to perform adaptive spatial binning of Integral-Field Spectroscopic (IFS) data to reach a chosen constant signal-to-noise ratio per bin.”

2) Fact: Stellar and gas kinematics are recovered via penalized pixel fitting (pPXF) working in pixel space, enabling masking and polynomials.
- Paper: Cappellari & Emsellem (2004), “Parametric Recovery of Line-of-Sight Velocity Distributions from Absorption-Line Spectra of Galaxies via Penalized Likelihood,” PASP, 116, 138. ADS: https://ui.adsabs.harvard.edu/abs/2004PASP..116..138C/abstract
- Verbatim: “We investigate the accuracy of the parametric recovery of the line-of-sight velocity distribution (LOSVD) of the stars in a galaxy while working in pixel space. … We propose a simple solution based on maximum penalized likelihood, and we apply it to the common situation in which the LOSVD is described by a Gauss-Hermite series.”

3) Fact: The upgraded pPXF implementation accurately handles kinematic extraction even when σ is smaller than the velocity sampling, by using an analytic Fourier transform of the Gauss–Hermite kernel.
- Paper: Cappellari (2017), “Improving the full spectrum fitting method: accurate convolution with Gauss-Hermite functions,” MNRAS, 466, 798. ADS: https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..798C/abstract
- Verbatim: “It avoids the evaluation of the undersampled kernel and instead directly computes its well-sampled analytic Fourier transform, for use with the convolution theorem. … The key advantage of the new approach is that it provides accurate velocities regardless of σ.”

4) Fact: We infer [α/Fe], age, and total metallicity using TMB03 simple stellar population models of Lick indices with variable element abundance ratios.
- Paper: Thomas, Maraston & Bender (2003), “Stellar population models of Lick indices with variable element abundance ratios,” MNRAS, 339, 897. ADS: https://ui.adsabs.harvard.edu/abs/2003MNRAS.339..897T/abstract
- Verbatim: “We provide the whole set of Lick indices … of simple stellar population models with, for the first time, variable element abundance ratios, [α/Fe]= 0.0, 0.3, 0.5. … The models cover ages between 1 and 15 Gyr, metallicities between 1/200 and 3.5 solar.”

5) Fact: The composite index [MgFe]′ (and closely related definitions) is nearly independent of [α/Fe] and traces total metallicity in TMB03.
- Paper: Thomas, Maraston & Bender (2003), same as above. ADS: https://ui.adsabs.harvard.edu/abs/2003MNRAS.339..897T/abstract
- Verbatim: “From our α/Fe-enhanced models we infer that the index [MgFe] defined by González is quite independent of α/Fe … We find that the index [MgFe]′, instead, is completely independent of α/Fe and serves best as a tracer of total metallicity.”

6) Fact: MUSE instrument characteristics used for context: 1×1 arcmin² field sampled at 0.2×0.2 arcsec²; 24 IFUs each with an image slicer, spectrograph, and 4k×4k detector.
- Paper: Bacon et al. (2010), “The MUSE second-generation VLT instrument,” SPIE 7735. ADS: https://ui.adsabs.harvard.edu/abs/2010SPIE.7735E..08B/abstract
- Verbatim: “MUSE has a field of 1×1 arcmin2 sampled at 0.2×0.2 arcsec2 … The instrument is a large assembly of 24 identical high performance integral field units, each one composed of an advanced image slicer, a spectrograph and a 4k×4k detector.”

7) Fact: Our radial [α/Fe] gradients are modeled as linear trends of [α/Fe] versus R/Rₑ with uncertainty on the slope from weighted least squares.
- Paper: General methodological choice; no single canonical citation. Included here for transparency.
- Verbatim: N/A (analysis choice; linear fits with uncertainties are standard; if desired we can cite analogous practices in the literature.)

Notes:
- Quotations are taken verbatim from the abstracts where possible to ensure concise and authoritative statements. For precise definitions (e.g., [MgFe]′ formula), consult the full text of TMB03.
- If our dataset uses instruments other than MUSE, replace the instrument sentence with the corresponding instrument paper.
