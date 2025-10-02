# 🫀 ECG Lead Misplacement Detection System

**Advanced AI-powered detection of electrocardiographic lead misplacements using machine learning and Gradio web interface.**

---

## 🚀 Overview

This application detects common ECG lead misplacements using AI models, based on clinical research from *Europace* journal ("Incorrect electrode cable connection during electrocardiographic recording" by Batchvarov et al.).  
It supports image (JPG, PNG), PDF, and DICOM ECG files.  
The app features educational resources and clinical guidelines for healthcare professionals.

---

## ✨ Features

- **Multi-format ECG File Analysis:** JPG, PNG, PDF, DICOM
- **AI Models:** Random Forest, Gradient Boosting, Neural Network
- **Detection Categories:**  
  - RA/LA Reversal (most common)  
  - RA/LL Reversal  
  - LA/LL Reversal  
  - Neutral electrode misplacements  
  - Precordial misplacements  
  - Multiple misplacements
- **Probability Scores & Visual Charts** for each category
- **Clinical Recommendations** based on detected patterns
- **Sample ECG Patterns** for reference and training
- **Clinical Guidelines Tab** summarizing detection criteria

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shivesh2334-ai/ECG-lead-placement-.git
   cd ECG-lead-placement-
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   python ecg_lead_misplacement_app.py
   ```
   The app will open at [http://localhost:7860](http://localhost:7860) (or public URL if `share=True`).

---

## 📂 File Structure

- `ecg_lead_misplacement_app.py` — Main Gradio application and ML pipeline
- `requirements.txt` — Required Python packages
- `README.md` — This documentation
- `LICENSE` — Project license

---

## 🩺 Clinical Usage

- **For educational and research use.**
- Results should always be confirmed by qualified clinicians.
- Re-record ECG if misplacement is suspected.

---

## 📜 License

See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Based on research by Batchvarov et al., *Europace* journal.
- Built with [Gradio](https://gradio.app/) and [scikit-learn](https://scikit-learn.org/).

---

## 💡 Disclaimer

This tool does **not** provide medical advice. Use for research/education only.
