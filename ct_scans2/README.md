CT Scan Y3 Progressive Analyzer
Automatic detection of the third lumbar vertebra (L3) from DICOM CT scan image series.

📂 Project Structure
ct_scans2/
├── data/               # Place your DICOM (.dcm) files here
├── debug_output/       # Debug images (auto-generated)
├── output/             # Output DICOM (detected L3)
├── output_web/         # Output for web app
├── src/                # Main source code
│   ├── detector.py     # Vertebra detection logic
│   ├── loader.py       # DICOM loader
│   ├── main.py         # CLI entry-point
│   ├── segmenter.py    # Binarization and preprocessing
│   └── skeleton.py     # Skeletonization and Y-points detection
├── static/             # Frontend (CSS, JS)
│   ├── css/
│   ├── js/
│   └── templates/
├── web_app.py          # Web application (Flask)
├── requirements.txt    # Python dependencies
└── README.md           # This file

🚀 How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Data Folder
Place your .dcm files inside the data/ directory.

3️⃣ Run CLI
python src/main.py
Output DICOM will be saved in the output/ folder.

4️⃣ Run Web App
python web_app.py
Access in your browser:
http://localhost:5000
🛠️ Technologies Used
Python 3.10+
Flask + Socket.IO
scikit-image
pydicom
matplotlib
scipy

📊 Project Features
Load CT scan DICOM images.

Apply medical windowing and adaptive binarization.

Detect vertebra shape and characteristic "Y" points.

Automatically save detected third lumbar vertebra (L3).

⚠️ Disclaimer
This project is for educational and research purposes only. It is not medically validated for clinical use.