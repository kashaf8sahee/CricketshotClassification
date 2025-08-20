# 🏏 Cricket Shot Classification  

This project uses **PyTorch** to classify cricket shots into four categories:  
- **Drive**  
- **Leg Glance**  
- **Pull Shot**  
- **Sweep**  

The model is built on **ResNet50 transfer learning** with data augmentation, Adam optimizer, and learning rate scheduling.  
Performance is evaluated using **accuracy, precision, recall, F1-score, confusion matrix, and ROC curves**, along with visualization of sample predictions.  

---

## 📂 Dataset  
- Total images: **4724**  
- Classes: **4**  
- Drive (1224), Leg Glance (1120), Pull Shot (1260), Sweep (1120)  
- Split into training, validation, and test sets  

---

## ⚙️ Tech Stack  
- **PyTorch** / Torchvision  
- **Scikit-learn**  
- **NumPy, Matplotlib, Seaborn**  

---

## 📌 Features  
- Transfer learning with **ResNet50**  
- **Data augmentation** for better generalization  
- **Learning rate scheduling**  
- Detailed evaluation: accuracy, precision, recall, F1-score, confusion matrix, ROC curves  
- Visualization of **sample predictions**  

---

## 🚀 Installation  


# Clone the repository
git clone https://github.com/yourusername/CricketShotClassification.git
cd CricketShotClassification

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

## ▶️ Usage

# Train the model
python train.py

# Evaluate the model
python evaluate.py


# Install dependencies
pip install -r requirements.txt
