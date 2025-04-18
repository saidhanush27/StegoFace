# StegoFace: Deep Learning-Based ID Image Security  

## 📌 Project Description  
StegoFace is a deep learning-based **steganography model** designed to enhance **ID image security** by embedding **hidden authentication data** within facial images. The system ensures **tamper detection** while preserving **image quality and integrity**, making it a reliable solution for **photo substitution attack prevention**.  

## 🔥 Features  
- 🛡️ **Deep CNN-Based Steganography** for secure message embedding.  
- 🔍 **Binary Error-Correcting Codes (BECC)** for robustness against noise and compression.  
- 🔄 **Autoencoder-Decoder Framework** for high-precision encoding and decoding.  
- 🎯 **Recurrent Proposal Network (RPN)** for accurate facial region detection.  
- ⚡ **Real-time verification** for ID security applications.  

## 🛠️ PROJECT OUTPUT
![Alt Text](https://github.com/saidhanush27/StegoFace/blob/47c66502600b9d84f0d64276ad3c58dc26453082/Stegoface-images/5.png) <br><br>
![Alt Text](https://github.com/saidhanush27/StegoFace/blob/5f46b3893a7a1747ec0a92e522d5187a6ea8547c/Stegoface-images/2.png)  <br><br>
![Alt Text](https://github.com/saidhanush27/StegoFace/blob/5f46b3893a7a1747ec0a92e522d5187a6ea8547c/Stegoface-images/3.png)  <br><br>
![Alt Text](https://github.com/saidhanush27/StegoFace/blob/5f46b3893a7a1747ec0a92e522d5187a6ea8547c/Stegoface-images/4.png)   <br><br>


  
## 🛠️ Technologies Used  
- 🐍 **Python**  
- 🤖 **TensorFlow/Keras** (Deep Learning)  
- 🖼️ **OpenCV** (Image Processing)  
- 📊 **NumPy & Pandas** (Data Handling)  
- 📈 **Matplotlib & Seaborn** (Visualization)  

## 📂 Project Structure  
📦 StegoFace <br>
┣ 📂 dataset  <br>
┣ 📂 models  <br>
┣ 📂 preprocessing <br>
┣ 📂 encoder_decoder  <br>
┣ 📂 utilss <br>
┣ 📜 requirements.txt <br>
┣ 📜 README.md  <br>
┗ 📜 main.py  <br>


## 🚀 Installation & Usage  

**1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/StegoFace.git
cd StegoFace
```


2. Install Dependencies
Ensure you have all necessary libraries installed by running:

```bash

pip install -r requirements.txt
```

3. Run the Program
Execute the main script to start the system:

```bash
python main.py
```
## 📌 How It Works
Preprocessing Module: Enhances image quality and extracts facial features.

Encoding Phase: The autoencoder securely embeds authentication data into ID images.

Decoding Phase: The auto decoder retrieves hidden messages for verification.

Verification: If the extracted message is intact, the ID is considered valid; otherwise, tampering is detected.

## 👨‍💻 Contributors
Sai Dhanush V.R

M. Mukunda

Surya J

## 📜 License
This project is licensed under the MIT License.

## 🏷️ Tags
#DeepLearning #Steganography #IDSecurity #ImageProcessing #NeuralNetworks
#Autoencoder #BinaryErrorCorrection #FaceRecognition #DocumentSecurity #Python
#MachineLearning #AI #ComputerVision #SecureAuthentication #DataEmbedding
