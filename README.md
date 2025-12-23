# Image Recognition System for Automated Checkout
> **Machine Learning Final Project** | **Team 22**

An automated checkout system for convenience stores leveraging computer vision techniques.

## 📦 Installation

### 1. Clone the Repository

Download the project files **best.pt** **smart_checkout.py** to your local machine

### 2. Install Dependencies

Open your terminal or command prompt in the project directory and install the required packages:

```bash
pip install opencv-python ultralytics
```


## 🚀 Usage
1. Prepare Camera Source
Launch the DroidCam application on your mobile device to use it as a wireless webcam.

2. Network Configuration
Choose one of the following connection methods:

**Option A**: Mobile Hotspot (Recommended) Connect your computer to your mobile device's Personal Hotspot.

Note: The default IP address in smart_checkout.py is set to 172.20.10.1 (iPhone default). **If you are using an iPhone hotspot, no code changes are need**.

**Option** B: Local Wi-Fi (LAN) If both devices are connected to the same Wi-Fi network:
- Find your phone's IP address displayed in the DroidCam app.

- Open smart_checkout.py.

- Manually update the IP configuration variable to match your phone's IP.

3. Execute the Program
Run the application using the following command:
```
python smart_checkout.py
```

Note: You can now use your cellphone's camera to identify and display the prices of all chips captured on the screen in real-time.

![image](https://github.com/shoumft/ML_finalProject/blob/main/resultExample1.jpg)
![image](https://github.com/shoumft/ML_finalProject/blob/main/resultExample2.jpg)
