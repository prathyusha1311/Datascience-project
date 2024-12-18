# Health Guard: A Complete Health Application

## Team Information

**Team Name:** Squad 11  
**Team Members:**
- Arun Prakash Jayakanthan
- Prathyusha Naresh Kumar
- Aniket Modi
- Chandini Satish Kumar

## Project Overview

### Project Name:
Health Guard: A Complete Health Application

### Abstract:
Health Guard is a health-focused application designed to detect falls in real-time and provide immediate assistance by notifying emergency contacts, contacting nearby hospitals, and sending telemetry data. The app utilizes advanced machine learning techniques and mobile sensors to ensure user safety. Additionally, it incorporates federated machine learning to enhance data privacy and model accuracy while updating the global model across multiple devices. 

Currently, the fall detection feature is implemented, and further features like hospital notifications and health suite integration are under development.

---

## Key Features

### Fall Detection
- **Real-time Monitoring:** Captures data from sensors such as accelerometers, gyroscopes, and rotation vectors.
- **Machine Learning Model:** Employs a logistic regression model for fall detection.
- **Proactive Alerts:** Notifies emergency services and nearby hospitals immediately upon detecting a fall.
- **Privacy:** Leverages federated learning to maintain user privacy.

### Federated Machine Learning
- **Decentralized Training:** Local models on client devices improve the global model without sharing raw data.
- **Performance Metrics:** Assesses performance using accuracy and loss after federated updates.
- **Model Sharing:** Updates and distributes weights across clients for continuous improvement.

---

## Tech Stack

1. **Frontend:** React Native
2. **Backend:** Flask
3. **Testing:** React Expo Go App
4. **Database:** Python SQL Alchemy for user data storage

---

## How to Run the Application

### Fall Detection Application:
1. Navigate to the `Fall Detection App` folder.
2. Install the Expo Go app on your mobile device.
3. Run the following command to start the frontend:
   ```bash
   npx expo start
   ```
   - Scan the QR code generated using the Expo Go app to open the application.
   - Register with your email ID and password to access the app.
4. Open a new terminal and navigate to the backend folder.
5. Start the backend by running:
   ```bash
   python app.py
   ```
   - Update the IP address in the frontend React Native code to match your PC’s IP.

### Federated Machine Learning and Prototype Testing:
1. Set up a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scriptsctivate`
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the federated learning notebook or test files.

---

## Files and Descriptions

### 1. **fall_detection_algorithm.py**
- Implements real-time fall detection.
- **Features:**
  - Sensor data collection via WebSocket.
  - Feature extraction for logistic regression model.
  - Alerts and notifications to hospitals and emergency contacts.

### 2. **Federated_machine_learning.ipynb**
- Simulates a client-server federated learning system.
- Includes steps for model initialization, weight sharing, and global model updates.

### 3. **Fall Detection App Folder**
- Complete interface for real-time fall detection.
- Combines React Native, machine learning, and backend services for an end-to-end application.

### Images

<!-- **System Architecture:**
![System Architecture](images/system_architecture.png) -->

**Fall Detection Flow:**
![Fall Detection Flow]("Images\Fall detection flow.png")

**Federated Learning Process:**
![Federated Learning Process]("https://www.google.com/search?sca_esv=0a0afea8e69e4a8b&q=federated+learning+flowchart&udm=2&fbs=AEQNm0Aa4sjWe7Rqy32pFwRj0UkWd8nbOJfsBGGB5IQQO6L3J7pRxUp2pI1mXV9fBsfh39Jw_Y7pXPv6W9UjIXzt09-Y-RVsUQytO3H9U9unQ4zjSmyc1am7RU9IOaZeZLN-vxqOLRVgtOkNIBInceOOInHD1Vy8A8dMZkK6qsEDDgBo37uamqwPID1ktpoxri6hURFY-RftoYl5J3cAxl4SOYvmGkrX6Q&sa=X&ved=2ahUKEwiq8PeQ-bGKAxVn4zgGHRLFNvYQtKgLegQIFBAB&biw=1470&bih=832&dpr=2#vhid=QapUOPtmBX_qiM&vssid=mosaic")

---

## Future Enhancements

1. Integrate notifications for nearby hospitals.
2. Expand health suite functionalities for holistic health management.
3. Refine federated learning mechanisms for better accuracy and privacy.

---

## Contact
For any queries, feel free to reach out to the team at [nareshku@usc.edu](mailto:nareshku@usc.edu).
