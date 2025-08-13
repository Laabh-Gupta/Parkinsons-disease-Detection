# Parkinson's Disease Detection using Machine Learning

This project utilizes an XGBoost Classifier to predict the presence of Parkinson's disease based on biomedical voice measurements. The model is trained on the UCI Parkinson's Disease Dataset and achieves high accuracy in distinguishing between healthy individuals and those with the condition.

## 📖 Table of Contents
* [About The Project](#about-the-project)
* [Dataset](#dataset)
* [Model & Results](#model--results)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Usage](#usage)
* [License](#license)
* [Contact](#contact)

---

## ℹ️ About The Project

The goal of this project is to build an effective machine learning model for a non-invasive diagnosis of Parkinson's disease. The project covers the entire machine learning pipeline, from data exploration and feature scaling to model training and evaluation using an Extreme Gradient Boosting (XGBoost) algorithm.

The core of this project can be found in `Parkinsons_disease_detection.ipynb`.

---

## 📊 Dataset

This project utilizes the **UCI Parkinson's Disease Dataset**, a popular collection of biomedical voice measurements used for classification tasks.

* **Concept**: Parkinson's disease can affect the muscles in the voice box, leading to changes in a person's speech. These changes, such as tremors and softness, can be quantified into specific measurements. A machine learning model can analyze these measurements to detect patterns that are indicative of the disease.
* **Content**: The dataset consists of 195 voice recordings from 31 individuals (23 with Parkinson's). Each recording includes 22 distinct voice features, such as average vocal fundamental frequency, jitter (frequency variation), and shimmer (amplitude variation).
* **Link**: [https://archive.ics.uci.edu/ml/datasets/parkinsons](https://archive.ics.uci.edu/ml/datasets/parkinsons)

---

## 🤖 Model & Results

An **XGBoost Classifier** was used for this classification task due to its high performance and efficiency. The features were first scaled to a range between -1 and 1 to ensure uniform influence on the model.

* **Model**: `XGBClassifier`
* **Preprocessing**: Features were scaled using `MinMaxScaler`.
* **Performance**: The model was trained and evaluated on a standard 80/20 train-test split, achieving the following result on the test set:

| Metric   | Score      |
| :------- | :--------- |
| **Accuracy** | **94.87%** |

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**: For loading and manipulating the dataset.
* **NumPy**: For numerical operations.
* **Scikit-learn**: For data preprocessing (`MinMaxScaler`) and model evaluation (`accuracy_score`).
* **XGBoost**: For the core classification model (`XGBClassifier`).
* **Jupyter Notebook**: For interactive development and analysis.

---

## ⚙️ Installation

To get a local copy up and running, follow these simple steps.

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/Laabh-Gupta/Parkinsons-disease-Detection.git](https://github.com/Laabh-Gupta/Parkinsons-disease-Detection.git)
    ```
2.  **Navigate to the project directory:**
    ```sh
    cd Parkinsons-disease-Detection
    ```
3.  **Install the required libraries:**
    ```sh
    pip install pandas numpy scikit-learn xgboost jupyterlab
    ```

---

## 🚀 Usage

To explore the project, you can run the Jupyter Notebook.

1.  Start Jupyter Lab:
    ```sh
    jupyter lab
    ```
2.  Open `Parkinsons_disease_detection.ipynb` to view the code, analysis, and results.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📫 Contact

Laabh Gupta - reachlaabhgupta@gmail.com - [https://www.linkedin.com/in/laabhgupta/](https://www.linkedin.com/in/laabhgupta/)

Project Link: [https://github.com/Laabh-Gupta/Parkinsons-disease-Detection](https://github.com/Laabh-Gupta/Parkinsons-disease-Detection)