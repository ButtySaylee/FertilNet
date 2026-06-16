\# FertilNet



A Hierarchical Knowledge Distillation Framework for Lightweight Real-Time Soil Fertility Prediction.



\## Overview



FertilNet is a machine learning framework that predicts soil fertility using physicochemical soil attributes while remaining lightweight enough for deployment on resource-constrained devices.



The project introduces a Teacher–Assistant–Student (TAS) knowledge distillation pipeline that compresses a large neural network into a compact model suitable for real-time inference.



\## Architecture



Teacher MLP (5 × 256)

&#x20;       ↓

Assistant MLP (3 × 128)

&#x20;       ↓

Student MLP (2 × 64)



\## Features



\- Soil fertility prediction

\- Hierarchical Knowledge Distillation

\- Teacher–Assistant–Student architecture

\- ONNX model export

\- Edge AI deployment support

\- Random Forest baseline comparison

\- Model compression for real-time inference



\## Dataset Features



The model uses 12 soil attributes:



\- Nitrogen (N)

\- Phosphorus (P)

\- Potassium (K)

\- pH

\- Electrical Conductivity (EC)

\- Organic Carbon (OC)

\- Sulfur (S)

\- Zinc (Zn)

\- Iron (Fe)

\- Copper (Cu)

\- Manganese (Mn)

\- Boron (B)



\## Technologies



\- Python

\- PyTorch

\- Scikit-Learn

\- XGBoost

\- ONNX

\- NumPy

\- Pandas

\- Matplotlib



\## Project Structure



```text

FertilNet/

│

├── data/

│   └── soil\_data.csv

│

├── images/

│

├── outputs/

│

├── main.py

├── requirements.txt

├── README.md

└── .gitignore

```



\## Installation



```bash

git clone https://github.com/ButtySaylee/FertilNet.git



cd FertilNet



pip install -r requirements.txt

```



\## Run



```bash

python main.py

```



\## Author



Butty Saylee



BTech Software Engineering  

Delhi Technological University



\## License



MIT License

