# 🏠 House Price Predictor – An MLOps Learning Project

Welcome to the **House Price Predictor** project! This is a real-world, end-to-end MLOps use case designed to help you master the art of building and operationalizing machine learning pipelines.

You'll start from raw data and move through data preprocessing, feature engineering, experimentation, model tracking with MLflow, and optionally using Jupyter for exploration – all while applying industry-grade tooling.

> 🚀 **Want to master MLOps from scratch?**  
Check out the [MLOps Bootcamp at School of DevOps](https://schoolofdevops.com) to level up your skills.

---

## 📦 Project Structure

```
house-price-predictor/
├── configs/                # YAML-based configuration for models
├── data/                   # Raw and processed datasets
├── deployment/
│   └── mlflow/             # Docker Compose setup for MLflow
├── models/                 # Trained models and preprocessors
├── notebooks/              # Optional Jupyter notebooks for experimentation
├── src/
│   ├── data/               # Data cleaning and preprocessing scripts
│   ├── features/           # Feature engineering pipeline
│   ├── models/             # Model training and evaluation
├── requirements.txt        # Python dependencies
└── README.md               # You’re here!
```

---

## 🛠️ Setting up Learning/Development Environment

To begin, ensure the following tools are installed on your system:

- [Python 3.11](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)
- [Visual Studio Code](https://code.visualstudio.com/) or your preferred editor
- [UV – Python package and environment manager](https://github.com/astral-sh/uv)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) **or** [Podman Desktop](https://podman-desktop.io/)

---

## 🚀 Preparing Your Environment

1. **Fork this repo** on GitHub.

2. **Clone your forked copy:**

   ```bash
   # Replace xxxxxx with your GitHub username or org
   git clone https://github.com/xxxxxx/house-price-predictor.git
   cd house-price-predictor
   ```

3. **Crear ambientes en Python usando**:

1 . `requirements.txt` con `pip`
2  . `environment.yml` con `conda`

---

## 🧪 1. Crear un ambiente con `requirements.txt` (usando `pip` y `venv`)

El archivo `requirements.txt` contiene una lista de paquetes con sus versiones, como esta:

```txt
# Python version: 3.10.12
pandas==2.1.3
numpy==1.24.4
scikit-learn==1.3.2
...
```

### ✅ Pasos para crear el ambiente con `pip`:

```bash
# 1. Crear un entorno virtual (por ejemplo, llamado 'venv')
python -m venv venv

# 2. Activar el entorno:
# En Windows:
venv\Scripts\activate

# En macOS/Linux:
source venv/bin/activate

# 3. Instalar los paquetes desde requirements.txt
pip install -r requirements.txt

# 4. Verificar que los paquetes estén instalados
pip list

# (Opcional) 5. Desactivar el entorno cuando termines
deactivate
```

---

## 🧬 2. Crear un ambiente con `environment.yml` (usando `conda`)

El archivo `environment.yml` se usa con `conda` o `mamba` y tiene un formato como:

```yaml
name: my_env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10.12
  - pandas=2.1.3
  - numpy=1.24.4
  - scikit-learn=1.3.2
  ...
```

### ✅ Pasos para crear el ambiente con `conda`:

```bash
# 1. Crear el entorno a partir del archivo environment.yml
conda env create -f environment.yml

# 2. Activar el entorno (usa el nombre definido en el archivo, por ejemplo 'my_env')
conda activate my_env

# 3. Verificar que los paquetes estén instalados
conda list

# (Opcional) 4. Desactivar el entorno cuando termines
conda deactivate
```

### 🔄 Si el entorno ya existe y quieres actualizarlo:

```bash
conda env update -f environment.yml --prune
```

> `--prune` elimina paquetes que ya no están listados en el `.yml`

---

## ⚠️ Recomendaciones

* Usa `conda` si trabajas con paquetes científicos pesados como `tensorflow`, `xgboost` o `mlflow` (mejor manejo de dependencias).
* Usa `pip` si necesitas algo más liviano o en entornos donde no puedes instalar `conda`.
* Puedes combinar ambos: usar `conda` para el ambiente base y `pip` para instalar lo que falte.


## Uso ``Script env_builder.py``

```bash
# Ejecuta el script
python env_builder.py
```

### 🔧 Qué hace este script

* Detecta versiones de los paquetes instalados.
* Genera:

  * `requirements.txt` para `pip`
  * `environment.yml` para `conda`
* Usa los canales `defaults` y `conda-forge` (puedes modificar esto).
* Asigna el nombre del entorno como `"my_env"` (puedes cambiarlo en `env_name`).



---

## 📊 Setup MLflow for Experiment Tracking

To track experiments and model runs:

```bash
cd deployment/mlflow
docker compose -f docker-compose.yaml up -d
docker compose ps
```

> 🐧 **Using Podman?** Use this instead:

```bash
podman compose -f mlflow-docker-compose.yml up -d
podman compose ps
```

Access the MLflow UI at [http://localhost:5555](http://localhost:5555)

---

## 📒 Using JupyterLab (Optional)

If you prefer an interactive experience, launch JupyterLab with:

```bash
python -m jupyterlab
```

---

## 🔁 Model Workflow

### 🧹 Step 1: Data Processing

Clean and preprocess the raw housing dataset:

```bash
python src/data/run_processing.py   --input data/raw/house_data.csv   --output data/processed/cleaned_house_data.csv
```

---

### 🧠 Step 2: Feature Engineering

Apply transformations and generate features:

```bash
python src/features/engineer.py   --input data/processed/cleaned_house_data.csv   --output data/processed/featured_house_data.csv   --preprocessor models/trained/preprocessor.pkl
```

---

### 📈 Step 3: Modeling & Experimentation

Train your model and log everything to MLflow:

```bash
python src/models/train_model.py   --config configs/best_model_config.yaml  --data data/processed/featured_house_data.csv   --models-dir models   --mlflow-tracking-uri http://localhost:5555

python src/models/train_model.py --config-path models/best_model_config.yaml --data-dir data/processed --models-dir models --mlflow-tracking-uri http://localhost:5555
```

---


## Building FastAPI and Streamlit 

The code for both the apps are available in `src/api` and `streamlit_app` already. To build and launch these apps 

  * Add a  `Dockerfile` in the root of the source code for building FastAPI  
  * Add `streamlit_app/Dockerfile` to package and build the Streamlit app  
  * Add `docker-compose.yaml` in the root path to launch both these apps. be sure to provide `API_URL=http://fastapi:8000` in the streamlit app's environment. 


Once you have launched both the apps, you should be able to access streamlit web ui and make predictions. 

You could also test predictions with FastAPI directly using 

```
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "sqft": 1500,
  "bedrooms": 3,
  "bathrooms": 2,
  "location": "suburban",
  "year_built": 2000,
  "condition": fair
}'

```

Be sure to replace `http://localhost:8000/predict` with actual endpoint based on where its running. 


## 🧠 Learn More About MLOps

This project is part of the [**MLOps Bootcamp**](https://schoolofdevops.com) at School of DevOps, where you'll learn how to:

- Build and track ML pipelines
- Containerize and deploy models
- Automate training workflows using GitHub Actions or Argo Workflows
- Apply DevOps principles to Machine Learning systems

🔗 [Get Started with MLOps →](https://schoolofdevops.com)

---

## 🤝 Contributing

We welcome contributions, issues, and suggestions to make this project even better. Feel free to fork, explore, and raise PRs!

---

Happy Learning!  
— Team **School of DevOps**
