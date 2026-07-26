# MynFit

MynFit is an AI-powered fashion size recommendation system built with
**FastAPI** for the backend and a static HTML/CSS/JavaScript frontend.
The backend recommends the best clothing size using a shopper's height,
weight, gender, brand, and category.

------------------------------------------------------------------------

# Project Structure

``` text
MynFit/
│
├── backend/
│   ├── app.py
│   ├── recommend.py
│   ├── clustering.py
│   ├── fit_stats.py
│   ├── requirements.txt
│   ├── data/
│   └── venv/                (created locally)
│
└── frontend-stitch/
    ├── mynfit_landing_page/
    ├── mynfit_product_listing_page/
    ├── mynfit_product_detail_page/
    ├── mynfit_recommendation_flow/
    └── shared/
```

------------------------------------------------------------------------

# Recommended Software

  Software       Version
  -------------- ---------------------
  Python         **3.11.9 (64-bit)**
  pip            Latest
  FastAPI        0.110.0
  Uvicorn        0.28.0
  NumPy          **1.26.4**
  Pandas         **2.2.1**
  Scikit-learn   1.4.1.post1

> **Avoid Python 3.13 and NumPy 2.x** unless every dependency has been
> verified compatible.

------------------------------------------------------------------------

# Recommended requirements.txt

``` txt
fastapi==0.110.0
uvicorn==0.28.0

numpy==1.26.4
pandas==2.2.1

scikit-learn==1.4.1.post1
openpyxl==3.1.2

pydantic==2.6.4
python-dotenv==1.0.1

google-generativeai==0.4.1
```

------------------------------------------------------------------------

# Backend Setup

## Step 1 --- Open Terminal

``` powershell
cd backend
```

------------------------------------------------------------------------

## Step 2 --- Check Python Version

``` powershell
python --version
```

Expected:

``` text
Python 3.11.9
```

If another version appears, install Python 3.11.9 and ensure it is first
in PATH.

------------------------------------------------------------------------

## Step 3 --- Remove Existing Virtual Environment (Recommended)

Windows CMD

``` cmd
rmdir /s /q venv
```

PowerShell

``` powershell
Remove-Item -Recurse -Force venv
```

------------------------------------------------------------------------

## Step 4 --- Create New Virtual Environment

``` powershell
python -m venv venv
```

------------------------------------------------------------------------

## Step 5 --- Activate Environment

PowerShell

``` powershell
.\venv\Scripts\Activate.ps1
```

If execution is blocked:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Activate again.

You should now see:

``` text
(venv)
```

------------------------------------------------------------------------

## Step 6 --- Upgrade pip

``` powershell
python -m pip install --upgrade pip
```

------------------------------------------------------------------------

## Step 7 --- Install Dependencies

``` powershell
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Step 8 --- Verify Installation

``` powershell
python --version
python -c "import numpy; print(numpy.__version__)"
python -c "import pandas; print(pandas.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
```

Expected:

``` text
Python 3.11.9
1.26.4
2.2.1
1.4.1.post1
```

------------------------------------------------------------------------

## Step 9 --- Start FastAPI

``` powershell
python -m uvicorn app:app --reload
```

Expected:

``` text
Uvicorn running on http://127.0.0.1:8000
```

Swagger:

    http://127.0.0.1:8000/docs

------------------------------------------------------------------------

# Running the Frontend

Open:

``` text
frontend-stitch/mynfit_landing_page/code.html
```

Keep the backend terminal running.

------------------------------------------------------------------------

# API

**POST** `/mynfit`

Example:

``` json
{
  "height_cm":165,
  "weight_kg":62,
  "gender":"Female",
  "brand":"Zara",
  "category":"Shirt"
}
```

------------------------------------------------------------------------

# Common Problems

## ModuleNotFoundError

``` text
No module named 'uvicorn'
```

Fix:

``` powershell
pip install -r requirements.txt
```

------------------------------------------------------------------------

## NumPy / Pandas Compatibility Error

Example:

``` text
ImportError: numpy.core.multiarray failed to import
ValueError: numpy.dtype size changed
```

Fix:

1.  Delete `venv`
2.  Create a new virtual environment
3.  Install dependencies again

------------------------------------------------------------------------

## Wrong Python Version

Check:

``` powershell
python --version
where python
```

If Python 3.13 appears, install Python 3.11.9 and recreate the virtual
environment.

------------------------------------------------------------------------

## Backend Not Reachable

Verify:

    http://127.0.0.1:8000/docs

If unavailable:

``` powershell
python -m uvicorn app:app --reload
```

------------------------------------------------------------------------

# Git Setup for Contributors

Clone:

``` bash
git clone <repository-url>
cd MynFit
```

Create a feature branch:

``` bash
git switch -c feature-name
```

Commit:

``` bash
git add .
git commit -m "Describe changes"
```

Push:

``` bash
git push -u origin feature-name
```

------------------------------------------------------------------------

# Tech Stack

-   FastAPI
-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   OpenPyXL
-   Google Generative AI
-   HTML
-   CSS
-   JavaScript

------------------------------------------------------------------------

Demo video link:- https://drive.google.com/file/d/1GM0ZyqciJR5fTco3PX6jvrukDs7RiaXD/view?usp=sharing
PPT link:- https://drive.google.com/file/d/1XpaYX2cyTd93lS7b2AXvGlFz8KkTmXlp/view?usp=sharing
