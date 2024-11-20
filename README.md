# Analiza i wizualizacja danych pomiarowych

This repository contains course materials for W10100-SI0001C.  

To install the required programs, follow [IDE Configuration Guide](/docs/ide_configuration.md).

## 📥 Cloning the Repository 

Follow these steps to clone the repository to your local machine:

1. Create a new directory for the project:
```bash
mkdir -p /path/to/your/folder
```
This command creates a new directory path. The -p flag creates parent directories as needed and doesn't raise an error if the directory already exists.

2. Navigate to that directory:
```bash
cd /path/to/your/folder
```
The cd command changes your current working directory to the newly created folder.

3. Clone the repository into the current folder:
```bash
git clone https://github.com/w10k57-education/data-science-ips.git .
```
This downloads all repository files into your current directory.
> **Note:** The trailing dot (`.`) after the repository URL tells Git to clone directly into the current directory instead of creating a new subdirectory.

## 🛠️ Environment Setup 
An `Anaconda Distribution` or `Miniconda` is required. Once downloaded, start with creating a virtual environment using:

```Bash
conda env create --name dsips_py310 -f environment.yml
```
This command creates a new conda environment named ainv_py310 with all the dependencies specified in the environment.yml file.

Before working on the code, make sure to activate the environment using:

```Bash
conda activate dsips_py310
```

## 📂 Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── examples           <- Examples to work with during the course
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── tasks              <- A folder to store task-related code. It is recommended to use subfolders.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ai_inv and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── environment.yml   <- The requirements file for reproducing the analysis environment.
│
├── setup.cfg          <- Configuration file for flake8
│
└── src                <- Package files
```
--------

