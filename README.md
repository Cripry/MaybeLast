# Wind Turbine Energy prediction

A monorepo with a project created using Next js where user can upload data, and get prediction for the next 6 hours about Wind Energy prediction.

## Prerequisites

Ensure you have the following installed on your machine:
- [Docker](https://www.docker.com/products/docker-desktop)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Getting Started

Clone the repository to your local machine:

```bash
git clone https://github.com/Cripry/WindTurbineEnergy.git
cd smartproject
```

To get the project up and running, execute the following command:
`docker-compose up --build`


Wait for the services to start up, and once the Next.js application is running, navigate to `http://127.0.0.1:3000` in your web browser.


## Usage

#### Uploading Data
1) Go to the "Data" page via the navigation menu.
2) Click on the "Upload" button and select the CSV file you wish to upload.
3) Once the file is selected, click on the "Submit" button.
4) A pop-up will appear indicating that the file has been uploaded successfully. Click "OK" to dismiss the pop-up.
5) Refresh the page to view the uploaded data.

#### Viewing Predicted Data
1) Navigate to the "Dashboard" page via the navigation menu.
2) On this page, you'll be able to see the forecasted data for the next 6 hours.



## Development

### Model Development and Exploration

The project contains several Jupyter notebooks documenting the model development process:

- In the directory containing Jupyter notebooks, the `Final.ipynb` notebook holds the comprehensive code for running the model.
- The `Task V3.ipynb` notebook, also located in the same directory, is where data exploration and initial model testing were performed.
  
### Model Fine-tuning

- In the `Fine Tuning` folder, various notebooks are available where different model architectures and hyperparameters were experimented with.
- The `Arch*` folders within the `Fine Tuning` directory contain results obtained after tuning.

Feel free to explore these notebooks to understand the evolution of the model and the various approaches tried during the development process.

