# Make sure your kaggle API is set up
conda activate 11785_project # change this to your virtual envrionment
mkdir local_data
cd local_data
echo "downloading data from kaggle"
kaggle competitions download -c ubiquant-market-prediction
unzip ubiquant-market-predictions.zip