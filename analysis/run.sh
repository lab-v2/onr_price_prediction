#!/bin/bash

python generate_predictions.py copper -f VOLZA -c    &   
# python generate_predictions.py copper -f VOLZA &   
python generate_predictions.py copper -f OIL -c    &   
# python generate_predictions.py copper -f OIL &   
python generate_predictions.py copper -f PRICE -c &
# python generate_predictions.py copper -f PRICE &
python generate_predictions.py copper -f OIL PRICE -c    & 
# python generate_predictions.py copper -f OIL PRICE    &
python generate_predictions.py copper -f VOLZA PRICE -c    & 
# python generate_predictions.py copper -f VOLZA PRICE    &
python generate_predictions.py copper -f ARIMA PRICE -c    & 
# python generate_predictions.py copper -f ARIMA PRICE    & 


# Wait for all background jobs to finish
wait

echo "All scripts have completed."