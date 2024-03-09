#!/bin/bash

python generate_predictions.py magnesium -f VOLZA -c    &   
# python generate_predictions.py magnesium -f VOLZA &   
python generate_predictions.py magnesium -f OIL -c    &   
# python generate_predictions.py magnesium -f OIL &   
python generate_predictions.py magnesium -f PRICE -c &
# python generate_predictions.py magnesium -f PRICE &
python generate_predictions.py magnesium -f OIL PRICE -c    & 
# python generate_predictions.py magnesium -f OIL PRICE    &
python generate_predictions.py magnesium -f VOLZA PRICE -c    & 
# python generate_predictions.py magnesium -f VOLZA PRICE    &
python generate_predictions.py magnesium -f ARIMA PRICE -c    & 
# python generate_predictions.py magnesium -f ARIMA PRICE    & 


# Wait for all background jobs to finish
wait

echo "All scripts have completed."