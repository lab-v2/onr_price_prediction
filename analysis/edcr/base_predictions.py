from utils import eval, spike, data_processing
from constants import (
    TARGET_COLUMN,
    RANDOM_STATE
)

from build_data import get_data
from sklearn.model_selection import train_test_split

COMMODITYS = [
    # 'cobalt', 
    # 'copper', 
    'magnesium', 
    # 'nickel',
    ]
#target_COMMODITY = "copper"
# target_COMMODITY = "cobalt"
WINDOW_SIZE = 20

pre_features = []
pre_labels = []
tar_features = []
tar_labels = []

for target_COMMODITY in COMMODITYS:

    # for COMMODITY in COMMODITYS:
    VOLZA_FILE_PATH = f"../volza/{target_COMMODITY}/{target_COMMODITY}.csv"
    PRICE_FILE_PATH = f"../volza/{target_COMMODITY}/{target_COMMODITY}_prices.csv"

    # Get the data
    data = get_data(VOLZA_FILE_PATH, PRICE_FILE_PATH, window_size=WINDOW_SIZE, center=False)

    # Add spike column
    data['spikes_streaming'] = spike.detect_spikes_shift(data, 'Price', window_size=WINDOW_SIZE)
        

    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    sampler = RandomOverSampler
    keyword = 'shift_new'

    # Evaluate and create pre-trained model
    output_file_path = f'{target_COMMODITY}_{keyword}_{WINDOW_SIZE}/test/results_test.csv'
    pred_file_path = f'{target_COMMODITY}_{keyword}_{WINDOW_SIZE}/test/predictions/test'
    model_path = f'{target_COMMODITY}_{keyword}_{WINDOW_SIZE}/best_model'

    print(pred_file_path)


    # Prepare price data
    X_price, y_price = data_processing.prepare_features_and_target(data, TARGET_COLUMN, 'spikes_streaming')
    # X_price = np.array(tar_features)
    # y_price = np.array(tar_labels)

    # Split price data
    X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(X_price, y_price, test_size=0.4, shuffle=False)
    X_train_price, y_train_price = RandomOverSampler(random_state=RANDOM_STATE).fit_resample(X_train_price, y_train_price)

    # Balancing
    X_train_price, y_train_price = sampler(random_state=RANDOM_STATE).fit_resample(X_train_price, y_train_price)

    # Scaling
    X_train_price, X_test_price = data_processing.scale_features_no_val(X_train_price, X_test_price)

    # Sequence making
    X_train_price, y_train_price = data_processing.create_sequences(X_train_price, y_train_price, WINDOW_SIZE)
    X_test_price, y_test_price = data_processing.create_sequences(X_test_price, y_test_price, WINDOW_SIZE)
    # X_val_price, y_val_price = data_processing.create_sequences(X_val_price, y_val_price, WINDOW_SIZE)

    # Use this for Bowen's method
    # X_train_price = np.expand_dims(X_train_price, axis = 2)
    # X_test_price = np.expand_dims(X_test_price, axis = 2)

    results_df  = eval.evaluate_all(X_train_price, y_train_price, None, None, X_test_price, y_test_price, output_file_path, pred_file_path, model_path, False, val=False)







