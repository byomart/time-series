from scripts import train, test, utils, data, preprocess
import logging, yaml, torch


logging.basicConfig(filename='logs/log.log', 
                    level='INFO')


# load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# load csv data
df = data.load_data_from_config(config)
logging.info(df.head())

# Dataset preprocessing
processor = preprocess.SunspotDataProcessor(sequence_size=10, batch_size=32) # class instance
x_train, y_train, x_test, y_test, test_dates = processor.preprocess_and_generate_sequences(df)
# DataLoaders configuration
train_loader, test_loader = processor.setup_data_loaders(x_train, y_train, x_test, y_test)


# MODEL
# parameters
model_path = config['paths']['model']
epochs = config['model parameters']['epochs']
lr = config['model parameters']['lr']

# cuda if possible
device = "cuda" if torch.cuda.is_available() else "cpu"

# # train
# model = utils.TransformerModel()
# trained_model = train.train_model(model, train_loader, test_loader, epochs, lr, patience=5, device="cuda", model_path)

# # load trained model
model = utils.TransformerModel().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# # test
scaler = processor.scaler
scaled_predictions, scaled_y_test, rmse = test.evaluate_model(model, test_loader, scaler, y_test, device)
logging.info(f"Score (RMSE): {rmse:.4f}")

# draw predictions
zoom = config['images']['zoom']
utils.draw_predictions(zoom, test_dates, scaled_predictions, scaled_y_test, rmse)
