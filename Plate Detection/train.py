import torch
from ultralytics import YOLO

    

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
torch.cuda.empty_cache()
print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
def run():
    torch.multiprocessing.freeze_support()
    #Use the model
    results = model.train(data="Plate Detection\config.yaml", epochs=1, batch=3)  # train the model

if __name__ == '__main__':
    run()    







