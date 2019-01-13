# match-paper-algo

## Requirements
- python3
- install requirements via ```
pip install -r requirements.txt``` 

## Run
```bash
cd $project_path
export PYTHONPATH="$project_path:$PYTHONPATH"

# train cnn model
python3 mcnn.py

# calculate paper similarity using pre-trained cnn model
python3 test.py
```
