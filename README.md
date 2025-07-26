# Floor1
This is the first floor.

## Installation
Support only python<=3.12. Install from beta version 
```shell
pip install -U git+https://github.com/asahi417/floor1.git@main
```


## Usage
### Backend
```shell
uvicorn app:app --host 0.0.0.0 --port 4444
```

Access API viewer http://0.0.0.0:4444/docs.

### Frontend
```shell
export ENDPOINT="0.0.0.0:4444"
export P_PROMPT="creative, inspiring, geometric, blooming, surrealistic, HQ"
export N_PROMPT="low quality, blur"
python interface_graphic.py
```
