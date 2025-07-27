# Floor1
This is the first floor.

## Installation
Support only python<=3.12. Install from beta version 
```shell
pip install -U git+https://github.com/asahi417/floor1.git@main
```


## Usage
### API Server
```shell
uvicorn app_server:app --host 0.0.0.0 --port 4444
```
Access API viewer http://0.0.0.0:4444/docs.


### API Client
```shell
export ENDPOINT=http://0.0.0.0:4444
uvicorn app_client:app --host 0.0.0.0 --port 4000
```
Access API viewer http://0.0.0.0:4000/docs.

### Frontend
```shell
export ENDPOINT="0.0.0.0:4444"
export P_PROMPT="creative, inspiring, geometric, blooming, surrealistic, HQ"
export N_PROMPT="low quality, blur"
python ui_graphic.py
```
