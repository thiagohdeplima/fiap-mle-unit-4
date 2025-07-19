import uvicorn

from fastapi import FastAPI

app = FastAPI()

def main():
  uvicorn.run(app, host="0.0.0.0", port=8000)
