from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Synthetic Image Detection Backend is Running!"}
