# mini_test.py
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def read_root():
    return {"ok":"up"}
if __name__=="__main__":
    import uvicorn
    uvicorn.run("mini_test:app", host="127.0.0.1", port=8000)
