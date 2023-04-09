from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db_connection import collections


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def root():
    return {"message to users": "This is a fastAPI server"}

@app.post("/register")
def register(username: str, password: str):
    try:
        # save user information to MongoDB
        collections.userCollection.insert_one({"username": username, "password": password})
    except Exception as e:
        # return error message if insertion fails
        return {"message": f"Error registering user: {str(e)}"}

    return {"message": "User registered successfully"}