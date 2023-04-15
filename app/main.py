import io
import json 
from bson import json_util
import uuid
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from db_connection import collections
from models import models
from bson.objectid import ObjectId
from gridfs import GridFS, GridOut
from PIL import Image
from cnn import cnn_model

app = FastAPI()
fs = GridFS(collections.images_db)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ROUTES

@app.get('/')
def root():
    return {"message to users": "This is a fastAPI server"}

# @app.post("/register/artist")
# def register_artist(payload: models.RegisterUser):
#     return register(payload, collections.artistsCollection)

# @app.post('/login/artist')
# def login_artist(payload: models.LoginUser):
#     return login(payload, collections.artistsCollection)

# @app.post('/register/collector')
# def register_collector(payload: models.RegisterUser):
#     return register(payload, collections.collectorCollection)

# @app.post('/login/collector')
# def login_collector(payload: models.LoginUser):
#     return login(payload, collections.collectorCollection)

# UPLOAD IMAGE

@app.post("/image/upload_image")
async def upload_image(image: bytes = File(...)):
    db = collections.encodingCollection
    flag  = False
    # encode the image and add to encoding collections
    try:
        encoding  = cnn_model.getStrEncoding(image)
        for file in db.find():
            similarity_score = cnn_model.cosine_similarity(file['encoding'], encoding)
            if similarity_score > 0.50 or similarity_score < -0.50:
                flag = True
                return {"message": f"Similar image already exists!", "similarity_score": similarity_score}
        if not flag:
            db.insert_one({"encoding": encoding, "length":len(encoding)})
    except Exception as e:
        return {"message": f"Error in encoding: {str(e)}"}
    file_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}.jpg"
    file_id = fs.put(image, filename=file_name)
    return JSONResponse({"file_id": str(file_id)})

@app.get("/image/{file_id}")
async def get_image(file_id: str):
    file_id = ObjectId(file_id)
    file = fs.get(file_id)
    return StreamingResponse(io.BytesIO(file.read()), media_type="image/jpeg")

@app.get("/images")
async def get_images():
    images = []
    for image in fs.find():
        image_data = {
            'filename': image.filename,
            'content_type': image.content_type,
            'data': str(image._id),
            # 'title': image.title,
            # 'desc': image.desc
        }
        images.append(image_data)
    return images

@app.get("/encodings")
async def get_encodings():
    encodings = []
    for file in collections.encodingCollection.find():
        encoding = file['encoding']
        codec = 'iso-8859-1'
        try:
            decoded_string = encoding.decode(codec)
            encodings.append({"encoded_string": f"{decoded_string[:128]}...", "length":file['length']})
        except UnicodeDecodeError:
            return {f"Could not decode byte string using {codec}"}
    return {"encodings": encodings}

# POSTS

# @app.post("/post/upload_post")
# async def upload_post(post: dict):
#     db = collections.postsCollection
#     try:
#         db.insert_one({"id":db.count_documents({}) + 1, "title":post["title"], "desc":post["desc"], "imgUrl":post["imgUrl"]})
#         return {"imgUrl": post["imgUrl"]}
#     except Exception as e:
#         return {"message": f"Error in uploading post: {str(e)}"}
@app.post("/post/upload_post")
async def upload_post(form_data: models.PostModel):
    db = collections.postsCollection
    if form_data.file_id is None:
        return {"message": f"Couldn't post the image online!"}
    try:
        db.insert_one({
            "title": form_data.title,
            "desc": form_data.desc,
            "file_id": form_data.file_id,
        })
        return {"data": form_data.file_id}
    except Exception as e:
        return {"error": f"Error in creating post: {str(e)}"}


@app.get("/post/get_posts")
async def get_all_posts():
    db = collections.postsCollection
    try:
        # Retrieve all documents from the MongoDB collection
        cursor = db.find({})
        
        # Convert the MongoDB Cursor object to a list of dictionaries
        posts = [post for post in cursor]
        
        # Serialize the list of dictionaries as a JSON string, converting ObjectId to string
        json_string = json.dumps(posts, default=json_util.default)
        
        return {"data": json.loads(json_string)}
    except Exception as e:
        return {"error": f"Error in getting posts: {str(e)}"}

# FUNCTIONS

def login(payload, db):
    user = db.find_one({"username": payload.username}) or db.find_one({"email": payload.username})
    if user is None:
        raise HTTPException(status_code=401, detail='Invalid username')
    if user['password'] == payload.password:
        return {'id': str(user['_id'])}
    else:
        raise HTTPException(status_code=401, detail='Invalid password')
    
def register(payload, db):
    if db.find_one({"username": payload.username}):
        raise HTTPException(status_code=400, detail='Username already exists')
    if db.find_one({"email": payload.email}):
        raise HTTPException(status_code=400, detail='Email already exists')
    try:
        db.insert_one(payload.dict())
    except Exception as e:
        return {"message": f"Error registering user: {str(e)}"}

    return {"message": "User registered successfully"}
