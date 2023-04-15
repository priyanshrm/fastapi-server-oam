from pydantic import BaseModel, EmailStr

class RegisterUser(BaseModel):
    username: str
    password: str
    name: str
    email: EmailStr
    bio: str

class LoginUser(BaseModel):
    username: str
    password: str

class PostModel(BaseModel):
    title: str
    desc: str
    file_id: str

