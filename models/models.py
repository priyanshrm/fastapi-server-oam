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

